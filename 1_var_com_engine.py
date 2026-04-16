import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import io

# Load environment variables from .env
load_dotenv()

# ==========================================
# 1. LANGGRAPH STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    df: pd.DataFrame
    hierarchy_cols: List[str]
    has_variance_col: bool
    variance_col: str
    base_scenario: str
    compare_scenario: str
    path_trace: List[str]
    final_summary: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Branched Drill-Down: Loops through primary categories and drills down to the final reason."""
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)
    
    path_trace = []
    
    if not hierarchy:
        return {"path_trace": ["⚠️ Error: No hierarchy columns selected."]}
        
    # Determine the target column for calculation
    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns:
            return {"path_trace": [f"⚠️ Error: Target column '{target_col}' not found."]}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns:
             return {"path_trace": ["⚠️ Error: Scenario columns not found."]}
        
        # Calculate dynamic variance
        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    # Calculate global total variance in Millions
    total_variance = df[target_col].fillna(0).sum()
    path_trace.append(f"**Overall Total Variance: {total_variance / 1e6:,.2f}M**\n")

    first_level = hierarchy[0]
    
    # Get the Top 5 primary categories from the first level of the hierarchy
    first_level_grouped = df.groupby(first_level)[target_col].sum()
    top_primary_categories = first_level_grouped.reindex(first_level_grouped.abs().sort_values(ascending=False).index).head(5)

    # Loop through each primary category (e.g., Region A, then Region B)
    for primary_cat, primary_val in top_primary_categories.items():
        path_trace.append(f"=========================================")
        path_trace.append(f"🔹 **Primary Category: '{primary_cat}'** (Total: {primary_val / 1e6:,.2f}M)")
        path_trace.append(f"=========================================\n")
        
        current_df = df[df[first_level] == primary_cat]
        
        # If there is only one level selected, just list its top drivers
        if len(hierarchy) == 1:
             path_trace.append(f"--- **FINAL LEVEL: Top 5 Reasons in '{first_level}'** ---")
             for idx, (name, val) in enumerate(top_primary_categories.items(), 1):
                 path_trace.append(f"  {idx}. '{name}': {val / 1e6:,.2f}M")
             continue

        # Drill down through the REMAINING levels of the hierarchy
        for i, level in enumerate(hierarchy[1:]):
            is_last_level = (i == len(hierarchy[1:]) - 1)
            
            if current_df.empty or level not in current_df.columns:
                break
                
            grouped = current_df.groupby(level)[target_col].sum()
            
            if grouped.empty or grouped.isna().all():
                break
                
            # Get Top 5 drivers for this specific branch
            top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
            
            # Explicitly flag the final level for the LLM to extract
            if is_last_level:
                path_trace.append(f"--- **FINAL LEVEL: Top 5 Reasons in '{level}' (Inside {primary_cat})** ---")
            else:
                path_trace.append(f"--- **Top 5 Drivers in '{level}' (Inside {primary_cat})** ---")
                
            for idx, (name, val) in enumerate(top_5.items(), 1):
                path_trace.append(f"  {idx}. '{name}': {val / 1e6:,.2f}M")
                
            if is_last_level:
                break # We reached the end, no need to filter further
                
            # Find the #1 node to continue the drill-down
            max_driver = top_5.index[0]
            
            if pd.isna(max_driver):
                break
                
            path_trace.append(f"\n*=> Drilling down into '{max_driver}'...*\n")
            current_df = current_df[current_df[level] == max_driver]
            
        path_trace.append("\n") 
        
    return {"path_trace": path_trace}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates an executive summary focusing on the final reasons."""
    
    if state["path_trace"] and "⚠️ Error" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid data configuration."}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Updated Prompt: Force the LLM to list the final level top 5 for each primary category
    system_prompt = (
        "You are a strict, professional financial data analyst. You are reviewing a branched variance "
        "analysis trace. All values are in Millions (M).\n\n"
        "Provide an Executive Summary formatted EXACTLY as follows:\n"
        "1. A brief 1-2 sentence overall conclusion regarding the total variance.\n"
        "2. A bulleted breakdown for each 'Primary Category' analyzed. Under each Primary Category, "
        "explicitly list the Top 5 reasons/drivers specifically from the 'FINAL LEVEL' identified in the trace, "
        "along with their exact variance amounts.\n\n"
        "Do not add conversational filler. Keep it structured and easy for an executive to read."
    )
    
    trace_text = "\n".join(state["path_trace"])
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Raw Data Trace:\n{trace_text}")
    ]
    
    response = llm.invoke(messages)
    return {"final_summary": response.content}

# ==========================================
# 3. GRAPH COMPILATION
# ==========================================
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("calculate_variance", calculate_variance_node)
    workflow.add_node("synthesize_insight", synthesize_insight_node)
    
    workflow.add_edge(START, "calculate_variance")
    workflow.add_edge("calculate_variance", "synthesize_insight")
    workflow.add_edge("synthesize_insight", END)
    
    return workflow.compile()

# ==========================================
# 4. STREAMLIT UI (SIDEBAR CONFIG)
# ==========================================
st.set_page_config(page_title="Branched Variance Analyzer", layout="wide")

st.title("Branched Root Cause Analyzer")
st.write("Upload your dataset to generate a structured executive summary linking your primary categories directly to their root causes.")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                file_buffer = io.BytesIO(uploaded_file.getvalue())
                xls = pd.ExcelFile(file_buffer)
                sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
                df = pd.read_excel(file_buffer, sheet_name=sheet_name)
            else:
                df = pd.read_csv(uploaded_file)
                
            st.header("2. Configure Metrics")
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            has_variance_col = st.checkbox("Variance column already present?", value=True)
            
            variance_col = ""
            base_scenario = ""
            compare_scenario = ""
            
            if has_variance_col:
                default_var = next((c for c in num_cols if 'var' in c.lower()), num_cols[0] if num_cols else None)
                idx = num_cols.index(default_var) if default_var in num_cols else 0
                variance_col = st.selectbox("Select Variance Column", num_cols, index=idx)
            else:
                st.caption("Calculate: (Base - Compare)")
                base_scenario = st.selectbox("Select Base Scenario (e.g. 2024_ACT)", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario (e.g. 2025_FC2+10)", num_cols)

            st.header("3. Select Hierarchy")
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            hierarchy = st.multiselect(
                "Hierarchy Flow (Left to Right)", 
                options=all_cols, 
                default=cat_cols,
                help="The bot will group by the first column, drill through the middle, and report the Top 5 reasons from the LAST column."
            )

            run_analysis = st.button("Generate Commentary", type="primary", use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")

# MAIN PAGE DISPLAY
if uploaded_file is not None and 'df' in locals():
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    if run_analysis:
        if not hierarchy:
            st.warning("Please select at least one column for the hierarchy.")
        elif has_variance_col and not variance_col:
            st.warning("Please select a variance column.")
        elif not has_variance_col and (not base_scenario or not compare_scenario):
            st.warning("Please select both Base and Compare scenarios.")
        else:
            with st.spinner("Executing branched drill-down analysis..."):
                app_graph = build_graph()
                
                inputs = {
                    "df": df,
                    "hierarchy_cols": hierarchy,
                    "has_variance_col": has_variance_col,
                    "variance_col": variance_col,
                    "base_scenario": base_scenario,
                    "compare_scenario": compare_scenario
                }
                
                result = app_graph.invoke(inputs)
                
                st.markdown("---")
                st.markdown("### 🤖 Executive Summary")
                
                # Render the LLM output with success styling
                if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                    st.error(result["final_summary"])
                else:
                    st.success(result["final_summary"])
                
                st.markdown("### 🧮 Branched Drill-Down Trace")
                for step in result["path_trace"]:
                    if step.startswith("===") or step.startswith("🔹") or step.startswith("**Overall"):
                        st.markdown(step)
                    elif step.startswith("---"):
                        st.markdown(step)
                    elif step.startswith("*=>"):
                        st.caption(step)
                    elif step.strip() == "":
                        st.write("") 
                    else:
                        st.text(step)

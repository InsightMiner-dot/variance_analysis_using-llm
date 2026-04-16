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
    variance_col: str       # Used if variance is pre-calculated
    base_scenario: str      # Used if calculating variance
    compare_scenario: str   # Used if calculating variance
    path_trace: List[str]
    final_summary: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic Pandas calculation tracking the top drivers (Values in Millions)"""
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)
    
    path_trace = []
    current_df = df.copy()
    
    # Determine the target column for calculation
    if has_var:
        target_col = state.get("variance_col")
        if target_col not in current_df.columns:
            return {"path_trace": [f"⚠️ Error: Target column '{target_col}' not found."]}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in current_df.columns or comp_col not in current_df.columns:
             return {"path_trace": ["⚠️ Error: Scenario columns not found."]}
        
        # Calculate dynamic variance
        target_col = "Calculated_Variance"
        current_df[target_col] = current_df[base_col] - current_df[comp_col]

    # Calculate base total in Millions
    total_variance = current_df[target_col].fillna(0).sum()
    path_trace.append(f"**Total Variance: {total_variance / 1e6:,.2f}M**\n")

    # Drill down calculation left-to-right
    for level in hierarchy:
        if current_df.empty or level not in current_df.columns:
            break
            
        grouped = current_df.groupby(level)[target_col].sum()
        
        if grouped.empty or grouped.isna().all():
            break
            
        # Get Top 5 drivers based on absolute impact
        top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
        
        path_trace.append(f"--- **Top Drivers at Level: '{level}'** ---")
        for idx, (name, val) in enumerate(top_5.items(), 1):
            # Format in millions (M)
            path_trace.append(f"  {idx}. '{name}': {val / 1e6:,.2f}M")
            
        # Find the #1 node to continue the drill-down
        max_driver = top_5.index[0]
        
        if pd.isna(max_driver):
            break
            
        path_trace.append(f"\n*=> Drilling down into '{max_driver}' (highest variance) for the next level...*\n")
        
        # Filter dataframe to only the top driver for the next level
        current_df = current_df[current_df[level] == max_driver]
        
    return {"path_trace": path_trace}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates the one-liner summary"""
    
    if state["path_trace"] and "⚠️ Error" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid data configuration."}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    system_prompt = (
        "You are a strict, professional financial data analyst. You are reviewing a data drill-down "
        "that highlights the top variance drivers at each hierarchical level. All values are in Millions (M). "
        "Translate the provided data trace into a single, professional, easily readable sentence "
        "explaining the primary root cause (the 'why') of the overall variance to an executive, "
        "tracing it through the hierarchy. Do not add conversational filler."
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
st.set_page_config(page_title="Variance Analyzer", layout="wide")

st.title("Root Cause Variance Analyzer")
st.write("Upload your dataset in the sidebar to begin. Values will be analyzed and displayed in Millions (M).")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Handle Excel Sheet Selection
            if uploaded_file.name.endswith('.xlsx'):
                # Read file into memory buffer to avoid Streamlit file pointer reset issues
                file_buffer = io.BytesIO(uploaded_file.getvalue())
                xls = pd.ExcelFile(file_buffer)
                sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
                df = pd.read_excel(file_buffer, sheet_name=sheet_name)
            else:
                df = pd.read_csv(uploaded_file)
                
            st.header("2. Configure Metrics")
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Variance Checkbox Logic
            has_variance_col = st.checkbox("Variance column already present?", value=True)
            
            variance_col = ""
            base_scenario = ""
            compare_scenario = ""
            
            if has_variance_col:
                # Pre-select a column if it has 'var' in the name, otherwise pick first
                default_var = next((c for c in num_cols if 'var' in c.lower()), num_cols[0] if num_cols else None)
                idx = num_cols.index(default_var) if default_var in num_cols else 0
                variance_col = st.selectbox("Select Variance Column", num_cols, index=idx)
            else:
                st.caption("Calculate: (Base - Compare)")
                base_scenario = st.selectbox("Select Base Scenario (e.g. 2024_ACT)", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario (e.g. 2025_FC2+10)", num_cols)

            st.header("3. Configure Hierarchy")
            # Auto-detect categorical columns for the default hierarchy
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            hierarchy = st.multiselect(
                "Hierarchy (Left to Right)", 
                options=all_cols, 
                default=cat_cols,
                help="Defaults to all text columns. Reorder or remove as needed."
            )

            # Execution Button in Sidebar
            run_analysis = st.button("Generate Commentary", type="primary", use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")

# MAIN PAGE DISPLAY
if uploaded_file is not None and 'df' in locals():
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    if run_analysis:
        if not hierarchy:
            st.warning("Please select at least one column for the hierarchy in the sidebar.")
        elif has_variance_col and not variance_col:
            st.warning("Please select a variance column.")
        elif not has_variance_col and (not base_scenario or not compare_scenario):
            st.warning("Please select both Base and Compare scenarios.")
        else:
            with st.spinner("Analyzing dataset step-by-step..."):
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
                
                if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                    st.error(result["final_summary"])
                else:
                    st.success(result["final_summary"])
                
                st.markdown("### 🧮 Step-by-Step Top Analysis")
                # Render the text trace cleanly
                for step in result["path_trace"]:
                    if step.startswith("---") or step.startswith("**Total"):
                        st.markdown(step)
                    elif step.startswith("*=>"):
                        st.caption(step)
                    else:
                        st.text(step)

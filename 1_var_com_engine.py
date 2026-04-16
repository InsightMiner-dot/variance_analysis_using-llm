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
        
    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns:
            return {"path_trace": [f"⚠️ Error: Target column '{target_col}' not found."]}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns:
             return {"path_trace": ["⚠️ Error: Scenario columns not found."]}
        
        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    total_variance = df[target_col].fillna(0).sum()
    path_trace.append(f"**Overall Total Variance: {total_variance / 1e6:,.2f}M**\n")

    first_level = hierarchy[0]
    
    first_level_grouped = df.groupby(first_level)[target_col].sum()
    top_primary_categories = first_level_grouped.reindex(first_level_grouped.abs().sort_values(ascending=False).index).head(5)

    for primary_cat, primary_val in top_primary_categories.items():
        path_trace.append(f"=========================================")
        path_trace.append(f"🔹 **Primary Category: '{primary_cat}'** (Total: {primary_val / 1e6:,.2f}M)")
        path_trace.append(f"=========================================\n")
        
        current_df = df[df[first_level] == primary_cat]
        
        if len(hierarchy) == 1:
             path_trace.append(f"--- **FINAL LEVEL: Top 5 Reasons in '{first_level}'** ---")
             for idx, (name, val) in enumerate(top_primary_categories.items(), 1):
                 path_trace.append(f"  {idx}. '{name}': {val / 1e6:,.2f}M")
             continue

        for i, level in enumerate(hierarchy[1:]):
            is_last_level = (i == len(hierarchy[1:]) - 1)
            
            if current_df.empty or level not in current_df.columns:
                break
                
            grouped = current_df.groupby(level)[target_col].sum()
            
            if grouped.empty or grouped.isna().all():
                break
                
            top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
            
            if is_last_level:
                path_trace.append(f"--- **FINAL LEVEL: Top 5 Reasons in '{level}' (Inside {primary_cat})** ---")
            else:
                path_trace.append(f"--- **Top 5 Drivers in '{level}' (Inside {primary_cat})** ---")
                
            for idx, (name, val) in enumerate(top_5.items(), 1):
                path_trace.append(f"  {idx}. '{name}': {val / 1e6:,.2f}M")
                
            if is_last_level:
                break
                
            max_driver = top_5.index[0]
            if pd.isna(max_driver):
                break
                
            path_trace.append(f"\n*=> Drilling down into '{max_driver}'...*\n")
            current_df = current_df[current_df[level] == max_driver]
            
        path_trace.append("\n") 
        
    return {"path_trace": path_trace}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates a highly structured, professional Markdown report."""
    
    if state["path_trace"] and "⚠️ Error" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid data configuration."}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    system_prompt = (
        "You are a Senior Financial Controller presenting a variance report to the executive board. "
        "You are reviewing a branched variance trace. All values are in Millions (M).\n\n"
        "Generate a highly professional, clean Markdown report STRICTLY using this structure:\n\n"
        "### 🎯 Executive Summary\n"
        "[Provide a crisp 2-sentence summary of the total variance and the overarching narrative.]\n\n"
        "### 📊 Key Root Causes by Primary Category\n"
        "[For each Primary Category analyzed in the trace, create a bold sub-bullet. Under it, "
        "list the Top 2-3 most impactful drivers specifically from the 'FINAL LEVEL' identified in the trace, "
        "including their exact monetary impact.]\n\n"
        "### 💡 Strategic Takeaway\n"
        "[Provide one single sentence of analytical insight based on where the largest variances are concentrated.]\n\n"
        "Keep the tone strictly professional, objective, and formatting exceptionally clean."
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
# 4. STREAMLIT UI (PROFESSIONAL LAYOUT)
# ==========================================
st.set_page_config(page_title="Variance Analyzer", layout="wide")

st.title("📊 Root Cause Variance Analyzer")
st.markdown("Automated hierarchical drill-down and executive commentary generation.")

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
                base_scenario = st.selectbox("Select Base Scenario", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario", num_cols)

            st.header("3. Select Hierarchy")
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            hierarchy = st.multiselect(
                "Hierarchy Flow (Left to Right)", 
                options=all_cols, 
                default=cat_cols
            )

            st.markdown("---")
            run_analysis = st.button("🚀 Generate Report", type="primary", use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")

# MAIN PAGE DISPLAY
if uploaded_file is not None and 'df' in locals():
    # Use an expander for data preview so it doesn't take up too much vertical space
    with st.expander("👀 View Raw Data Preview", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    if run_analysis:
        if not hierarchy:
            st.warning("Please select at least one column for the hierarchy.")
        elif has_variance_col and not variance_col:
            st.warning("Please select a variance column.")
        elif not has_variance_col and (not base_scenario or not compare_scenario):
            st.warning("Please select both Base and Compare scenarios.")
        else:
            with st.spinner("Executing branched drill-down and drafting report..."):
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
                
                if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                    st.error(result["final_summary"])
                else:
                    # TABS FOR CLEAN UI SEPARATION
                    tab1, tab2 = st.tabs(["📄 Final Executive Report", "🧮 Calculation Trace"])
                    
                    with tab1:
                        # Display the beautiful Markdown report inside a stylized container
                        st.markdown('<div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; border: 1px solid #e0e0e0;">', unsafe_allow_html=True)
                        st.markdown(result["final_summary"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.write("") # Spacer
                        
                        # Add a download button for the generated report
                        st.download_button(
                            label="📥 Download Report (.md)",
                            data=result["final_summary"],
                            file_name="executive_variance_report.md",
                            mime="text/markdown",
                            use_container_width=False
                        )
                        
                    with tab2:
                        st.info("This is the exact mathematical path the Pandas engine calculated to feed the LLM.")
                        # Render the text trace cleanly
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

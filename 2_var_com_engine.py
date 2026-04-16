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
    
    # Decoupled Data States
    error: str                      
    llm_trace_text: str             
    ui_trace_data: List[Dict]       
    
    final_summary: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Branched Drill-Down producing BOTH an LLM text trace and a structured UI trace."""
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)
    
    if not hierarchy:
        return {"error": "⚠️ Please select at least one hierarchy column."}
        
    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns:
            return {"error": f"⚠️ Target column '{target_col}' not found."}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns:
             return {"error": "⚠️ Scenario columns not found."}
        
        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    llm_trace = []
    ui_data = []

    total_variance = df[target_col].fillna(0).sum()
    llm_trace.append(f"Overall Total Variance: {total_variance / 1e6:,.2f}M\n")

    first_level = hierarchy[0]
    
    first_level_grouped = df.groupby(first_level)[target_col].sum()
    top_primary_categories = first_level_grouped.reindex(first_level_grouped.abs().sort_values(ascending=False).index).head(5)

    for primary_cat, primary_val in top_primary_categories.items():
        branch_data = {
            "primary_category": primary_cat,
            "total_val": primary_val,
            "levels": []
        }
        
        llm_trace.append(f"🔹 Primary Category: '{primary_cat}' (Total: {primary_val / 1e6:,.2f}M)")
        current_df = df[df[first_level] == primary_cat]
        
        if len(hierarchy) == 1:
             level_data = {"level_name": first_level, "is_final": True, "drivers": []}
             llm_trace.append(f"--- FINAL LEVEL: Top 5 Reasons in '{first_level}' ---")
             
             for name, val in top_primary_categories.items():
                 llm_trace.append(f" '{name}': {val / 1e6:,.2f}M")
                 level_data["drivers"].append({"name": name, "val": val})
                 
             branch_data["levels"].append(level_data)
             ui_data.append(branch_data)
             continue

        for i, level in enumerate(hierarchy[1:]):
            is_last_level = (i == len(hierarchy[1:]) - 1)
            
            if current_df.empty or level not in current_df.columns:
                break
                
            grouped = current_df.groupby(level)[target_col].sum()
            if grouped.empty or grouped.isna().all():
                break
                
            top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
            
            level_data = {
                "level_name": level, 
                "is_final": is_last_level, 
                "drivers": [],
                "drill_target": top_5.index[0] if not is_last_level and not pd.isna(top_5.index[0]) else None
            }
            
            llm_trace.append(f"--- {'FINAL LEVEL' if is_last_level else 'Top 5 Drivers'} in '{level}' ---")
                
            for name, val in top_5.items():
                llm_trace.append(f" '{name}': {val / 1e6:,.2f}M")
                level_data["drivers"].append({"name": name, "val": val})
                
            branch_data["levels"].append(level_data)
                
            if is_last_level:
                break 
                
            max_driver = top_5.index[0]
            if pd.isna(max_driver):
                break
                
            llm_trace.append(f"=> Drilling down into '{max_driver}'...\n")
            current_df = current_df[current_df[level] == max_driver]
            
        ui_data.append(branch_data)
        
    return {"llm_trace_text": "\n".join(llm_trace), "ui_trace_data": ui_data, "error": ""}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates an executive summary using the text trace."""
    
    if state.get("error"):
        return {"final_summary": state["error"]}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    system_prompt = (
        "You are a Senior Financial Controller presenting a variance report to the executive board. "
        "Generate a highly professional, clean Markdown report STRICTLY using this structure:\n\n"
        "### 🎯 Executive Summary\n"
        "[Provide a crisp 1-2 sentence overall conclusion regarding the total variance.]\n\n"
        "### 📊 Key Root Causes by Category\n"
        "[For each 'Primary Category' analyzed, create a bold sub-heading. Under it, "
        "explicitly list the Top 5 reasons specifically from the 'FINAL LEVEL' identified in the trace, "
        "including exact variance amounts.]\n\n"
        "Keep the tone strictly professional, objective, and the formatting exceptionally clean."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Raw Data Trace:\n{state['llm_trace_text']}")
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
# 4. STREAMLIT UI (COMPACT LAYOUT)
# ==========================================
st.set_page_config(page_title="Branched Variance Analyzer", layout="wide")

st.title("📊 Root Cause Analyzer")

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
                base_scenario = st.selectbox("Select Base Scenario", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario", num_cols)

            st.header("3. Select Hierarchy")
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            hierarchy = st.multiselect("Hierarchy Flow", options=all_cols, default=cat_cols)

            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")

# MAIN PAGE DISPLAY
if uploaded_file is not None and 'df' in locals():
    with st.expander("👀 View Raw Data Preview", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    if run_analysis:
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
            
            if result.get("error"):
                st.error(result["error"])
            else:
                # ---------------------------------------------------------
                # 1. PROFESSIONAL EXECUTIVE REPORT
                # ---------------------------------------------------------
                st.markdown("---")
                st.markdown(result["final_summary"])
                st.write("") 
                
                # ---------------------------------------------------------
                # 2. UI VISUAL TRACE (COMPACT FLAT LIST)
                # ---------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🧮 Drill-Down Trace")
                st.write("")
                
                # Iterate through the structured UI data and print clean, inline text
                for branch in result["ui_trace_data"]:
                    # Primary Category Header
                    st.markdown(f"**🏢 {branch['primary_category']}** `({branch['total_val'] / 1e6:,.2f}M)`")
                    
                    for lvl in branch["levels"]:
                        is_final = lvl['is_final']
                        
                        # Format drivers into a single compact line separated by pipes
                        driver_strings = [f"{d['name']} `{d['val'] / 1e6:,.2f}M`" for d in lvl["drivers"]]
                        drivers_inline = " &nbsp;|&nbsp; ".join(driver_strings)
                        
                        if is_final:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🎯 **Final Root Causes ({lvl['level_name']}):** {drivers_inline}")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🔸 **{lvl['level_name']}:** {drivers_inline}")
                            if lvl.get("drill_target"):
                                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*↳ Deep diving into {lvl['drill_target']}...*")
                    
                    st.markdown("---")

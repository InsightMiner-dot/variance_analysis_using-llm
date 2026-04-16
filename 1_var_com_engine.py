import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env
load_dotenv()

# ==========================================
# 1. LANGGRAPH STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    df: pd.DataFrame
    path_trace: List[str]
    final_summary: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Automated Pandas calculation highlighting Top 5 drivers per level"""
    df = state["df"].copy()
    path_trace = []
    
    # Auto-detect Hierarchy: Find all text/categorical columns (Left to Right)
    hierarchy = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    # Auto-detect Metric: Find numeric columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        return {"path_trace": ["⚠️ Error: No numeric columns found in the dataset to analyze."]}
    if not hierarchy:
        return {"path_trace": ["⚠️ Error: No text/categorical columns found to form a hierarchy."]}
        
    # Prefer a column with 'var', 'diff', or 'delta' in its name, otherwise pick the first numeric
    var_col = next((col for col in num_cols if any(x in col.lower() for x in ['var', 'diff', 'delta', 'metric'])), num_cols[0])

    path_trace.append(f"🔍 **Auto-detected Hierarchy:** {' ➔ '.join(hierarchy)}")
    path_trace.append(f"📊 **Auto-detected Target Metric:** '{var_col}'\n")
    
    current_df = df.copy()
    total_variance = current_df[var_col].fillna(0).sum()
    path_trace.append(f"**Total {var_col}: {total_variance:,.2f}**\n")

    # Drill down calculation left-to-right
    for level in hierarchy:
        if current_df.empty or level not in current_df.columns:
            break
            
        grouped = current_df.groupby(level)[var_col].sum()
        
        if grouped.empty or grouped.isna().all():
            break
            
        # Get Top 5 drivers based on absolute impact
        top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
        
        path_trace.append(f"--- **Top 5 Drivers at Level: '{level}'** ---")
        for idx, (name, val) in enumerate(top_5.items(), 1):
            path_trace.append(f"  {idx}. '{name}': {val:,.2f}")
            
        # Find the #1 node to continue the drill-down
        max_driver = top_5.index[0]
        
        if pd.isna(max_driver):
            break
            
        path_trace.append(f"\n*=> Drilling down into '{max_driver}' for the next level...*\n")
        
        # Filter dataframe to only the top driver for the next level drill-down
        current_df = current_df[current_df[level] == max_driver]
        
    return {"path_trace": path_trace}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates the one-liner executive summary"""
    
    if state["path_trace"] and "⚠️ Error" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid dataset structure."}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    system_prompt = (
        "You are a strict, professional financial data analyst. You are reviewing an automated data "
        "drill-down that highlights the top 5 variance drivers at each hierarchical level. "
        "Translate the provided data trace into a single, professional, easily readable sentence "
        "explaining the primary root cause of the overall variance to an executive. "
        "Do not add conversational filler."
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
# 4. STREAMLIT UI (AUTOMATED FLOW)
# ==========================================
st.set_page_config(page_title="Automated Variance Analyzer", layout="wide")

st.title("Automated Root Cause Analyzer")
st.write("Upload your dataset. The system will automatically detect the hierarchy and numeric metrics to generate a commentary on the top drivers.")

# 1. Simple File Upload
uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("### Data Preview")
        st.dataframe(df.head(3))

        # 2. Single Action Button
        if st.button("Generate Commentary", type="primary"):
            with st.spinner("Analyzing dataset step-by-step..."):
                app_graph = build_graph()
                
                # Pass only the dataframe; the logic handles the rest
                inputs = {"df": df}
                
                result = app_graph.invoke(inputs)
                
                st.markdown("---")
                st.markdown("### 🤖 Executive Summary")
                
                if "aborted" in result["final_summary"].lower():
                    st.error(result["final_summary"])
                else:
                    st.success(result["final_summary"])
                
                st.markdown("### 🧮 Step-by-Step Top 5 Analysis")
                # Iterate through the trace and render it cleanly
                for step in result["path_trace"]:
                    if step.startswith("---") or step.startswith("🔍") or step.startswith("📊") or step.startswith("**Total"):
                        st.markdown(step)
                    elif step.startswith("*=>"):
                        st.caption(step)
                    else:
                        st.text(step)
                        
    except Exception as e:
        st.error(f"Failed to read or process the file. Error: {e}")

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
    hierarchy_cols: List[str]
    variance_col: str
    path_trace: List[str]
    final_summary: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic Pandas calculation with strict error handling"""
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    var_col = state.get("variance_col", "")
    
    path_trace = []
    
    # Check if the target variance column actually exists
    if var_col not in df.columns:
        return {"path_trace": [f"⚠️ Error: The target metric column '{var_col}' was not found in the dataset."]}
        
    current_df = df.copy()
    
    # Calculate base total, safely handling any NaN values
    total_variance = current_df[var_col].fillna(0).sum()
    path_trace.append(f"Total Variance/Metric: {total_variance:,.2f}")

    # Drill down calculation left-to-right
    for level in hierarchy:
        # Check if the hierarchy column exists
        if level not in current_df.columns:
            path_trace.append(f"⏭️ [Skipped] Level '{level}' does not exist in the dataset.")
            continue
            
        if current_df.empty:
            break
            
        # Group by the current level and sum the variance
        grouped = current_df.groupby(level)[var_col].sum()
        
        # Check if grouping resulted in empty or all-NaN data
        if grouped.empty or grouped.isna().all():
            path_trace.append(f"⚠️ Level '{level}' yielded no valid data.")
            break
            
        # Find the node with the highest absolute impact
        max_driver = grouped.abs().idxmax()
        
        # Ensure the max_driver isn't an invalid/null value
        if pd.isna(max_driver):
            break
            
        actual_var = grouped.loc[max_driver]
        
        path_trace.append(f"Level '{level}': '{max_driver}' drove variance by {actual_var:,.2f}")
        
        # Filter dataframe to only this driver for the next level drill-down
        current_df = current_df[current_df[level] == max_driver]
        
    return {"path_trace": path_trace}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates the one-liner summary"""
    
    # If the pandas node failed to find the target column, skip LLM call
    if state["path_trace"] and "⚠️ Error" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to missing data column."}
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    system_prompt = (
        "You are a strict, professional financial data analyst. You are analyzing a data drill-down "
        "showing the primary drivers of variance in a dataset. "
        "Translate the provided data trace into a single, professional, easily readable sentence "
        "explaining the root cause of the variance. Do not add conversational filler."
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
# 4. STREAMLIT UI (TOP NAV DESIGN)
# ==========================================
st.set_page_config(page_title="Fault-Tolerant Variance Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for the Top Navigation Bar
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        .top-nav {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin-bottom: 2rem;
            font-weight: bold;
            display: flex;
            gap: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-nav">📊 File Upload | ⚙️ Data Mapping | 🚀 Analysis Output</div>', unsafe_allow_html=True)
st.title("Root Cause Variance Analyzer")

# File Upload
uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("### Data Preview")
        st.dataframe(df.head(3))

        st.write("### Configuration")
        cfg_col1, cfg_col2 = st.columns(2)
        
        with cfg_col1:
            all_columns = df.columns.tolist()
            # User types the hierarchy explicitly
            hierarchy = st.multiselect("Select Hierarchy (Left to Right)", options=all_columns)
        
        with cfg_col2:
            num_columns = df.select_dtypes(include=['number']).columns.tolist()
            # User picks the specific variance/metric column
            variance_col = st.selectbox("Target Variance/Metric Column", options=num_columns)

        if st.button("Generate Insight", type="primary", use_container_width=True):
            if not hierarchy or not variance_col:
                st.warning("Please configure the hierarchy and select a target column before running.")
            else:
                with st.spinner("Analyzing variance drivers..."):
                    app_graph = build_graph()
                    
                    inputs = {
                        "df": df,
                        "hierarchy_cols": hierarchy,
                        "variance_col": variance_col
                    }
                    
                    result = app_graph.invoke(inputs)
                    
                    st.markdown("---")
                    st.markdown("### 🤖 Executive Summary")
                    
                    # Display appropriate styling based on whether it hit an error
                    if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                        st.error(result["final_summary"])
                    else:
                        st.success(result["final_summary"])
                    
                    st.markdown("### 🧮 Calculation Trace")
                    for step in result["path_trace"]:
                        st.code(step, language="text")
                        
    except Exception as e:
        st.error(f"Failed to read the file. Please ensure it is a valid CSV or Excel file. Error: {e}")

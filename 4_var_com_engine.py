import io
import os
import json
import sqlite3
import re
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# PPTX IMPORTS FOR PRO STYLING
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

# Load environment variables
load_dotenv()

# ==========================================
# 0. HELPER: MARKDOWN CLEANER
# ==========================================
def clean_markdown_for_ppt(text: str) -> str:
    """Removes LLM markdown artifacts (**, *, #) for clean PowerPoint text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text) # Remove italics
    text = re.sub(r'#(.*?)\n', r'\1\n', text) # Remove Headers
    return text.strip()

# ==========================================
# 1. BACKEND DATABASE SETUP (SQLITE)
# ==========================================
def init_db():
    conn = sqlite3.connect("analysis_history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  filename TEXT,
                  hierarchy TEXT,
                  total_variance TEXT,
                  summary TEXT,
                  feedback INTEGER DEFAULT 0)''')
    try:
        c.execute("ALTER TABLE runs ADD COLUMN feedback INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  prompt TEXT,
                  response TEXT,
                  feedback INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def save_run_to_db(filename: str, hierarchy: List[str], total_variance: str, summary: str) -> int:
    conn = sqlite3.connect("analysis_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO runs (timestamp, filename, hierarchy, total_variance, summary) VALUES (?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, json.dumps(hierarchy), total_variance, summary))
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    return run_id

def update_run_feedback(run_id: int, feedback: int):
    conn = sqlite3.connect("analysis_history.db")
    c = conn.cursor()
    c.execute("UPDATE runs SET feedback = ? WHERE id = ?", (feedback, run_id))
    conn.commit()
    conn.close()

def save_chat_to_db(prompt: str, response: str) -> int:
    conn = sqlite3.connect("analysis_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_logs (timestamp, prompt, response) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prompt, response))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def update_chat_feedback(chat_id: int, feedback: int):
    conn = sqlite3.connect("analysis_history.db")
    c = conn.cursor()
    c.execute("UPDATE chat_logs SET feedback = ? WHERE id = ?", (feedback, chat_id))
    conn.commit()
    conn.close()

def fetch_history_from_db() -> pd.DataFrame:
    conn = sqlite3.connect("analysis_history.db")
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()

def handle_run_feedback_click(run_id: int, score: int):
    update_run_feedback(run_id, score)
    st.session_state.run_feedback_submitted = True
    st.toast("✅ Analysis feedback recorded!")

def handle_chat_feedback_click(chat_id: int, score: int, msg_idx: int):
    update_chat_feedback(chat_id, score)
    st.session_state.chat_feedback_submitted[msg_idx] = True
    st.toast("✅ Chat feedback recorded!")

# ==========================================
# 2. DATA CACHING & PPTX GENERATOR
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_cache_data(file_bytes, file_name, sheet_name=None):
    file_buffer = io.BytesIO(file_bytes)
    if file_name.endswith(".xlsx"):
        return pd.read_excel(file_buffer, sheet_name=sheet_name)
    else:
        return pd.read_csv(file_buffer)

def add_corporate_header(slide, title_text: str, primary_color: RGBColor, accent_color: RGBColor):
    """Helper to draw professional headers on slides."""
    # Header Text
    header = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(1))
    hf = header.text_frame
    hp = hf.paragraphs[0]
    hp.text = title_text
    hp.font.size = Pt(28)
    hp.font.bold = True
    hp.font.color.rgb = primary_color
    
    # Elegant Underline
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.5), Inches(1.1), Inches(9.5), Inches(1.1))
    line.line.color.rgb = accent_color
    line.line.width = Pt(2)

def generate_ppt_deck(total_variance: str, summary: str, tree_data: List[Dict[str, Any]]) -> io.BytesIO:
    prs = Presentation()
    
    # Corporate Colors (Slate & Dark Blue)
    primary_color = RGBColor(14, 43, 92)   # Deep Blue
    accent_color = RGBColor(0, 163, 224)   # Vivid Cyan
    text_color = RGBColor(89, 89, 89)      # Charcoal
    
    blank_slide_layout = prs.slide_layouts[6] # Blank Layout for absolute control
    
    # ---------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # ---------------------------------------------
    slide1 = prs.slides.add_slide(blank_slide_layout)
    
    # Top decorative banner
    banner = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(0.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = primary_color
    banner.line.color.rgb = primary_color
    
    # Main Title
    txBox = slide1.shapes.add_textbox(Inches(1), Inches(2.5), Inches(8), Inches(2))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "Variance Analysis & Root Cause Report"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = primary_color
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    p2 = tf.add_paragraph()
    p2.text = f"Total Impact: {total_variance}\nReport Date: {datetime.now().strftime('%B %d, %Y')}"
    p2.font.size = Pt(18)
    p2.font.color.rgb = text_color
    p2.alignment = PP_ALIGN.CENTER

    # ---------------------------------------------
    # SLIDE 2: EXECUTIVE SUMMARY
    # ---------------------------------------------
    slide2 = prs.slides.add_slide(blank_slide_layout)
    add_corporate_header(slide2, "Executive Summary", primary_color, accent_color)
    
    # Summary Content Box
    content_box = slide2.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(9), Inches(5.5))
    cf = content_box.text_frame
    cf.word_wrap = True
    
    clean_summary = clean_markdown_for_ppt(summary)
    
    for idx, para in enumerate(clean_summary.split('\n')):
        clean_para = para.strip()
        if clean_para:
            p = cf.paragraphs[0] if idx == 0 else cf.add_paragraph()
            
            # Detect if it is a bullet point from the LLM
            if clean_para.startswith('-') or clean_para.startswith('*') or (len(clean_para) > 1 and clean_para[1] == '.'):
                # Clean the prefix and set indent
                p.text = clean_para.lstrip('-*1234567890. ')
                p.level = 1
                p.font.size = Pt(14)
            else:
                p.text = clean_para
                p.level = 0
                p.font.size = Pt(16)
                p.font.bold = (idx == 0) # Bold the opening conclusion statement
                
            p.font.color.rgb = text_color

    # ---------------------------------------------
    # SLIDE 3: PRIMARY DRIVERS BREAKDOWN
    # ---------------------------------------------
    slide3 = prs.slides.add_slide(blank_slide_layout)
    add_corporate_header(slide3, "Primary Variance Drivers", primary_color, accent_color)
    
    df_box = slide3.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(9), Inches(5.5))
    df_frame = df_box.text_frame
    df_frame.word_wrap = True
    
    for node in tree_data:
        p = df_frame.paragraphs[0] if node == tree_data[0] else df_frame.add_paragraph()
        p.text = f"{node['item']} (Impact: {node['value_display']})"
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = primary_color
        p.level = 0
        
        for child in node.get('children', [])[:4]: # Limit to top 4 subs to prevent overflow
            p_sub = df_frame.add_paragraph()
            p_sub.text = f"{child['column']} - {child['item']} ({child['value_display']})"
            p_sub.font.size = Pt(14)
            p_sub.font.color.rgb = text_color
            p_sub.level = 1

    ppt_stream = io.BytesIO()
    prs.save(ppt_stream)
    ppt_stream.seek(0)
    return ppt_stream

# ==========================================
# 3. LANGGRAPH STATE & NODES 
# ==========================================
class AgentState(TypedDict):
    df: pd.DataFrame
    hierarchy_cols: List[str]
    has_variance_col: bool
    variance_col: str
    base_scenario: str
    compare_scenario: str
    path_trace: List[str]
    final_level_data: List[str]
    tree_data: List[Dict[str, Any]]
    final_summary: str

def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)

    if not hierarchy: return {"path_trace": ["Error: No hierarchy columns selected."], "final_level_data": [], "tree_data": []}

    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns: return {"path_trace": [f"Error: Target column '{target_col}' not found."], "final_level_data": [], "tree_data": []}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns: return {"path_trace": ["Error: Scenario columns not found."], "final_level_data": [], "tree_data": []}
        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    def format_variance(value: float) -> str: return f"{value / 1e6:,.2f}M"

    total_variance = df[target_col].fillna(0).sum()
    path_trace = [f"Overall Total Variance: {format_variance(total_variance)}"]
    final_level_data = [f"Overall Total Variance: {format_variance(total_variance)}"]

    def build_tree(current_df: pd.DataFrame, depth: int, indent: str) -> Any:
        if depth >= len(hierarchy) or current_df.empty: return [], [], []
        col_name = hierarchy[depth]
        is_first, is_last = depth == 0, depth == len(hierarchy) - 1
        grouped = current_df.groupby(col_name)[target_col].sum()
        if grouped.empty or grouped.isna().all(): return [], [], []
        top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
        trace_lines, final_lines, tree_nodes = [], [], []

        for item, val in top_5.items():
            if pd.isna(item): continue
            item_label, value_label = str(item), format_variance(float(val))
            if is_first:
                trace_lines.append(f"Primary Category: '{item_label}' (Total: {value_label})")
                final_lines.append(f"\nPrimary Category: '{item_label}' (Total Variance: {value_label})")
                title = f"Primary Category: {item_label} ({value_label})"
            elif is_last:
                trace_lines.append(f"{indent}Final Level ({col_name}): '{item_label}' -> {value_label}")
                final_lines.append(f"  - {col_name} '{item_label}': {value_label}")
                title = f"Final Level | {col_name}: {item_label} ({value_label})"
            else:
                trace_lines.append(f"{indent}Driver ({col_name}): '{item_label}' -> {value_label}")
                title = f"Driver | {col_name}: {item_label} ({value_label})"

            node = {"column": col_name, "item": item_label, "value": float(val), "value_display": value_label, "title": title, "children": []}
            if not is_last:
                next_df = current_df[current_df[col_name] == item]
                sub_trace, sub_final, sub_nodes = build_tree(next_df, depth + 1, indent + "      ")
                trace_lines.extend(sub_trace)
                final_lines.extend(sub_final)
                node["children"] = sub_nodes
            tree_nodes.append(node)
        return trace_lines, final_lines, tree_nodes

    tree_trace, tree_final, tree_nodes = build_tree(df, 0, "")
    path_trace.extend(tree_trace)
    final_level_data.extend(tree_final)
    return {"path_trace": path_trace, "final_level_data": final_level_data, "tree_data": tree_nodes}

def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    if state["path_trace"] and "Error:" in state["path_trace"][0]: return {"final_summary": "Analysis aborted."}
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    system_prompt = (
        "You are a strict, professional financial data analyst. Provide an Executive Summary formatted EXACTLY as follows:\n"
        "1. A brief 1-2 sentence overall conclusion regarding the total variance.\n"
        "2. A bulleted breakdown for each 'Primary Category' analyzed. Under each, list the Top 5 reasons/drivers provided, along with exact variance amounts.\n"
        "Do not add conversational filler."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Filtered Final Level Data:\n{chr(10).join(state['final_level_data'])}")]
    response = llm.invoke(messages)
    return {"final_summary": response.content}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("calculate_variance", calculate_variance_node)
    workflow.add_node("synthesize_insight", synthesize_insight_node)
    workflow.add_edge(START, "calculate_variance")
    workflow.add_edge("calculate_variance", "synthesize_insight")
    workflow.add_edge("synthesize_insight", END)
    return workflow.compile()

def render_trace_tree(nodes: List[Dict[str, Any]]) -> None:
    for node in nodes:
        if node.get("children"):
            with st.expander(node["title"], expanded=False): render_trace_tree(node["children"])
        else: st.markdown(f"- **{node['title']}**")

def count_leaf_nodes(nodes: List[Dict[str, Any]]) -> int:
    leaf_count = 0
    for node in nodes:
        children = node.get("children", [])
        if children: leaf_count += count_leaf_nodes(children)
        else: leaf_count += 1
    return leaf_count

# ==========================================
# 4. STREAMLIT UI 
# ==========================================
st.set_page_config(page_title="Branched Variance Analyzer", layout="wide")

# Initialize session state variables
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "current_hierarchy" not in st.session_state: st.session_state.current_hierarchy = None
if "current_run_id" not in st.session_state: st.session_state.current_run_id = None
if "run_feedback_submitted" not in st.session_state: st.session_state.run_feedback_submitted = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_feedback_submitted" not in st.session_state: st.session_state.chat_feedback_submitted = {}

st.title("📊 Branched Root Cause Analyzer")

# --- SIDEBAR CONFIGURATION ---
uploaded_file = None
df = None
variance_col, base_scenario, compare_scenario = "", "", ""
all_cols, cat_cols = [], []
has_variance_col = True

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        try:
            if file_name.endswith(".xlsx"):
                xls_temp = pd.ExcelFile(io.BytesIO(file_bytes))
                sheet_name = st.selectbox("Select Sheet", xls_temp.sheet_names)
                df = load_and_cache_data(file_bytes, file_name, sheet_name)
            else:
                df = load_and_cache_data(file_bytes, file_name)
            
            st.header("2. Configure Metrics")
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            all_cols = df.columns.tolist()
            
            has_variance_col = st.checkbox("Variance column already present?", value=True)
            
            if has_variance_col:
                default_var = next((c for c in num_cols if "var" in c.lower()), num_cols[0] if num_cols else None)
                idx = num_cols.index(default_var) if default_var in num_cols else 0
                variance_col = st.selectbox("Select Variance Column", num_cols, index=idx)
            else:
                st.caption("Calculate: (Base - Compare)")
                base_scenario = st.selectbox("Select Base Scenario (e.g. ACT)", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario (e.g. FC)", num_cols)

        except Exception as e:
            st.error(f"Error loading file: {e}")

tab1, tab2, tab3 = st.tabs(["🚀 New Analysis", "💬 Chat with Data", "🗄️ Run History"])

# --- TAB 1: NEW ANALYSIS ---
with tab1:
    if uploaded_file is not None and df is not None:
        st.markdown("### 3. Select Hierarchy")
        hierarchy = st.multiselect("Hierarchy Flow (Left to Right)", options=all_cols, default=cat_cols)

        run_analysis = st.button("Generate Commentary", type="primary", use_container_width=True)

        if run_analysis:
            if not hierarchy: st.warning("Please select at least one column for the hierarchy.")
            elif has_variance_col and not variance_col: st.warning("Please select a variance column.")
            elif not has_variance_col and (not base_scenario or not compare_scenario): st.warning("Please select both Base and Compare scenarios.")
            else:
                with st.spinner("Executing recursive drill-down analysis..."):
                    st.session_state.run_feedback_submitted = False 
                    
                    app_graph = build_graph()
                    inputs = {
                        "df": df, "hierarchy_cols": hierarchy, "has_variance_col": has_variance_col,
                        "variance_col": variance_col, "base_scenario": base_scenario, "compare_scenario": compare_scenario,
                    }
                    
                    st.session_state.analysis_result = app_graph.invoke(inputs)
                    st.session_state.current_hierarchy = hierarchy
                    
                    if "aborted" not in st.session_state.analysis_result["final_summary"].lower():
                        total_var = st.session_state.analysis_result["path_trace"][0].replace("Overall Total Variance: ", "")
                        run_id = save_run_to_db(file_name, hierarchy, total_var, st.session_state.analysis_result["final_summary"])
                        st.session_state.current_run_id = run_id

        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            st.markdown("---")
            
            if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                st.error(result["final_summary"])
            else:
                total_var = result["path_trace"][0].replace("Overall Total Variance: ", "")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Variance", total_var)
                col2.metric("Hierarchy Levels", str(len(st.session_state.current_hierarchy)))
                col3.metric("Primary Branches", str(len(result.get("tree_data", []))))
                col4.metric("Final Nodes", str(count_leaf_nodes(result.get("tree_data", []))))

                st.markdown("<br>", unsafe_allow_html=True)
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Executive Summary")
                    st.info(result["final_summary"])
                with col_right:
                    st.subheader("Recursive Drill-Down Trace")
                    st.caption(result["path_trace"][0])
                    render_trace_tree(result.get("tree_data", []))

                # --- FEEDBACK MODULE (Now positioned above Export) ---
                st.markdown("---")
                if st.session_state.current_run_id:
                    if st.session_state.run_feedback_submitted:
                        st.success("Thank you for your feedback! ✅")
                    else:
                        st.markdown("##### How did the AI perform on this analysis?")
                        col_f1, col_f2, _ = st.columns([1, 1, 10])
                        col_f1.button("👍", key="run_up", on_click=handle_run_feedback_click, args=(st.session_state.current_run_id, 1))
                        col_f2.button("👎", key="run_down", on_click=handle_run_feedback_click, args=(st.session_state.current_run_id, -1))

                # --- POWERPOINT EXPORT BUTTON (Small column width) ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Export Results")
                
                ppt_file = generate_ppt_deck(
                    total_variance=total_var,
                    summary=result["final_summary"],
                    tree_data=result.get("tree_data", [])
                )
                
                # Wrapping in a column to make the button smaller
                col_dl1, col_dl2 = st.columns([1, 2])
                with col_dl1:
                    st.download_button(
                        label="📥 Download Deck (.pptx)",
                        data=ppt_file,
                        file_name=f"Variance_Analysis_Deck_{datetime.now().strftime('%Y%m%d')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
    else:
        st.info("👈 Please upload and configure your data in the sidebar to begin.")

# --- TAB 2: CHAT WITH DATA ---
with tab2:
    if uploaded_file is not None and df is not None:
        st.markdown("### 💬 Ask Questions About Your Data")
        
        for idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant" and "db_id" in msg:
                    if st.session_state.chat_feedback_submitted.get(idx):
                        st.success("Feedback recorded ✅")
                    else:
                        f1, f2, _ = st.columns([1, 1, 10])
                        f1.button("👍", key=f"chat_up_{idx}", on_click=handle_chat_feedback_click, args=(msg["db_id"], 1, idx))
                        f2.button("👎", key=f"chat_down_{idx}", on_click=handle_chat_feedback_click, args=(msg["db_id"], -1, idx))

        if prompt := st.chat_input("Ask your dataset a question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    llm_chat = AzureChatOpenAI(
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.getenv("AZURE_OPENAI_KEY"),
                        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        temperature=0,
                    )
                    agent = create_pandas_dataframe_agent(
                        llm_chat, df, verbose=True, agent_type="openai-functions", allow_dangerous_code=True 
                    )
                    
                    response = agent.invoke({"input": prompt}, {"callbacks": [st_callback]})
                    answer = response["output"]
                    st.markdown(answer)
                    
                    chat_id = save_chat_to_db(prompt, answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "db_id": chat_id})
                    st.rerun() 
                    
                except Exception as e:
                    st.error(f"Failed to query data: {e}")
    else:
        st.info("👈 Please upload a data file in the sidebar to chat with it.")

# --- TAB 3: HISTORY ---
with tab3:
    st.markdown("### 🗄️ Historical Analysis Runs")
    history_df = fetch_history_from_db()
    if history_df.empty:
        st.write("No history found. Run an analysis to generate logs!")
    else:
        history_df['feedback_status'] = history_df['feedback'].map({1: '👍', -1: '👎', 0: '➖'})
        st.dataframe(history_df[["id", "timestamp", "filename", "total_variance", "feedback_status"]], use_container_width=True, hide_index=True)
        selected_run = st.selectbox("Select a Run ID to view details:", history_df["id"].tolist())
        if selected_run:
            run_details = history_df[history_df["id"] == selected_run].iloc[0]
            st.markdown(f"**Filename:** `{run_details['filename']}` | **Ran at:** `{run_details['timestamp']}`")
            st.markdown(f"**Current Feedback Score:** `{run_details['feedback_status']}`")
            st.info(run_details['summary'])

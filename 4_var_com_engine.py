import io
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Variance Intelligence Hub",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

# ==========================================
# GLOBAL CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* App shell */
.stApp { background: #0d1117; color: #c9d1d9; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1f2937 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #9ca3af !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e5e7eb !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px 6px;
    gap: 4px;
    border: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #6b7280 !important;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 8px 20px;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #f0f6fc !important;
}

/* KPI cards */
.kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    margin-bottom: 4px;
}
.kpi-card-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4b5563;
    font-weight: 600;
    margin-bottom: 10px;
    margin-top: 4px;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f0f6fc;
    line-height: 1;
}
.kpi-icon {
    position: absolute;
    right: 16px;
    top: 16px;
    font-size: 1.2rem;
    opacity: 0.15;
}

/* Section title */
.sec-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4b5563;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1f2937;
}

/* Summary card */
.summary-wrap {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 0.88rem;
    line-height: 1.85;
    color: #c9d1d9;
    white-space: pre-wrap;
}

/* Leaf bar row */
.leaf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0;
    padding: 7px 10px;
    border-radius: 7px;
    background: #0d1117;
    border: 1px solid #1f2937;
}
.leaf-label {
    flex: 0 0 220px;
    font-size: 0.76rem;
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.leaf-bar-wrap {
    flex: 1;
    background: #1f2937;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.leaf-bar-fill { height: 100%; border-radius: 4px; }
.leaf-val {
    flex: 0 0 80px;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    font-weight: 600;
}
.pos { color: #3fb950; }
.neg { color: #f85149; }

/* History cards */
.hist-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.hist-ts {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #4b5563;
    margin-bottom: 4px;
}
.hist-file { font-size: 0.9rem; font-weight: 500; color: #e5e7eb; }
.hist-meta { font-size: 0.76rem; color: #6b7280; margin-top: 4px; }
.fb-badge {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 8px;
}
.fb-up   { background: #0d2818; color: #3fb950; border: 1px solid #1a4a28; }
.fb-down { background: #2d0f0f; color: #f85149; border: 1px solid #4a1a1a; }
.fb-none { background: #1f2937; color: #6b7280; border: 1px solid #374151; }

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    border: 1px solid #1f2937 !important;
    background: #111827 !important;
    color: #c9d1d9 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1f2937 !important;
    border-color: #374151 !important;
    color: #f0f6fc !important;
}
button[kind="primary"] {
    background: #1f6feb !important;
    border-color: #388bfd !important;
    color: #ffffff !important;
}

/* Inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div {
    background: #111827 !important;
    border-color: #1f2937 !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #1f2937;
    border-radius: 10px;
}

hr { border-color: #1f2937 !important; margin: 16px 0 !important; }

.stCheckbox label { color: #9ca3af !important; font-size: 0.85rem !important; }

.stDownloadButton > button {
    width: 100%;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #9ca3af !important;
}

/* Expander */
details summary {
    background: #111827 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.82rem !important;
    color: #c9d1d9 !important;
    border: 1px solid #1f2937 !important;
    margin-bottom: 2px !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATABASE LAYER
# ==========================================
DB_PATH = "variance_master.db"


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at            TEXT NOT NULL,
            file_name         TEXT,
            hierarchy         TEXT,
            variance_mode     TEXT,
            variance_col      TEXT,
            base_scenario     TEXT,
            compare_scenario  TEXT,
            total_variance    TEXT,
            primary_branches  INTEGER,
            final_nodes       INTEGER,
            executive_summary TEXT,
            tree_data_json    TEXT,
            path_trace_json   TEXT,
            feedback          INTEGER DEFAULT NULL,
            feedback_note     TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(meta: Dict, result: Dict) -> int:
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO analyses (
            run_at, file_name, hierarchy, variance_mode, variance_col,
            base_scenario, compare_scenario, total_variance,
            primary_branches, final_nodes, executive_summary,
            tree_data_json, path_trace_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        meta["file_name"],
        json.dumps(meta["hierarchy"]),
        meta["variance_mode"],
        meta.get("variance_col", ""),
        meta.get("base_scenario", ""),
        meta.get("compare_scenario", ""),
        meta["total_variance"],
        meta["primary_branches"],
        meta["final_nodes"],
        result["final_summary"],
        json.dumps(result["tree_data"]),
        json.dumps(result["path_trace"]),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_feedback(row_id: int, score: int, note: str = ""):
    conn = get_conn()
    conn.execute(
        "UPDATE analyses SET feedback=?, feedback_note=? WHERE id=?",
        (score, note, row_id),
    )
    conn.commit()
    conn.close()


def fetch_history() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT id, run_at, file_name, hierarchy, total_variance,
                  primary_branches, final_nodes, feedback, feedback_note
           FROM analyses ORDER BY run_at DESC""",
        conn,
    )
    conn.close()
    return df


def fetch_by_id(row_id: int) -> Dict:
    conn = get_conn()
    cur = conn.execute("SELECT * FROM analyses WHERE id=?", (row_id,))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    conn.close()
    return dict(zip(cols, row)) if row else {}


init_db()


# ==========================================
# LANGGRAPH STATE
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


# ==========================================
# LANGGRAPH NODES  (logic unchanged)
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)

    if not hierarchy:
        return {"path_trace": ["Error: No hierarchy columns selected."],
                "final_level_data": [], "tree_data": []}

    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns:
            return {"path_trace": [f"Error: Column '{target_col}' not found."],
                    "final_level_data": [], "tree_data": []}
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns:
            return {"path_trace": ["Error: Scenario columns not found."],
                    "final_level_data": [], "tree_data": []}
        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    def fmt(v: float) -> str:
        return f"{v / 1e6:,.2f}M"

    total = df[target_col].fillna(0).sum()
    path_trace = [f"Overall Total Variance: {fmt(total)}"]
    final_level_data = [f"Overall Total Variance: {fmt(total)}"]

    def build_tree(cur_df, depth, indent):
        if depth >= len(hierarchy) or cur_df.empty:
            return [], [], []
        col = hierarchy[depth]
        is_first = depth == 0
        is_last  = depth == len(hierarchy) - 1
        grouped = cur_df.groupby(col)[target_col].sum()
        if grouped.empty or grouped.isna().all():
            return [], [], []
        top5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)
        t_lines, f_lines, nodes = [], [], []
        for item, val in top5.items():
            if pd.isna(item):
                continue
            lbl = str(item)
            vl  = fmt(float(val))
            if is_first:
                t_lines.append(f"Primary Category: '{lbl}' (Total: {vl})")
                f_lines.append(f"\nPrimary Category: '{lbl}' (Total Variance: {vl})")
                title = f"Primary Category: {lbl} ({vl})"
            elif is_last:
                t_lines.append(f"{indent}Final Level ({col}): '{lbl}' -> {vl}")
                f_lines.append(f"  - {col} '{lbl}': {vl}")
                title = f"Final Level | {col}: {lbl} ({vl})"
            else:
                t_lines.append(f"{indent}Driver ({col}): '{lbl}' -> {vl}")
                title = f"Driver | {col}: {lbl} ({vl})"
            node: Dict[str, Any] = {
                "column": col, "item": lbl,
                "value": float(val), "value_display": vl,
                "title": title, "children": [],
            }
            if not is_last:
                ndf = cur_df[cur_df[col] == item]
                st_, sf_, sn_ = build_tree(ndf, depth + 1, indent + "      ")
                t_lines.extend(st_)
                f_lines.extend(sf_)
                node["children"] = sn_
            nodes.append(node)
        return t_lines, f_lines, nodes

    tt, tf, tn = build_tree(df, 0, "")
    path_trace.extend(tt)
    final_level_data.extend(tf)
    return {"path_trace": path_trace, "final_level_data": final_level_data, "tree_data": tn}


def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    if state["path_trace"] and "Error:" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid data configuration."}
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    system_prompt = (
        "You are a strict, professional financial data analyst. All values are in Millions (M).\n\n"
        "Provide an Executive Summary formatted EXACTLY as:\n"
        "1. A brief 1-2 sentence overall conclusion on the total variance.\n"
        "2. A bulleted breakdown for each 'Primary Category' with Top 5 drivers and exact amounts.\n"
        "No filler. Structured for executive readability."
    )
    trace_text = "\n".join(state["final_level_data"])
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Filtered Final Level Data:\n{trace_text}"),
    ])
    return {"final_summary": resp.content}


def build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("calculate_variance", calculate_variance_node)
    wf.add_node("synthesize_insight", synthesize_insight_node)
    wf.add_edge(START, "calculate_variance")
    wf.add_edge("calculate_variance", "synthesize_insight")
    wf.add_edge("synthesize_insight", END)
    return wf.compile()


# ==========================================
# UI HELPERS
# ==========================================
def count_leaves(nodes: List[Dict]) -> int:
    c = 0
    for n in nodes:
        ch = n.get("children", [])
        c += count_leaves(ch) if ch else 1
    return c


def render_tree(nodes: List[Dict], depth: int = 0):
    """Recursively render drill-down. Leaves → mini bar chart rows."""
    for node in nodes:
        val  = node.get("value", 0)
        vd   = node.get("value_display", "")
        vc   = "pos" if val >= 0 else "neg"
        col  = node.get("column", "")
        item = node.get("item", "")

        if node.get("children"):
            with st.expander(node["title"], expanded=(depth == 0)):
                render_tree(node["children"], depth + 1)
        else:
            bar_pct   = min(abs(val) / 1e8 * 100, 100)
            bar_color = "#3fb950" if val >= 0 else "#f85149"
            st.markdown(f"""
            <div class="leaf-row">
                <div class="leaf-label" title="{item}">{item}</div>
                <div class="leaf-bar-wrap">
                    <div class="leaf-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="leaf-val {vc}">{vd}</div>
            </div>
            """, unsafe_allow_html=True)


def render_kpis(total_var: str, hier_count: int, branches: int, leaves: int):
    cards = [
        ("#388bfd", "💰", "TOTAL VARIANCE",    total_var),
        ("#3fb950", "🔗", "HIERARCHY LEVELS",  str(hier_count)),
        ("#d29922", "🌿", "PRIMARY BRANCHES",  str(branches)),
        ("#8b949e", "📍", "FINAL NODES",       str(leaves)),
    ]
    cols = st.columns(4)
    for col, (accent, icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-accent" style="background:{accent}"></div>
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def render_feedback(run_id: int):
    """Thumbs up / down with optional note."""
    st.markdown("<div class='sec-title'>Feedback</div>", unsafe_allow_html=True)
    fb_key = f"fb_submitted_{run_id}"
    if fb_key not in st.session_state:
        st.session_state[fb_key] = None

    if st.session_state[fb_key] is not None:
        icon = "👍" if st.session_state[fb_key] == 1 else "👎"
        st.markdown(
            f"<div style='color:#6b7280;font-size:0.82rem'>{icon} Feedback recorded — thank you.</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("👍  Helpful", key=f"up_{run_id}", use_container_width=True):
            update_feedback(run_id, 1)
            st.session_state[fb_key] = 1
            st.rerun()
    with c2:
        if st.button("👎  Not Helpful", key=f"dn_{run_id}", use_container_width=True):
            update_feedback(run_id, -1)
            st.session_state[fb_key] = -1
            st.rerun()
    with c3:
        note = st.text_input(
            "Comment",
            key=f"note_{run_id}",
            placeholder="Optional comment …",
            label_visibility="collapsed",
        )
        if note:
            if st.button("Save Note", key=f"nb_{run_id}"):
                update_feedback(run_id, 0, note)
                st.session_state[fb_key] = 0
                st.rerun()


def build_export_text(meta: Dict, result: Dict) -> str:
    hier = json.loads(meta["hierarchy"]) if isinstance(meta["hierarchy"], str) else meta["hierarchy"]
    lines = [
        "=" * 64,
        "  VARIANCE INTELLIGENCE HUB  —  ANALYSIS REPORT",
        "=" * 64,
        f"  Run At          : {meta.get('run_at', '')}",
        f"  File            : {meta.get('file_name', '')}",
        f"  Hierarchy       : {' -> '.join(hier)}",
        f"  Total Variance  : {meta.get('total_variance', '')}",
        f"  Primary Branches: {meta.get('primary_branches', '')}",
        f"  Final Nodes     : {meta.get('final_nodes', '')}",
        "=" * 64,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        result["final_summary"],
        "",
        "DRILL-DOWN TRACE",
        "-" * 40,
    ]
    lines.extend(result["path_trace"])
    return "\n".join(lines)


# ==========================================
# HISTORY TAB
# ==========================================
def render_history_tab():
    st.markdown("<div class='sec-title'>Past Analyses</div>", unsafe_allow_html=True)
    hist = fetch_history()

    if hist.empty:
        st.markdown("""
        <div style='color:#4b5563;font-size:0.85rem;padding:40px 0;text-align:center'>
            No analyses stored yet.<br>Run your first analysis to see history here.
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary stats
    total_runs = len(hist)
    helpful    = int((hist["feedback"] == 1).sum())
    unhelpful  = int((hist["feedback"] == -1).sum())
    pending    = int(hist["feedback"].isna().sum())

    for col, label, val, color in zip(
        st.columns(4),
        ["Total Runs", "👍 Helpful", "👎 Not Helpful", "⏳ Pending"],
        [total_runs, helpful, unhelpful, pending],
        ["#8b949e", "#3fb950", "#f85149", "#d29922"],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="padding:14px 18px">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.3rem;color:{color}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-row cards
    for _, row in hist.iterrows():
        fb = row.get("feedback")
        if fb == 1:
            fb_html = "<span class='fb-badge fb-up'>👍 Helpful</span>"
        elif fb == -1:
            fb_html = "<span class='fb-badge fb-down'>👎 Not Helpful</span>"
        else:
            fb_html = "<span class='fb-badge fb-none'>No Feedback</span>"

        hier_list = json.loads(row["hierarchy"]) if row["hierarchy"] else []
        hier_str  = " → ".join(hier_list) if hier_list else "—"

        st.markdown(f"""
        <div class="hist-card">
            <div class="hist-ts">#{row['id']} &nbsp;·&nbsp; {row['run_at']} {fb_html}</div>
            <div class="hist-file">📄 {row['file_name']}</div>
            <div class="hist-meta">
                Hierarchy: {hier_str} &nbsp;·&nbsp;
                Variance: <span style="font-family:IBM Plex Mono,monospace">{row['total_variance']}</span>
                &nbsp;·&nbsp; Branches: {row['primary_branches']}
                &nbsp;·&nbsp; Nodes: {row['final_nodes']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"↗ Load #{row['id']}", key=f"load_{row['id']}"):
            st.session_state["loaded_run_id"] = row["id"]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = hist.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Export History as CSV",
        data=csv_bytes,
        file_name=f"variance_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==========================================
# LOADED HISTORICAL RUN DISPLAY
# ==========================================
def render_loaded_run() -> bool:
    rid = st.session_state.get("loaded_run_id")
    if rid is None:
        return False
    rec = fetch_by_id(rid)
    if not rec:
        st.error("Record not found.")
        return False

    hier  = json.loads(rec["hierarchy"]) if rec["hierarchy"] else []
    tree  = json.loads(rec["tree_data_json"]) if rec["tree_data_json"] else []
    trace = json.loads(rec["path_trace_json"]) if rec["path_trace_json"] else []

    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1f2937;border-left:3px solid #388bfd;
                border-radius:10px;padding:12px 18px;margin-bottom:20px;
                font-size:0.82rem;color:#6b7280">
        📂 Viewing saved run
        <span style="color:#e5e7eb;font-weight:600">#{rid}</span>
        &nbsp;·&nbsp; {rec['run_at']}
        &nbsp;·&nbsp; {rec['file_name']}
    </div>
    """, unsafe_allow_html=True)

    if st.button("✕ Close & Return to New Analysis", key="close_loaded"):
        del st.session_state["loaded_run_id"]
        st.rerun()

    render_kpis(
        total_var  = rec["total_variance"],
        hier_count = len(hier),
        branches   = rec["primary_branches"],
        leaves     = rec["final_nodes"],
    )

    c1, c2 = st.columns([5, 4], gap="large")
    with c1:
        st.markdown("<div class='sec-title'>Executive Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-wrap'>{rec['executive_summary']}</div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sec-title'>Drill-Down Trace</div>", unsafe_allow_html=True)
        if trace:
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1f2937;border-left:3px solid #388bfd;
                        border-radius:8px;padding:10px 14px;margin-bottom:10px;
                        font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#388bfd">
                {trace[0]}
            </div>
            """, unsafe_allow_html=True)
        render_tree(tree)

    render_feedback(rid)
    return True


# ==========================================
# SIDEBAR
# ==========================================
df               = None
num_cols         = []
cat_cols         = []
all_cols         = []
has_variance_col = True
variance_col     = ""
base_scenario    = ""
compare_scenario = ""
uploaded_file    = None

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px">
        <div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;
                    color:#388bfd;font-weight:600;letter-spacing:0.04em">VIH</div>
        <div style="font-size:0.68rem;color:#4b5563;letter-spacing:0.12em;
                    text-transform:uppercase;margin-top:2px">
            Variance Intelligence Hub
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Upload Data**")
    uploaded_file = st.file_uploader("CSV or Excel", type=["csv", "xlsx"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                buf = io.BytesIO(uploaded_file.getvalue())
                xls = pd.ExcelFile(buf)
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df = pd.read_excel(buf, sheet_name=sheet)
            else:
                df = pd.read_csv(uploaded_file)

            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            all_cols = df.columns.tolist()

            st.markdown("---")
            st.markdown("**Metric Configuration**")
            has_variance_col = st.checkbox("Pre-computed variance column?", value=True)

            if has_variance_col:
                default_v = next((c for c in num_cols if "var" in c.lower()), num_cols[0] if num_cols else None)
                idx = num_cols.index(default_v) if default_v in num_cols else 0
                variance_col = st.selectbox("Variance Column", num_cols, index=idx)
            else:
                st.caption("Computes: Base − Compare")
                base_scenario    = st.selectbox("Base Scenario", num_cols)
                compare_scenario = st.selectbox("Compare Scenario", num_cols)

            st.markdown("---")
            st.markdown(f"""
            <div style="font-size:0.72rem;color:#4b5563;line-height:2">
                <span style="color:#6b7280">Rows</span> &nbsp; {len(df):,}<br>
                <span style="color:#6b7280">Columns</span> &nbsp; {len(df.columns)}<br>
                <span style="color:#6b7280">Numeric</span> &nbsp; {len(num_cols)}<br>
                <span style="color:#6b7280">Categorical</span> &nbsp; {len(cat_cols)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")


# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown("""
<div style="margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #1f2937">
    <div style="font-family:IBM Plex Mono,monospace;font-size:1.9rem;
                font-weight:600;color:#f0f6fc;letter-spacing:-0.02em">
        Variance Intelligence Hub
    </div>
    <div style="font-size:0.76rem;color:#4b5563;margin-top:5px;
                letter-spacing:0.1em;text-transform:uppercase">
        Branched Root-Cause Analyzer &nbsp;·&nbsp; LangGraph + Azure OpenAI
    </div>
</div>
""", unsafe_allow_html=True)

tab_analysis, tab_history = st.tabs(["📊  Analysis", "🕓  History"])

# ── HISTORY TAB ─────────────────────────────────────────
with tab_history:
    render_history_tab()

# ── ANALYSIS TAB ─────────────────────────────────────────
with tab_analysis:

    # Show loaded historical run if requested
    if "loaded_run_id" in st.session_state:
        render_loaded_run()
        st.stop()

    # Empty state
    if df is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;color:#4b5563">
            <div style="font-size:2.8rem;margin-bottom:16px;opacity:0.4">📂</div>
            <div style="font-size:0.95rem;font-weight:500;color:#6b7280;margin-bottom:8px">
                No data loaded
            </div>
            <div style="font-size:0.82rem;color:#4b5563">
                Upload a CSV or Excel file in the sidebar to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Data preview
    with st.expander("📋  Data Preview", expanded=False):
        st.dataframe(df.head(5), use_container_width=True, height=200)

    # Hierarchy
    st.markdown("<div class='sec-title'>Hierarchy Configuration</div>", unsafe_allow_html=True)
    hierarchy = st.multiselect(
        "Drill-down order (left → right = top → bottom of hierarchy)",
        options=all_cols,
        default=cat_cols,
        help="Groups by the first column, drills through middle levels, reports from the last column.",
    )

    if hierarchy:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
            f"color:#388bfd;background:#0d1117;border:1px solid #1f2937;"
            f"border-radius:8px;padding:10px 14px;margin-top:6px'>"
            f"{'  →  '.join(hierarchy)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Generate Analysis", type="primary", use_container_width=True)

    if run_btn:
        errors = []
        if not hierarchy:
            errors.append("Select at least one hierarchy column.")
        if has_variance_col and not variance_col:
            errors.append("Select a variance column.")
        if not has_variance_col and (not base_scenario or not compare_scenario):
            errors.append("Select both Base and Compare scenario columns.")

        if errors:
            for e in errors:
                st.warning(e)
        else:
            with st.spinner("Running recursive drill-down …"):
                graph  = build_graph()
                inputs = {
                    "df": df,
                    "hierarchy_cols": hierarchy,
                    "has_variance_col": has_variance_col,
                    "variance_col": variance_col,
                    "base_scenario": base_scenario,
                    "compare_scenario": compare_scenario,
                }
                result = graph.invoke(inputs)

            is_error = (
                "aborted" in result["final_summary"].lower()
                or (result["path_trace"] and "Error:" in result["path_trace"][0])
            )

            if is_error:
                st.error(result["final_summary"])
            else:
                total_var_str    = result["path_trace"][0].replace("Overall Total Variance: ", "")
                primary_branches = len(result.get("tree_data", []))
                final_nodes      = count_leaves(result.get("tree_data", []))

                # ── Persist to SQLite ──
                meta = {
                    "file_name":        uploaded_file.name,
                    "hierarchy":        hierarchy,
                    "variance_mode":    "pre_computed" if has_variance_col else "calculated",
                    "variance_col":     variance_col,
                    "base_scenario":    base_scenario,
                    "compare_scenario": compare_scenario,
                    "total_variance":   total_var_str,
                    "primary_branches": primary_branches,
                    "final_nodes":      final_nodes,
                    "run_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                run_id = save_analysis(meta, result)

                # ── KPI row ──
                render_kpis(total_var_str, len(hierarchy), primary_branches, final_nodes)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Main two-column output ──
                col_l, col_r = st.columns([5, 4], gap="large")

                with col_l:
                    st.markdown("<div class='sec-title'>Executive Summary</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='summary-wrap'>{result['final_summary']}</div>",
                        unsafe_allow_html=True,
                    )

                with col_r:
                    st.markdown("<div class='sec-title'>Drill-Down Trace</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #1f2937;
                                border-left:3px solid #388bfd;border-radius:8px;
                                padding:10px 14px;margin-bottom:10px;
                                font-family:IBM Plex Mono,monospace;
                                font-size:0.8rem;color:#388bfd">
                        {result["path_trace"][0]}
                    </div>
                    """, unsafe_allow_html=True)
                    render_tree(result.get("tree_data", []))

                # ── Export buttons ──
                st.markdown("<div class='sec-title'>Export</div>", unsafe_allow_html=True)
                ex1, ex2 = st.columns(2)
                with ex1:
                    txt = build_export_text(meta, result)
                    st.download_button(
                        label="⬇ Full Report (.txt)",
                        data=txt.encode(),
                        file_name=f"variance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with ex2:
                    st.download_button(
                        label="⬇ Executive Summary (.txt)",
                        data=result["final_summary"].encode(),
                        file_name=f"exec_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                # ── Feedback ──
                render_feedback(run_id)

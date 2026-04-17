import io
import os
from typing import Any, Dict, List, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

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
    final_level_data: List[str]
    tree_data: List[Dict[str, Any]]
    final_summary: str


# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def calculate_variance_node(state: AgentState) -> Dict[str, Any]:
    """Branched Drill-Down: Uses recursion to check the top 5 of every branch."""
    df = state["df"].copy()
    hierarchy = state.get("hierarchy_cols", [])
    has_var = state.get("has_variance_col", True)

    if not hierarchy:
        return {
            "path_trace": ["Error: No hierarchy columns selected."],
            "final_level_data": [],
            "tree_data": [],
        }

    if has_var:
        target_col = state.get("variance_col")
        if target_col not in df.columns:
            return {
                "path_trace": [f"Error: Target column '{target_col}' not found."],
                "final_level_data": [],
                "tree_data": [],
            }
    else:
        base_col = state.get("base_scenario")
        comp_col = state.get("compare_scenario")
        if base_col not in df.columns or comp_col not in df.columns:
            return {
                "path_trace": ["Error: Scenario columns not found."],
                "final_level_data": [],
                "tree_data": [],
            }

        target_col = "Calculated_Variance"
        df[target_col] = df[base_col] - df[comp_col]

    def format_variance(value: float) -> str:
        return f"{value / 1e6:,.2f}M"

    total_variance = df[target_col].fillna(0).sum()
    path_trace = [f"Overall Total Variance: {format_variance(total_variance)}"]
    final_level_data = [f"Overall Total Variance: {format_variance(total_variance)}"]

    def build_tree(current_df: pd.DataFrame, depth: int, indent: str) -> Any:
        if depth >= len(hierarchy) or current_df.empty:
            return [], [], []

        col_name = hierarchy[depth]
        is_first = depth == 0
        is_last = depth == len(hierarchy) - 1

        grouped = current_df.groupby(col_name)[target_col].sum()
        if grouped.empty or grouped.isna().all():
            return [], [], []

        top_5 = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(5)

        trace_lines: List[str] = []
        final_lines: List[str] = []
        tree_nodes: List[Dict[str, Any]] = []

        for item, val in top_5.items():
            if pd.isna(item):
                continue

            item_label = str(item)
            value_label = format_variance(float(val))

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

            node: Dict[str, Any] = {
                "column": col_name,
                "item": item_label,
                "value": float(val),
                "value_display": value_label,
                "title": title,
                "children": [],
            }

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

    return {
        "path_trace": path_trace,
        "final_level_data": final_level_data,
        "tree_data": tree_nodes,
    }


def synthesize_insight_node(state: AgentState) -> Dict[str, Any]:
    """Azure OpenAI generates an executive summary using only the strict final column data."""
    if state["path_trace"] and "Error:" in state["path_trace"][0]:
        return {"final_summary": "Analysis aborted due to invalid data configuration."}

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    system_prompt = (
        "You are a strict, professional financial data analyst. You are reviewing variance data. "
        "All values are in Millions (M).\n\n"
        "Provide an Executive Summary formatted EXACTLY as follows:\n"
        "1. A brief 1-2 sentence overall conclusion regarding the total variance.\n"
        "2. A bulleted breakdown for each 'Primary Category' analyzed. Under each Primary Category, "
        "explicitly list the Top 5 reasons/drivers provided in the data, along with their exact variance amounts. "
        "(If there are more than 5 provided, pick the 5 with the highest absolute variance impact).\n\n"
        "Do not add conversational filler. Keep it structured and easy for an executive to read."
    )

    trace_text = "\n".join(state["final_level_data"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Filtered Final Level Data:\n{trace_text}"),
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


def render_trace_tree(nodes: List[Dict[str, Any]]) -> None:
    """Render the recursive drill-down as collapsed expanders."""
    for node in nodes:
        if node.get("children"):
            with st.expander(node["title"], expanded=False):
                render_trace_tree(node["children"])
        else:
            st.markdown(f"- {node['title']}")


def count_leaf_nodes(nodes: List[Dict[str, Any]]) -> int:
    leaf_count = 0
    for node in nodes:
        children = node.get("children", [])
        if children:
            leaf_count += count_leaf_nodes(children)
        else:
            leaf_count += 1
    return leaf_count


def render_kpi_cards(
    total_variance_label: str,
    hierarchy_count: int,
    primary_branch_count: int,
    final_node_count: int,
) -> None:
    metric_cols = st.columns(4)
    metrics = [
        ("Total Variance", total_variance_label),
        ("Hierarchy Levels", str(hierarchy_count)),
        ("Primary Branches", str(primary_branch_count)),
        ("Final Nodes", str(final_node_count)),
    ]

    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.metric(label, value)


def render_app_shell() -> None:
    """Apply minimal page styling and result card layout."""
    st.html(
        """
        <style>
            @keyframes fadeSlideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .section-title {
                color: #0f172a;
                font-size: 1.12rem;
                font-weight: 700;
                margin: 0.15rem 0 0.2rem 0;
            }

            .section-subtitle {
                color: #64748b;
                font-size: 0.92rem;
                margin-bottom: 0.55rem;
            }

            .block-card {
                border: 1px solid rgba(30, 41, 59, 0.08);
                border-radius: 14px;
                padding: 0.8rem 0.95rem 0.3rem 0.95rem;
                background: #ffffff;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 0.85rem;
                animation: fadeSlideIn 0.45s ease-out;
            }

            .result-card {
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                background: #ffffff;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
                padding: 1rem 1rem 0.9rem 1rem;
                margin-bottom: 1rem;
                animation: fadeSlideIn 0.45s ease-out;
            }

            .result-card-title {
                color: #0f172a;
                font-size: 1.18rem;
                font-weight: 750;
                margin: 0 0 0.2rem 0;
            }

            .result-card-subtitle {
                color: #64748b;
                font-size: 0.92rem;
                margin-bottom: 0.85rem;
            }
        </style>
        """
    )


def render_section_header(title: str, subtitle: str) -> None:
    st.html(
        f"""
        <div class="block-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """
    )


def open_result_card(title: str, subtitle: str) -> None:
    st.html(
        f"""
        <div class="result-card">
            <div class="result-card-title">{title}</div>
            <div class="result-card-subtitle">{subtitle}</div>
        """
    )


def close_result_card() -> None:
    st.html("</div>")


# ==========================================
# 4. STREAMLIT UI (SIDEBAR CONFIG)
# ==========================================
st.set_page_config(page_title="Branched Variance Analyzer", layout="wide")

render_app_shell()

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                file_buffer = io.BytesIO(uploaded_file.getvalue())
                xls = pd.ExcelFile(file_buffer)
                sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
                df = pd.read_excel(file_buffer, sheet_name=sheet_name)
            else:
                df = pd.read_csv(uploaded_file)

            st.header("2. Configure Metrics")
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()

            has_variance_col = st.checkbox("Variance column already present?", value=True)

            variance_col = ""
            base_scenario = ""
            compare_scenario = ""

            if has_variance_col:
                default_var = next((c for c in num_cols if "var" in c.lower()), num_cols[0] if num_cols else None)
                idx = num_cols.index(default_var) if default_var in num_cols else 0
                variance_col = st.selectbox("Select Variance Column", num_cols, index=idx)
            else:
                st.caption("Calculate: (Base - Compare)")
                base_scenario = st.selectbox("Select Base Scenario (e.g. 2024_ACT)", num_cols)
                compare_scenario = st.selectbox("Select Compare Scenario (e.g. 2025_FC2+10)", num_cols)

            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            all_cols = df.columns.tolist()

        except Exception as e:
            st.error(f"Error loading file: {e}")


# MAIN PAGE DISPLAY
if uploaded_file is not None and "df" in locals():
    render_section_header(
        "Data Preview",
        "Review the incoming dataset before selecting the hierarchy flow.",
    )
    st.dataframe(df.head(3))

    render_section_header(
        "3. Select Hierarchy",
        "Choose the hierarchy order from left to right for the recursive drill-down.",
    )
    hierarchy = st.multiselect(
        "Hierarchy Flow (Left to Right)",
        options=all_cols,
        default=cat_cols,
        help="The bot will group by the first column, recursively drill through the middle, and report from the LAST column.",
    )

    run_analysis = st.button("Generate Commentary", type="primary", use_container_width=True)

    if run_analysis:
        if not hierarchy:
            st.warning("Please select at least one column for the hierarchy.")
        elif has_variance_col and not variance_col:
            st.warning("Please select a variance column.")
        elif not has_variance_col and (not base_scenario or not compare_scenario):
            st.warning("Please select both Base and Compare scenarios.")
        else:
            with st.spinner("Executing recursive drill-down analysis..."):
                app_graph = build_graph()

                inputs = {
                    "df": df,
                    "hierarchy_cols": hierarchy,
                    "has_variance_col": has_variance_col,
                    "variance_col": variance_col,
                    "base_scenario": base_scenario,
                    "compare_scenario": compare_scenario,
                }

                result = app_graph.invoke(inputs)

                st.markdown("---")
                
                if "aborted" in result["final_summary"].lower() or "error" in result["final_summary"].lower():
                    st.error(result["final_summary"])
                else:
                    total_variance_label = result["path_trace"][0].replace("Overall Total Variance: ", "")
                    primary_branch_count = len(result.get("tree_data", []))
                    final_node_count = count_leaf_nodes(result.get("tree_data", []))

                    render_kpi_cards(
                        total_variance_label=total_variance_label,
                        hierarchy_count=len(hierarchy),
                        primary_branch_count=primary_branch_count,
                        final_node_count=final_node_count,
                    )

                    col_left, col_right = st.columns(2)

                    with col_left:
                        open_result_card(
                            "Executive Summary",
                            "AI-generated narrative summary based on the filtered final-level drill-down output.",
                        )
                        st.success(result["final_summary"])
                        close_result_card()

                    with col_right:
                        open_result_card(
                            "Recursive Drill-Down Trace",
                            "Expandable variance tree showing the major branches and their downstream drivers.",
                        )
                        st.info(result["path_trace"][0])
                        render_trace_tree(result.get("tree_data", []))
                        close_result_card()

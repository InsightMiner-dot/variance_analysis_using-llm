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


def initialize_hierarchy_state(all_cols: List[str], default_hierarchy: List[str], source_key: str) -> None:
    """Reset hierarchy ordering when the uploaded dataset changes."""
    if st.session_state.get("hierarchy_source_key") != source_key:
        st.session_state["hierarchy_source_key"] = source_key
        st.session_state["hierarchy_order"] = [col for col in default_hierarchy if col in all_cols]

    st.session_state["hierarchy_order"] = [
        col for col in st.session_state.get("hierarchy_order", []) if col in all_cols
    ]


def render_hierarchy_selector(all_cols: List[str], default_hierarchy: List[str], source_key: str) -> List[str]:
    """Render an ordered hierarchy picker with add, remove, and reorder actions."""
    initialize_hierarchy_state(all_cols, default_hierarchy, source_key)

    st.markdown("### Select Hierarchy Flow")
    st.caption("Arrange fields from left to right. Use the arrows to swap positions without removing everything.")

    remaining_cols = [col for col in all_cols if col not in st.session_state["hierarchy_order"]]
    add_col_left, add_col_right, add_col_reset = st.columns([3, 1, 1])

    with add_col_left:
        selected_to_add = st.selectbox(
            "Add hierarchy field",
            options=remaining_cols if remaining_cols else ["No more fields available"],
            disabled=not remaining_cols,
            key="hierarchy_add_select",
        )
    with add_col_right:
        if st.button("Add", use_container_width=True, disabled=not remaining_cols):
            st.session_state["hierarchy_order"].append(selected_to_add)
            st.rerun()
    with add_col_reset:
        if st.button("Reset", use_container_width=True):
            st.session_state["hierarchy_order"] = [col for col in default_hierarchy if col in all_cols]
            st.rerun()

    if not st.session_state["hierarchy_order"]:
        st.warning("No hierarchy selected yet. Add one or more fields to continue.")
        return []

    st.markdown("#### Current Order")
    for idx, col_name in enumerate(st.session_state["hierarchy_order"]):
        row_cols = st.columns([5, 1, 1, 1])
        with row_cols[0]:
            level_label = "Root" if idx == 0 else "Final" if idx == len(st.session_state["hierarchy_order"]) - 1 else "Middle"
            st.markdown(f"**{idx + 1}. {col_name}**")
            st.caption(f"{level_label} level")
        with row_cols[1]:
            if st.button("Up", key=f"hierarchy_up_{idx}", use_container_width=True, disabled=idx == 0):
                order = st.session_state["hierarchy_order"]
                order[idx - 1], order[idx] = order[idx], order[idx - 1]
                st.session_state["hierarchy_order"] = order
                st.rerun()
        with row_cols[2]:
            last_idx = len(st.session_state["hierarchy_order"]) - 1
            if st.button("Down", key=f"hierarchy_down_{idx}", use_container_width=True, disabled=idx == last_idx):
                order = st.session_state["hierarchy_order"]
                order[idx], order[idx + 1] = order[idx + 1], order[idx]
                st.session_state["hierarchy_order"] = order
                st.rerun()
        with row_cols[3]:
            if st.button("Remove", key=f"hierarchy_remove_{idx}", use_container_width=True):
                st.session_state["hierarchy_order"].pop(idx)
                st.rerun()

    return st.session_state["hierarchy_order"]


# ==========================================
# 4. STREAMLIT UI (SIDEBAR CONFIG)
# ==========================================
st.set_page_config(page_title="Branched Variance Analyzer", layout="wide")

st.title("Branched Root Cause Analyzer")
st.write(
    "Upload your dataset to generate a structured executive summary linking your "
    "primary categories directly to their root causes."
)

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
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    hierarchy_source_key = f"{uploaded_file.name}:{','.join(df.columns.astype(str))}"
    hierarchy = render_hierarchy_selector(all_cols, cat_cols, hierarchy_source_key)

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
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("### Executive Summary")
                        st.success(result["final_summary"])

                    with col_right:
                        st.markdown("### Recursive Drill-Down Trace")
                        st.info(result["path_trace"][0])
                        render_trace_tree(result.get("tree_data", []))

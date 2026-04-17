import streamlit as st
import pandas as pd
import os
import json
import io
import plotly.express as px
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# =========================
# ENV
# =========================
load_dotenv()

st.set_page_config(page_title="AI Decomposition Tree", layout="wide")

# =========================
# 🔷 LLM FUNCTIONS
# =========================
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

def get_summary(tree_json):
    llm = get_llm()

    prompt = f"""
You are a financial analyst.

Explain this variance tree:
{json.dumps(tree_json, indent=2)}

Provide:
1. 2-line summary
2. Key drivers
"""

    return llm.invoke(prompt).content


def get_root_path(tree_json):
    llm = get_llm()

    prompt = f"""
You are a financial analyst.

Given this decomposition tree:
{json.dumps(tree_json)}

Find the MOST impactful root cause path.

Return ONLY JSON:
{{
  "path": ["Level1", "Level2", "Level3"]
}}
"""

    response = llm.invoke(prompt)

    try:
        return json.loads(response.content)["path"]
    except:
        return []

# =========================
# 🔷 TREE BUILDER
# =========================
def build_tree(df, hierarchy, value_col, level=0):
    if level >= len(hierarchy):
        return []

    col = hierarchy[level]

    grouped = df.groupby(col)[value_col].sum()
    grouped = grouped.sort_values(key=abs, ascending=False).head(5)

    nodes = []

    for name, val in grouped.items():
        child_df = df[df[col] == name]

        node = {
            "name": str(name),
            "value": float(val)
        }

        children = build_tree(child_df, hierarchy, value_col, level + 1)

        if children:
            node["children"] = children

        nodes.append(node)

    return nodes

# =========================
# 🔷 TREE VISUALS
# =========================
def plot_tree(df, hierarchy, value_col):
    fig = px.treemap(
        df,
        path=hierarchy,
        values=value_col,
        title="Decomposition Tree"
    )
    return fig


def render_tree(nodes, highlight_path=[]):
    for node in nodes:
        label = f"{node['name']} ({node['value']/1e6:.2f}M)"

        if node["name"] in highlight_path:
            label = "🔴 " + label
        else:
            label = "🔵 " + label

        with st.expander(label):
            if "children" in node:
                render_tree(node["children"], highlight_path)

# =========================
# 🔷 UI
# =========================
st.title("🌳 AI-Powered Decomposition Tree Analyzer")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded_file is not None:

    # =========================
    # LOAD DATA
    # =========================
    if uploaded_file.name.endswith(".xlsx"):
        file_buffer = io.BytesIO(uploaded_file.getvalue())
        xls = pd.ExcelFile(file_buffer)
        sheet = st.selectbox("Select Sheet", xls.sheet_names)
        df = pd.read_excel(file_buffer, sheet_name=sheet)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    # =========================
    # CONFIG
    # =========================
    st.sidebar.header("2. Configuration")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.columns.tolist()

    has_variance = st.sidebar.checkbox("Variance column exists", True)

    if has_variance:
        variance_col = st.sidebar.selectbox("Variance Column", num_cols)
    else:
        base_col = st.sidebar.selectbox("Base Column", num_cols)
        comp_col = st.sidebar.selectbox("Compare Column", num_cols)

    hierarchy = st.sidebar.multiselect(
        "Hierarchy (Top → Bottom)",
        options=cat_cols,
        default=cat_cols[:3]
    )

    run = st.sidebar.button("Run Analysis")

    # =========================
    # RUN
    # =========================
    if run and hierarchy:

        with st.spinner("Analyzing..."):

            if has_variance:
                target_col = variance_col
            else:
                df["Variance"] = df[base_col] - df[comp_col]
                target_col = "Variance"

            total = df[target_col].sum()

            # ===== TREE =====
            tree = {
                "name": "Total Variance",
                "value": float(total),
                "children": build_tree(df, hierarchy, target_col)
            }

            # ===== LLM =====
            root_path = get_root_path(tree)
            summary = get_summary(tree)

            # =========================
            # LAYOUT
            # =========================
            left, right = st.columns([2, 1])

            # LEFT
            with left:
                st.markdown("## 🌳 Decomposition Tree")

                fig = plot_tree(df, hierarchy, target_col)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🔍 Expandable Tree")
                render_tree(tree["children"], root_path)

            # RIGHT
            with right:
                st.markdown("## 🤖 AI Insight")

                st.success(summary)

                st.markdown("### 🎯 Root Cause Path")
                st.info(" → ".join(root_path) if root_path else "Not found")

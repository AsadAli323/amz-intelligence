import streamlit as st

from common import inject_css

st.set_page_config(page_title="Amz-Intelligence", page_icon="🏷️", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()

pages = [
    st.Page("views/analyzer.py", title="Product Analyzer", icon=":material/sell:", default=True),
    st.Page("views/explorer.py", title="Market Explorer", icon=":material/insights:"),
    st.Page("views/batch.py", title="Batch Scoring", icon=":material/upload_file:"),
    st.Page("views/about.py", title="About the models", icon=":material/info:"),
]
st.navigation(pages, position="top").run()

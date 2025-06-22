import pathlib
import streamlit as st

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

st.logo(
    str(CURRENT_DIR / "logo" / "dw_intel.svg"),
    link="https://www.dw_intel.com",
    icon_image=str(CURRENT_DIR / "logo" / "favicon-32x32.png"),
)

st.set_page_config(
    page_title="Cashflow DataMap Schema Analyzer",
    page_icon="👋",
    layout="wide",
)

pg = st.navigation(
    {
        "Cashflow DataMap Schema Analyzer": [
            st.Page("pages/1_data_analyser.py", title="Data Analyser"),
            st.Page("pages/2_data_exploration.py", title="Data Exploration"),
        ]
    },
    expanded=True,
)

pg.run()

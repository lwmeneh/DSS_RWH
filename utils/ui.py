from pathlib import Path
import streamlit as st
BASE = Path(__file__).resolve().parents[1]

def load_css():
    css = (BASE / "static" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_html(name):
    html = (BASE / "templates" / name).read_text(encoding="utf-8")
    st.markdown(html, unsafe_allow_html=True)

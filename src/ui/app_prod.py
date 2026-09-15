"""
src/ui/app_prod.py — Ontario Oversight CRAG Streamlit frontend (production)
Calls the FastAPI backend over HTTP. Run with:
    streamlit run src/ui/app_prod.py

Requires the FastAPI backend to be running:
    uvicorn src.api.main:app

Set API_URL in .env to point to the backend (default: http://localhost:8000).
"""
import os, sys, base64, pathlib, tempfile
from datetime import datetime
import requests
from urllib.parse import quote
import streamlit as st
from dotenv import load_dotenv
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
load_dotenv()

from src.ui.export import build_session_docx

st.set_page_config(
    page_title="Ontario Oversight CRAG — Police Oversight Assistant",
    page_icon="⚖️", layout="wide", initial_sidebar_state="expanded",
)

# Early CSS injection to prevent white flash during loading
st.markdown("""<style>
[data-testid="stStatusWidget"],[data-testid="stStatusWidget"] *,
.stSpinner,.stSpinner *,
div[role="progressbar"],div[role="progressbar"] *,
.stProgress,.stProgress *{background:transparent!important;border-color:rgba(120,160,255,0.12)!important}
</style>""", unsafe_allow_html=True)

# ── Icons (raw SVG strings with customizable stroke color) ────────────
I_SUN    = "M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
I_MOON   = "M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
I_BOOK   = "M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
I_SEND   = "M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"
I_CHECK  = "M5 13l4 4L19 7"
I_SEARCH = "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
I_SCALE  = "M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"
I_TRASH  = "M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"

def svg_i(path_d: str, stroke: str, size: int = 16) -> str:
    inner = f'<path d="{path_d}"/>' if not path_d.startswith("<") else path_d
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;display:inline-block;">{inner}</svg>'

def svg_uri(svg: str) -> str:
    return f"data:image/svg+xml,{quote(svg, safe='')}"

# ── Theme icons ──
def svg_sun(c):
    return f'<svg width="20" height="20" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="{c}" stroke-width="4"><line x1="32" y1="8" x2="32" y2="16"/><line x1="32" y1="56" x2="32" y2="48"/><line x1="56" y1="32" x2="48" y2="32"/><line x1="8" y1="32" x2="16" y2="32"/><line x1="48.97" y1="15.03" x2="43.31" y2="20.69"/><line x1="15.03" y1="48.97" x2="20.69" y2="43.31"/><line x1="48.97" y1="48.97" x2="43.31" y2="43.31"/><line x1="15.03" y1="15.03" x2="20.69" y2="20.69"/><circle cx="32" cy="32" r="8"/></svg>'
def svg_moon(c):
    return f'<svg width="20" height="20" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="{c}" stroke-width="4"><path d="M46 44a26 26 0 0 1-24.94-33.36 24 24 0 1 0 32.3 32.3A26.24 26.24 0 0 1 46 44z"/></svg>'

# ── Chat avatars ──
def svg_qa(fill):
    return f'<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" width="24" height="24"><path fill="{fill}" d="M10.7,31.2l-3.9-5.8H3c-1.3,0-2.4-1.1-2.4-2.4v-10C.6,11.7,1.7,10.6,3,10.6h11.6V3c0-1.3,1.1-2.4,2.4-2.4h12C30.3.6,31.4,1.7,31.4,3v8c0,1.3-1.1,2.4-2.4,2.4h-2.8l-1.9,3.8-.6-.3,2-4c.1-.1.2-.2.3-.2h3c.9,0,1.6-.7,1.6-1.6V3c0-.9-.7-1.6-1.6-1.6H17c-.9,0-1.6.7-1.6,1.6v7.6H18c1.2,0,2.2.9,2.3,2h2.7v.7h-2.6v9.6c0,1.3-1.1,2.4-2.4,2.4h-7v-.7h7c.9,0,1.6-.7,1.6-1.6v-10c0-.9-.7-1.6-1.6-1.6H3c-.9,0-1.6.7-1.6,1.6v10c0,.9.7,1.6,1.6,1.6h4c.1,0,.2.1.3.2l4,6L10.7,31.2z M23.4,8.5h-.7c0-1.2.2-1.7,1-2.1.6-.3.8-.7.8-1.2,0-1-.8-1.3-1.4-1.3-.8,0-1.4.6-1.4,1.3h-.7c0-1.1,1-2,2.2-2,1.3,0,2.2.8,2.2,2,0,1.1-.7,1.5-1.1,1.8-.5.3-.7.5-.7,1.5z M11.5,18c0,.6-.4,1-1,1s-1-.4-1-1,.4-1,1-1,1,.4,1,1z M15.5,17c-.6,0-1,.4-1,1s.4,1,1,1,1-.4,1-1-.4-1-1-1z M5.5,17c-.6,0-1,.4-1,1s.4,1,1,1,1-.4,1-1-.4-1-1-1z M23,10.6c.3,0,.6-.3.6-.6s-.3-.6-.6-.6-.6.3-.6.6.3.6.6.6z"/></svg>'
def svg_book(fill):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 66 70"><g fill="{fill}"><path d="M34.2,26.9a2.8,2.8,0,0,0,2.4-2.2l-5.3-1.1c.4,1.4,1.4,3.3,2.9,3.3"/><path d="M44.3,26.9c1.4.1,2.5-1.9,2.9-3.3l-5.3,1.1a2.8,2.8,0,0,0,2.4,2.2"/><path d="M25.9,22.2a11.4,11.4,0,0,0,22.7,0V20.7c.4-1.2,3.2-10.3,1.1-14.3A6.8,6.8,0,0,0,43.9,2.4C41.1.1,34.5-2.2,27.3,3.8c-6.4,5.4-2.2,15.7-1.4,17.5Zm20.2,0a8.8,8.8,0,0,1-17.7,0V17.6c0-1.1,0-3.2,6.7-4.1a17.8,17.8,0,0,0,7-2.4,10,10,0,0,0,.7,1.2,14,14,0,0,0,2.6,2.9c.2.2.5.4.6.5v.1ZM28.9,5.8c7.7-6.4,13.4-1.4,13.6-1.2a1.3,1.3,0,0,0,1,.3,4.2,4.2,0,0,1,3.9,2.7c.6,1.2.5,3.6.2,6a14.2,14.2,0,0,1-2.7-2.8A9.7,9.7,0,0,1,43.8,8.3c-.2-.6-.4-.9-.8-.9a1.9,1.9,0,0,0-1.1.6,13.7,13.7,0,0,1-7.2,2.9c-4.8.7-7.2,2.1-8.2,3.9-.5-3.1-.3-6.7,2.4-9"/><path d="M65.4,63.3,60.1,47.4a12.5,12.5,0,0,0-11.8-8.5H44.5a3.9,3.9,0,0,0-2.7,1.2c-1.6,1.5-5.1,4-8.3,4-3.3,0-6.8-2.6-8.2-4a3.9,3.9,0,0,0-2.7-1.2H18.4A12.4,12.4,0,0,0,6.8,46.9L.5,63.1a6.9,6.9,0,0,0,6.4,9.3h5.4l.1,1.1a3.7,3.7,0,0,0,3.7,3.3H49.8a3.7,3.7,0,0,0,3.7-3.3l.1-1.1h5.4a6.9,6.9,0,0,0,6.5-9.1M49.8,74H16a.9.9,0,0,1-.9-.8L13.3,52.5a.9.9,0,0,1,.9-.9h37.5a.9.9,0,0,1,.9,1L50.6,73.2a.9.9,0,0,1-.9.8M62.2,67.9a4,4,0,0,1-3.3,1.7h-5.2l1.5-16.8a3.7,3.7,0,0,0-3.7-4h-37.5a3.7,3.7,0,0,0-3.7,4.1l1.5,16.8H6.9A4,4,0,0,1,3.1,64.1L9.4,47.9a9.6,9.6,0,0,1,9-6.1h4.2a1.3,1.3,0,0,1,.7.4c.5.5,5.1,4.8,10.2,4.8,3.8,0,7.9-2.6,10.2-4.8a1.2,1.2,0,0,1,.7-.4h3.8a9.6,9.6,0,0,1,9.1,6.6l5.4,15.9a4,4,0,0,1-.5,3.6"/><path d="M39.4,64.6a2.1,2.1,0,1,0,2.1,2.1,2.1,2.1,0,0,0-2.1-2.1"/></g></svg>'

# ── Sidebar system icons ──
def svg_i(paths, c):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
I_CPU='<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
I_DB='<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
I_LAYERS='<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
I_GIT='<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 21V9a9 9 0 0 0 9 9"/>'
I_SEARCH='<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'

# ═══════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
def is_dark():
    return st.session_state.theme == "dark"

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════

def build_css(dark):
    if dark:
        v=dict(bg="#141b2d",card="#1c2438",card_h="#222d47",inp="#1c2438",
            brd="rgba(120,160,255,0.12)",brd_h="rgba(120,160,255,0.25)",
            t1="#c9d1d9",t2="#8b95a5",t3="#586170",
            acc="#6e9eff",acc_bg="rgba(110,158,255,0.10)",
            ok="#34d399",ok_bg="rgba(52,211,153,0.10)",
            u_bg="rgba(110,158,255,0.08)",u_brd="rgba(110,158,255,0.20)",
            sb="#111827",exp="#1a2235",sl="#6e9eff",icon="#8b95a5")
    else:
        v=dict(bg="#f5f7fb",card="#ffffff",card_h="#eef2ff",inp="#eef1f6",
            brd="rgba(0,0,0,0.08)",brd_h="rgba(0,0,0,0.15)",
            t1="#2d3748",t2="#4a5568",t3="#94a3b8",
            acc="#4361ee",acc_bg="rgba(67,97,238,0.07)",
            ok="#059669",ok_bg="rgba(5,150,105,0.07)",
            u_bg="rgba(67,97,238,0.05)",u_brd="rgba(67,97,238,0.12)",
            sb="#eef1f6",exp="#f5f7fb",sl="#4361ee",icon="#4a5568")

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* === GLOBAL — kill all white surfaces === */
html,body,.stApp,.main,.main .block-container,
div[data-testid="stAppViewContainer"],div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],div[data-testid="stVerticalBlockBorderWrapper"],
.element-container,.stMarkdown,
div[data-testid="stBottom"],div[data-testid="stBottom"]>div{{
    background-color:{v["bg"]}!important;color:{v["t1"]}!important;
    font-family:'DM Sans',-apple-system,sans-serif!important}}
/* Kill ALL top padding on main containers so logos touch the top */
.main .block-container{{padding-top:0!important;margin-top:0!important}}
div[data-testid="stMainBlockContainer"]{{padding-top:0!important;margin-top:0!important}}
div[data-testid="stAppViewContainer"]>div:first-child{{padding-top:0!important}}
div[data-testid="stVerticalBlockBorderWrapper"]{{padding-top:0!important;margin-top:0!important}}
div[data-testid="stVerticalBlockBorderWrapper"]>div{{padding-top:0!important;margin-top:0!important}}
/* Also kill Streamlit's default block container padding */
.block-container{{padding:0 1rem!important;max-width:100%!important}}
header[data-testid="stHeader"]{{display:none!important}}
#MainMenu,footer,.stDeployButton{{display:none!important}}

/* === LOADING / SPINNER / PROGRESS BAR — no white at all === */
div[data-testid="stStatusWidget"],
div[data-testid="stStatusWidget"]>div,
div[data-testid="stStatusWidget"] label,
div[data-testid="stStatusWidget"] div[role="status"],
.stSpinner,.stSpinner>div,
div[data-testid="stCachedStResourceSpinner"],
div[data-testid="stCachedStResourceSpinner"]>div,
div.stSpinner>div>div{{
    background-color:{v["bg"]}!important;
    color:{v["t2"]}!important;
    border-color:{v["brd"]}!important}}
/* Progress bar track and fill */
div[role="progressbar"],
div[role="progressbar"]>div,
div[role="progressbar"]>div>div,
div[data-testid="stStatusWidget"] div[role="progressbar"],
.stProgress,.stProgress>div,.stProgress>div>div,
div[data-testid="stStatusWidget"] *[role="progressbar"],
div[data-testid="stStatusWidget"] *[role="progressbar"]>*{{
    background-color:{v["card"]}!important;
    background:{v["card"]}!important}}
/* The filled portion */
div[role="progressbar"]>div>div>div,
.stProgress>div>div>div{{
    background-color:{v["acc"]}!important;
    background:{v["acc"]}!important}}
/* Spinner circle */
div[data-testid="stStatusWidget"] svg circle,
.stSpinner svg circle{{stroke:{v["acc"]}!important}}

/* === SIDEBAR === */
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{{
    background:{v["sb"]}!important;border-right:1px solid {v["brd"]}!important}}
section[data-testid="stSidebar"]>div,
section[data-testid="stSidebar"]>div>div,
section[data-testid="stSidebar"]>div>div>div{{
    padding-top:0!important;margin-top:0!important}}
/* Sidebar collapse button — float it so it doesn't eat vertical space */
section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"]{{
    position:absolute!important;top:6px!important;right:6px!important;z-index:100!important}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{{gap:.15rem!important}}
section[data-testid="stSidebar"] *{{color:{v["t1"]}!important}}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span{{
    color:{v["t2"]}!important;font-size:.92rem!important;line-height:1.4!important}}
section[data-testid="stSidebar"] h3{{font-size:.68rem!important;font-weight:500!important;
    text-transform:uppercase!important;letter-spacing:.1em!important;
    color:{v["t3"]}!important;margin:.15rem 0 .05rem!important}}
section[data-testid="stSidebar"] hr{{border-color:{v["brd"]}!important;margin:.25rem 0!important}}
section[data-testid="stSidebar"] .stButton>button{{
    background:{v["card"]}!important;border:1px solid {v["brd"]}!important;
    color:{v["t2"]}!important;font-size:.86rem!important;font-weight:400!important;
    text-align:left!important;padding:5px 10px!important;border-radius:8px!important;
    transition:all .2s!important;width:100%!important;margin:0!important}}
section[data-testid="stSidebar"] .stButton>button:hover{{
    background:{v["card_h"]}!important;border-color:{v["brd_h"]}!important;
    color:{v["t1"]}!important;transform:translateX(2px)!important}}
div[data-testid="stSlider"] label{{color:{v["t2"]}!important;font-size:.92rem!important}}
div[data-testid="stSlider"] div[role="slider"]{{background:{v["sl"]}!important}}
div[data-testid="stSlider"] p{{color:{v["t2"]}!important}}
section[data-testid="stSidebar"] div[data-testid="stSlider"]{{padding-bottom:0!important;margin-bottom:0!important}}

/* === THEME TOGGLE === */
.theme-btn{{position:fixed;top:12px;right:20px;z-index:999999;
    background:{v["card"]};border:1px solid {v["brd"]};border-radius:10px;
    width:40px;height:40px;display:flex;align-items:center;justify-content:center;
    cursor:pointer;transition:all .25s;box-shadow:0 2px 8px rgba(0,0,0,.12)}}
.theme-btn:hover{{border-color:{v["brd_h"]};transform:scale(1.08)}}
button[kind="primary"][data-testid="stBaseButton-primary"]{{
    position:fixed!important;top:12px!important;right:20px!important;
    width:40px!important;height:40px!important;opacity:0!important;
    z-index:9999999!important;cursor:pointer!important;padding:0!important;min-height:0!important}}

/* === HERO === */
.hero-badge{{display:inline-flex;align-items:center;gap:6px;background:{v["acc_bg"]};
    border:1px solid {v["acc"]}33;color:{v["acc"]};font-size:.76rem;font-weight:500;
    letter-spacing:.06em;text-transform:uppercase;padding:5px 14px;border-radius:100px;
    margin-bottom:12px;font-family:'JetBrains Mono',monospace}}
.hero-title{{font-size:2.2rem;font-weight:700;letter-spacing:-.04em;line-height:1.1;color:{v["acc"]};margin:0 0 8px}}
.hero-sub{{font-size:1rem;font-weight:300;color:{v["t2"]};margin:0;line-height:1.5}}
.status-bar{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}}
.s-chip{{display:inline-flex;align-items:center;gap:5px;font-size:.76rem;font-weight:500;
    font-family:'JetBrains Mono',monospace;padding:4px 10px;border-radius:6px;border:1px solid}}
.s-ok{{background:{v["ok_bg"]};border-color:{v["ok"]}33;color:{v["ok"]}}}
.s-info{{background:{v["acc_bg"]};border-color:{v["acc"]}22;color:{v["acc"]}}}
.hero-line{{height:1px;background:linear-gradient(90deg,{v["acc"]},transparent 60%);margin-top:16px;opacity:.2}}

/* === CHAT === */
.stChatMessage{{background:transparent!important;border:none!important}}
.stChatMessage p,.stChatMessage li,.stChatMessage span{{color:{v["t1"]}!important;font-size:1rem!important;line-height:1.7!important}}
.stChatMessage strong{{color:{v["acc"]}!important;font-weight:600!important}}
.stChatMessage code{{font-family:'JetBrains Mono',monospace!important;background:{v["acc_bg"]}!important;
    border:1px solid {v["acc"]}22!important;padding:1px 6px!important;border-radius:4px!important;
    font-size:.85em!important;color:{v["acc"]}!important}}
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])>div:last-child{{
    background:{v["u_bg"]}!important;border:1px solid {v["u_brd"]}!important;
    border-radius:14px 14px 4px 14px!important;padding:14px 18px!important}}
.stChatMessage[data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"]))>div:last-child{{
    background:{v["card"]}!important;border:1px solid {v["brd"]}!important;
    border-radius:4px 14px 14px 14px!important;padding:14px 18px!important}}

/* === AVATAR — remove circle background, keep SVG from file === */
div[data-testid="chatAvatarIcon-user"],
div[data-testid="chatAvatarIcon-assistant"],
.stChatMessage div[data-testid="chatAvatarIcon-user"],
.stChatMessage div[data-testid="chatAvatarIcon-assistant"]{{
    background:transparent!important;border:none!important;border-radius:0!important}}
/* Also target the img inside avatar containers */
div[data-testid="chatAvatarIcon-user"] img,
div[data-testid="chatAvatarIcon-assistant"] img{{
    border-radius:0!important}}

/* === CHAT INPUT === */
div[data-testid="stChatInput"],div[data-testid="stChatInput"]>div{{background:{v["bg"]}!important}}
div[data-testid="stChatInput"] textarea{{background:{v["inp"]}!important;color:{v["t1"]}!important;
    border:1px solid {v["brd"]}!important;border-radius:12px!important;
    font-family:'DM Sans',sans-serif!important;font-size:1rem!important}}
div[data-testid="stChatInput"] textarea::placeholder{{color:{v["t3"]}!important}}
div[data-testid="stChatInput"] button{{background:{v["acc"]}!important;border-radius:10px!important;border:none!important}}
div[data-testid="stChatInput"] button svg{{stroke:#fff!important}}

/* === EXPANDERS / SOURCES === */
details{{background:{v["card"]}!important;border:1px solid {v["brd"]}!important;border-radius:10px!important}}
details summary{{color:{v["t2"]}!important;font-size:.92rem!important}}
details summary:hover{{color:{v["t1"]}!important}}
details>div{{background:{v["exp"]}!important}}
.src-card{{background:{v["card"]};border:1px solid {v["brd"]};padding:10px 14px;border-radius:8px;margin:4px 0;
    font-size:.88rem;display:flex;justify-content:space-between;align-items:center;transition:background .2s}}
.src-card:hover{{background:{v["card_h"]}}}
.src-cite{{font-family:'JetBrains Mono',monospace;font-weight:500;color:{v["acc"]};font-size:.82rem}}
.src-ttl{{color:{v["t2"]};margin-left:8px;font-weight:300}}
.src-sc{{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:{v["ok"]};background:{v["ok_bg"]};padding:2px 8px;border-radius:4px}}

/* === INFO CARD === */
.info-card{{background:{v["card"]};border:1px solid {v["brd"]};padding:10px 12px;border-radius:8px;font-size:.88rem}}
.info-row{{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid {v["brd"]}}}
.info-row:last-child{{border-bottom:none}}
.info-lbl{{font-size:.7rem;font-weight:500;text-transform:uppercase;letter-spacing:.05em;color:{v["t3"]}!important;display:flex;align-items:center;gap:5px}}
.info-val{{color:{v["t1"]}!important;font-weight:400;font-family:'JetBrains Mono',monospace;font-size:.8rem}}

.stAlert>div{{background:{v["card"]}!important;color:{v["t1"]}!important;border:1px solid {v["brd"]}!important;border-radius:10px!important}}
.stAlert p{{color:{v["t1"]}!important}}
::-webkit-scrollbar{{width:5px}}::-webkit-scrollbar-track{{background:transparent}}::-webkit-scrollbar-thumb{{background:{v["brd"]};border-radius:3px}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.fade-in{{animation:fadeIn .35s ease forwards}}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════

st.markdown(build_css(is_dark()), unsafe_allow_html=True)

# Theme toggle
ic = "#8b95a5" if is_dark() else "#4a5568"
st.markdown(f'<div class="theme-btn" title="Toggle theme">{svg_sun(ic) if is_dark() else svg_moon(ic)}</div>', unsafe_allow_html=True)
if st.button("t", key="theme_btn_hidden", type="primary"):
    st.session_state.theme = "light" if is_dark() else "dark"
    st.rerun()

# Themed accent color
accent_color = "#7ab0ff" if is_dark() else "#1a365d"

# Write themed avatar SVGs to temp dir (cross-platform)
_tmp = tempfile.gettempdir()
_qa_path = os.path.join(_tmp, "ontario_crag_qa_avatar.svg")
_book_path = os.path.join(_tmp, "ontario_crag_book_avatar.svg")
with open(_qa_path, "w") as f:
    f.write(svg_qa(accent_color))
with open(_book_path, "w") as f:
    f.write(svg_book(accent_color))

# ── Backend connection ────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

@st.cache_resource(show_spinner="Connecting to Ontario Oversight CRAG backend...")
def initialise_pipeline():
    res = requests.get(f"{API_URL}/health", timeout=10)
    res.raise_for_status()
    data = res.json()
    if data.get("status") != "ok":
        raise Exception("Backend not ready")
    return data.get("chunks", 0)

def query_backend(question: str) -> dict:
    res = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120)
    res.raise_for_status()
    return res.json()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Ontario Police Oversight Assistant**")
    st.divider()
    st.markdown("### About")
    st.markdown("AI-powered QA grounded in the **Community Safety and Policing Act, 2019 (CSPA)**, Ontario Regulations (O.Regs), and **LECA** guidelines. Answers cite specific sections.")
    st.divider()
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b")
    st.divider()
    st.markdown("### System")
    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-row"><span class="info-lbl">{svg_i(I_CPU,ic)} Model</span><span class="info-val">{model_name}</span></div>'
        f'<div class="info-row"><span class="info-lbl">{svg_i(I_DB,ic)} Source</span><span class="info-val">CSPA + O.Regs + LECA (~1,868 chunks)</span></div>'
        f'<div class="info-row"><span class="info-lbl">{svg_i(I_LAYERS,ic)} Retrieval</span><span class="info-val">Hybrid</span></div>'
        f'<div class="info-row"><span class="info-lbl">{svg_i(I_GIT,ic)} Fusion</span><span class="info-val">RRF (k=60)</span></div>'
        f'</div>', unsafe_allow_html=True)
    st.divider()
    if st.button("Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()
    st.markdown("### Try asking")
    for q in ["What are the duties of a chief of police?","How can a complaint be made against a police officer?","What is adequate and effective policing?","What powers does the Inspector General have?","Requirements for a police service board?"]:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state.pending_question = q
            st.rerun()
    st.divider()
    # ── Export session ─────────────────────────────────────────────────
    if st.session_state.get("messages"):
        _src = st.session_state.get("sources", {})
        docx_bytes = build_session_docx(st.session_state["messages"], _src)
        st.download_button(
            label="📄 Export Session to Word",
            data=docx_bytes,
            file_name=f"ontario_crag_session_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="export_btn",
        )
    else:
        st.caption("Ask a question to enable export.")

# ── Hero ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fade-in" style="margin-top:8px;">
    <div class="hero-badge">{svg_i(I_SEARCH, "#6e9eff" if is_dark() else "#4361ee")} AI-Powered Legislative Analysis</div>
    <h1 class="hero-title">Ontario Oversight CRAG</h1>
    <p class="hero-sub">Corrective RAG Pipeline for the CSPA 2019, Ontario Regulations, and LECA Guidelines — every answer cites specific legislative sections</p>
    <div class="status-bar">
        <span class="s-chip s-ok">&#9679; Pipeline Ready</span>
        <span class="s-chip s-info">Hybrid Retrieval</span>
        <span class="s-chip s-info">RRF Fusion</span>
    </div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# Spacer before chat area
st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

# ── State & pipeline ─────────────────────────────────────────────────
for k,d in [("messages",[]),("sources",{}),("pending_question",None)]:
    if k not in st.session_state: st.session_state[k]=d
try:
    initialise_pipeline()
    pipeline_ready=True
except Exception as e:
    st.error(f"Failed to connect to backend: {e}")
    st.info(f"Make sure the FastAPI backend is running: uvicorn src.api.main:app\nAPI_URL={API_URL}")
    pipeline_ready=False

def render_confidence(label: str):
    if label == "High":
        st.markdown("🟢 **High Confidence**")
    elif label == "Medium":
        st.markdown("🟡 **Medium Confidence**")
    else:
        st.markdown("🔴 **Low Confidence**")

def render_sources(sl):
    if not sl: return
    with st.expander(f"{len(sl)} legislative sections retrieved"):
        for s in sl:
            st.markdown(f'<div class="src-card"><div><span class="src-cite">{s.get("citation","?")}</span><span class="src-ttl">{s.get("section_title","")}</span></div><span class="src-sc">{s.get("rrf_score",0):.3f}</span></div>',unsafe_allow_html=True)

for i,msg in enumerate(st.session_state.messages):
    av = _qa_path if msg["role"]=="user" else _book_path
    with st.chat_message(msg["role"], avatar=av):
        if msg["role"]=="assistant" and msg.get("confidence"):
            render_confidence(msg["confidence"])
        st.markdown(msg["content"])
        if msg["role"]=="assistant":
            render_sources(st.session_state.sources.get(i,[]) if isinstance(st.session_state.sources,dict) else [])
            _js_str = json.dumps(msg["content"])
            st.components.v1.html(
                f"""<script>var _copyText={_js_str};</script>
<button onclick="var t=_copyText;if(navigator.clipboard){{navigator.clipboard.writeText(t).then(()=>{{this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}})}}else{{var ta=document.createElement('textarea');ta.style.position='fixed';ta.style.opacity='0';ta.value=t;try{{document.body.appendChild(ta);ta.select();document.execCommand('copy');this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}}finally{{document.body.removeChild(ta)}}}}"
    style="background:none;border:1px solid #555;color:#aaa;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;">Copy answer</button>""",
                height=36,
            )

if st.session_state.pending_question:
    q=st.session_state.pending_question; st.session_state.pending_question=None
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("assistant", avatar=_book_path):
        with st.spinner("Searching legislation..."):
            r=query_backend(q); a=r.get("answer","No answer."); ret=r.get("sources",[])
            conf=r.get("confidence","Low")
            render_confidence(conf); st.markdown(a); st.session_state.sources[len(st.session_state.messages)]=ret
            render_sources(ret)
            _js_str = json.dumps(a)
            st.components.v1.html(
                f"""<script>var _copyText={_js_str};</script>
<button onclick="var t=_copyText;if(navigator.clipboard){{navigator.clipboard.writeText(t).then(()=>{{this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}})}}else{{var ta=document.createElement('textarea');ta.style.position='fixed';ta.style.opacity='0';ta.value=t;try{{document.body.appendChild(ta);ta.select();document.execCommand('copy');this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}}finally{{document.body.removeChild(ta)}}}}"
    style="background:none;border:1px solid #555;color:#aaa;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;">Copy answer</button>""",
                height=36,
            )
            st.session_state.messages.append({"role":"assistant","content":a,"confidence":conf})
    st.rerun()

if pipeline_ready:
    if prompt:=st.chat_input("Ask a question about Ontario police oversight..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar=_qa_path): st.markdown(prompt)
        with st.chat_message("assistant", avatar=_book_path):
            with st.spinner("Searching legislation..."):
                try:
                    r=query_backend(prompt); a=r.get("answer","No answer."); ret=r.get("sources",[])
                    conf=r.get("confidence","Low")
                    render_confidence(conf); st.markdown(a); idx=len(st.session_state.messages)
                    if isinstance(st.session_state.sources,dict): st.session_state.sources[idx]=ret
                    render_sources(ret)
                    _js_str = json.dumps(a)
                    st.components.v1.html(
                        f"""<script>var _copyText={_js_str};</script>
<button onclick="var t=_copyText;if(navigator.clipboard){{navigator.clipboard.writeText(t).then(()=>{{this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}})}}else{{var ta=document.createElement('textarea');ta.style.position='fixed';ta.style.opacity='0';ta.value=t;try{{document.body.appendChild(ta);ta.select();document.execCommand('copy');this.innerText='Copied!';setTimeout(()=>this.innerText='Copy answer',1500)}}finally{{document.body.removeChild(ta)}}}}"
    style="background:none;border:1px solid #555;color:#aaa;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;">Copy answer</button>""",
                        height=36,
                    )
                    st.session_state.messages.append({"role":"assistant","content":a,"confidence":conf})
                except Exception as e:
                    st.error(str(e)); st.session_state.messages.append({"role":"assistant","content":str(e)})

"""
app.py — לוח בחירות עם שמירת פרטיות  (Streamlit)
==================================================
הרצה:  streamlit run app.py
דרישות: voting_dp.py באותה תיקייה.
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from voting_dp import (
    randomized_response,
    estimate_rr_frequency,
    laplace_mechanism,
    k_randomized_response,
    estimate_krr_frequency,
    laplace_margin_of_error,
)

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

SHOW_PRIVACY_SETTINGS = True
ACTIVIST_NAME         = "א. כהן  |  מטה תל-אביב"

# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="ElectorApp — לוח DP",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS
# =============================================================================

SEED           = 42
N_VOTERS       = 180
PARTY_NAMES    = ["חירות", "קדמה", "אחדות", "מורשת", "ירוקים"]
ORGANISER      = PARTY_NAMES[0]
DEFECTION_RATE = 0.25
CITIES = [
    "תל אביב", "ירושלים", "חיפה",
    "באר שבע", "נתניה", "ראשון לציון",
]

DEFAULT_EPS_VOTE  = 1.0
DEFAULT_EPS_PARTY = 1.5
DEFAULT_EPS_COUNT = 1.0

PARTY_COLORS = {
    "חירות"  : "#4C72B0",
    "קדמה"   : "#DD8452",
    "אחדות"  : "#55A868",
    "מורשת"  : "#C44E52",
    "ירוקים" : "#8172B2",
}
STATUS_VOTED   = "#27ae60"
STATUS_PENDING = "#e74c3c"
STATUS_MISSING = "#e67e22"

# =============================================================================
# GLOBAL STYLES
# =============================================================================

def inject_styles():
    """
    CSS strategy:
    - Heebo Hebrew font
    - Full light theme (no dark colours on any component)
    - Targeted text-align:right for ALL Hebrew content without touching layout
    - <p> tags and widget labels explicitly right-aligned
    - Title on LEFT of top bar; user badge on RIGHT
    - Stationary sidebar (collapse button hidden)
    """
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Heebo'
        ':wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <style>
    /* ── Font — text elements only, never * ─────────────────────────────
       Using  *  would override Streamlit's icon fonts (Material Icons),
       turning the expander arrow SVG into its text fallback "arrow_down".
       We list every text-bearing element explicitly instead.             */
    html, body,
    p, div, label, input, textarea, select, button,
    h1, h2, h3, h4, h5, h6, li, td, th, caption, figcaption,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stWidgetLabel"],
    .page-title, .user-name, .top-bar-title,
    .top-bar-subtitle {
        font-family: 'Heebo', 'Segoe UI', Arial, sans-serif !important;
    }
    

    /* ── Light theme: backgrounds ───────────────────────────────────── */
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fa !important;
        color: #1a2340 !important;
    }
    [data-testid="stMain"] {
        background-color: #f5f7fa !important;
        color: #1a2340 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f4f8 !important;
        border-right: 1px solid #dde4ee !important;
    }

    /* ── Light theme: sidebar text ───────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #1a2340 !important;
    }

    /* ── Sidebar section headings (####) ─────────────────────────────── */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #2d4a8a !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin: 18px 0 8px 0 !important;
        padding-bottom: 4px !important;
        border-bottom: 1px solid #c8d5e8 !important;
    }

    /* ── Light theme: metrics ────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #dde4ee !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: #1a2340 !important; }
    [data-testid="stMetricLabel"] { color: #1a2340 !important; }
    [data-testid="stMetricDelta"] {
        color: #3b6cb7 !important;
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
    [data-testid="stMetricDelta"] > div {
        display: flex !important;
        flex-direction: row-reverse !important;
    }

    /* ── Light theme: alerts ─────────────────────────────────────────── */
    [data-testid="stAlert"], div[role="alert"] {
        background-color: #eef4ff !important;
        color: #1a2340 !important;
        border-color: #a8c0e0 !important;
    }

    /* ── Light theme: inputs ─────────────────────────────────────────── */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a2340 !important;
        border: 1px solid #c0ccde !important;
    }
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #1a2340 !important;
        border-color: #c0ccde !important;
    }

    /* ── Light theme: sliders, dataframes, borders ───────────────────── */
    [data-testid="stSlider"] * { color: #1a2340 !important; }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * {
        background-color: #ffffff !important;
        color: #1a2340 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-color: #dde4ee !important;
    }
    hr { border-color: #dde4ee !important; }

    /* ── Captions and labels: black (not grey) ───────────────────────── */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    small, caption { color: #1a2340 !important; direction: rtl !important;
    }

    /* ── Hide default Streamlit chrome ───────────────────────────────── */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"]      { display: none !important; }
    [data-testid="stDecoration"]   { display: none !important; }
    #MainMenu                      { display: none !important; }
    footer                         { display: none !important; }

    /* ── Stationary sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="collapsedControl"]        { display: none !important; }

    /* ── Push content below top bar ──────────────────────────────────── */
    [data-testid="stAppViewContainer"] > section:first-child {
        padding-top: 68px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 68px !important;
    }

    /* ── Top bar: title LEFT, user badge RIGHT ────────────────────────── */
    .top-bar {
        position: fixed; top: 0; left: 0; right: 0; height: 58px;
        background: linear-gradient(90deg, #2d4a8a 0%, #3b6cb7 100%);
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 20px; z-index: 999999;
        box-shadow: 0 2px 8px rgba(45,74,138,0.25);
    }
    .top-bar-brand {
        display: flex; align-items: center; gap: 10px;
    }
    .top-bar-title {
        font-size: 21px; font-weight: 700; color: #ffffff;
        display: flex; align-items: center; gap: 10px;
    }
    .top-bar-subtitle { font-size: 12px; color: #c0d4f0; margin-top: 2px; }
    .user-badge {
        display: flex; align-items: center; gap: 9px;
        background: rgba(255,255,255,0.15);
        border-radius: 28px; padding: 6px 14px 6px 10px;
    }
    .user-avatar {
        width: 30px; height: 30px; border-radius: 50%; background: #5a8fd8;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; color: #ffffff; font-weight: 700; flex-shrink: 0;
    }
    .user-name { font-size: 13px; color: #e4edff; font-weight: 500; }

    /* ── Sidebar brand ───────────────────────────────────────────────── */
    .sidebar-brand {
        font-size: 17px; font-weight: 700; color: #2d4a8a;
        padding: 0 4px 12px 4px;
        border-bottom: 2px solid #c8d5e8; margin-bottom: 8px;
    }

    /* ── Page title ──────────────────────────────────────────────────── */
    .page-title {
        font-size: 2.3rem !important; font-weight: 800 !important;
        color: #1a2340 !important; margin-bottom: 4px; line-height: 1.2;
        text-align: right !important;
    }

    /* ── All buttons: light theme ────────────────────────────────────── */
    button,
    .stButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        background-color: #dde8f8 !important;
        color: #1a2340 !important;
        border: 1px solid #a8bede !important;
        border-radius: 8px !important;
        font-family: 'Heebo', sans-serif !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: background-color 0.15s, border-color 0.15s !important;
    }
    button:hover, .stButton > button:hover {
        background-color: #c4d6f2 !important;
        border-color: #3b6cb7 !important;
        color: #1a2340 !important;
    }

    /* ── Sidebar nav buttons ─────────────────────────────────────────── */
    div[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;
        border: 1px solid #dde4ee !important;
        color: #1a2340 !important;
        font-size: 14px !important;
        padding: 9px 12px !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 3px !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #e8f0fc !important;
        border-color: #3b6cb7 !important;
    }
    div[data-testid="stSidebar"] .active-nav .stButton > button {
        background-color: #e4edfb !important;
        border: 1px solid #3b6cb7 !important;
        border-left: 4px solid #3b6cb7 !important;
        color: #1a2340 !important;
        font-weight: 700 !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       HEBREW TEXT ALIGNMENT — right-aligns content text.
       Scope: content elements only; layout containers are untouched
       so Streamlit's sidebar / column engine is unaffected.
       To switch to left-aligned: change every "right" to "left" below.
       ══════════════════════════════════════════════════════════════════ */

    /* All <p> tags — catches every paragraph in the app */
    p { text-align: right !important; }

    /* Metric cards — align BOTH label and value to the right */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {
        text-align: right !important;
        display: block !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        text-align: right !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] div {
        text-align: right !important;
        display: flex !important;
        flex-direction: row-reverse !important;
    }
    [data-testid="metric-container"] {
        text-align: right !important;
    }

    /* Markdown containers (st.markdown, st.write) */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] blockquote,
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {
        text-align: right !important;
        direction: rtl !important;
    }
    
    .top-bar {
    direction: rtl !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] p {
        text-align: right !important;
    }

    /* Widget labels (slider, selectbox, text input) */
    [data-testid="stWidgetLabel"] {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        width: 100% !important;
        text-align: right !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        text-align: right !important;
    }

    /* Sidebar widget labels specifically */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        text-align: right !important;
    }

    /* Custom HTML elements */
    .page-title, .sidebar-brand { text-align: right !important; }

    /* ── Expander: title right-aligned; content p/li right-aligned ──────
       We use direction:rtl ONLY on the summary element so the title text
       reads right-to-left.  This does not affect the overall layout.
       The expand/collapse icon is an SVG; direction:rtl moves it to the
       left, which is the correct position for Hebrew/RTL expanders.     */
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details summary p {
        direction: rtl !important;
        text-align: right !important;
    }
    /* Hide the Material Icons span (keyboard_arrow_right) and replace
       with a pure-CSS chevron — no icon font dependency at all.       */
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpander"] details summary span[role="img"],
    [data-testid="stExpander"] details summary svg {
        display: none !important;
        font-size: 0 !important;
    }
    [data-testid="stExpander"] details summary::before {
        content: "▶";
        font-size: 11px;
        color: #3b6cb7;
        margin-left: 8px;
        margin-right: 2px;
        display: inline-block;
        font-family: inherit !important;
    }
    [data-testid="stExpander"] details[open] summary::before {
        content: "▼";
    }
    /* Content inside the expander */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] ul,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] p,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li {
        text-align: right !important;
        direction: rtl !important;
        width: 100%
    }
    /* ── Info tooltip icon ──────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] { overflow: visible !important; }
    .info-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        vertical-align: middle;
    }
    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 15px; height: 15px;
        background: #3b6cb7;
        color: #ffffff;
        border-radius: 50%;
        font-size: 10px; font-weight: 700;
        line-height: 1;
        vertical-align: middle;
        margin-right: 5px;
        font-style: normal;
    }
    .tooltip-text {
        visibility: hidden;
        background: #1a2340;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 14px;
        position: absolute;
        z-index: 9999;
        bottom: 130%; right: 0;
        width: 310px;
        font-size: 14px; line-height: 1.6;
        opacity: 0;
        transition: opacity 0.2s ease;
        direction: rtl; text-align: right;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        pointer-events: none;
    }
    .info-tooltip:hover .tooltip-text { visibility: visible; opacity: 1; }
    /* ══════════════════════════════════════════════════════════════════ */
    </style>
    """, unsafe_allow_html=True)
    # Block Ctrl+C from triggering Streamlit's "Clear cache" shortcut.
    # stopImmediatePropagation prevents Streamlit's React handlers from seeing
    # the event; preventDefault is intentionally omitted so the browser's
    # native copy action still works normally.
    components.html("""
    <script>
    try {
        window.parent.document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'c'
                    && !e.shiftKey && !e.altKey) {
                e.stopImmediatePropagation();
            }
        }, true);
    } catch(err) {}
    </script>
    """, height=0)


# =============================================================================
# TOP BAR  (title LEFT, user badge RIGHT)
# =============================================================================

def render_top_bar(activist_name):
    """Fixed top bar: ElectorApp title on the LEFT, user badge on the RIGHT."""
    initials = "".join(w[0] for w in activist_name.split()
                       if w and w[0].isalpha())[:2].upper() or "U"
    st.markdown(
        f"""
        <div class="top-bar">
            <div class="top-bar-brand">
                <div class="top-bar-title">🗳️ Differentially Private Elector (PoC)</div>
                <div class="top-bar-subtitle">ניהול משמר פרטיות של מערכת הבחירות</div>
            </div>
            <div class="user-badge">
                <div class="user-avatar">{initials}</div>
                <div class="user-name">{activist_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _static_bar(value, label):
    """Pure HTML progress bar — one-way render, cannot trigger reruns."""
    pct   = int(round(min(value, 1.0) * 100))
    color = "#27ae60" if pct >= 70 else "#e67e22" if pct >= 40 else "#e74c3c"
    st.markdown(
        f'<div style="margin:6px 0 12px 0;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:12px;margin-bottom:3px;">'
        f'<span>{label}</span><span>{pct}%</span></div>'
        f'<div style="background:#e0e4ec;border-radius:6px;height:8px;">'
        f'<div style="background:{color};width:{pct}%;height:8px;'
        f'border-radius:6px;"></div></div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# DP ACCURACY BANNER
# =============================================================================

def render_accuracy_banner(eps_vote, eps_count, n_reported):
    p_rr       = np.exp(eps_vote) / (1.0 + np.exp(eps_vote))
    flip       = 1.0 - p_rr
    lap_margin = laplace_margin_of_error(sensitivity=1.0, epsilon=eps_count)

    expected_city_size = N_VOTERS / len(CITIES)
    lap_accuracy = max(0.0, 1.0 - lap_margin / expected_city_size)

    if min(float(eps_vote), float(eps_count)) >= 2.0:
        level_icon, level_txt = "🟢", "דיוק גבוה (הגנת פרטיות חלשה)"
    elif min(float(eps_vote), float(eps_count)) >= 1.0:
        level_icon, level_txt = "🟡", "דיוק בינוני"
    else:
        level_icon, level_txt = "🔴", "דיוק נמוך (הגנת פרטיות חזקה)"

    with st.expander(f"{level_icon}  הערכת דיוק הנתונים — {level_txt}", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown(
                    "📊 **סטטוס הצבעה** "
                    "<span class='info-tooltip'><span class='info-icon'>i</span>"
                    "<span class='tooltip-text'>"
                    "Binary Randomized Response (LDP) — מנגנון פרטיות מקומית. "
                    "כל דיווח מוצפן לפני שמירה: בהסתברות p הדיווח זהה לאמת, "
                    "בהסתברות 1-p הוא מתהפך אקראית כדי להגן על פרטיות המשתמש."
                    "</span></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**תקציב פרטיות: ε = {eps_vote}**")
                st.divider()
                n_correct = int(round(n_reported * p_rr))
                n_flipped = n_reported - n_correct
                if n_reported > 0:
                    st.markdown(
                        f"- ✅ **דיווחים נכונים:** {p_rr:.1%} "
                        f"({n_correct} מתוך {n_reported} מצביעים מדווחים)\n"
                        f"- 🔀 **הפוכים עקב הגנת פרטיות:** {flip:.1%} "
                        f"({n_flipped} מתוך {n_reported} מצביעים מדווחים)\n"
                        f"- **דיוק כולל: {p_rr:.0%}**"
                    )
                else:
                    st.markdown(
                        f"- ✅ **שיעור דיווחים נכונים:** {p_rr:.1%}\n"
                        f"- 🔀 **שיעור הפוכים עקב הגנת פרטיות:** {flip:.1%}\n"
                        f"- **דיוק כולל: {p_rr:.0%}**"
                    )
                _static_bar(p_rr, f"דיוק: {p_rr:.0%}")

        with col2:
            with st.container(border=True):
                st.markdown(
                    "🏙️ **ספירות עיר** "
                    "<span class='info-tooltip'><span class='info-icon'>i</span>"
                    "<span class='tooltip-text'>"
                    "מנגנון Laplace (DP) — מנגנון פרטיות גלובלית. "
                    "רעש אקראי המחולק לפי Laplace מתווסף לכל ספירה, "
                    "כך שלא ניתן לזהות את תרומתו של אדם בודד לתוצאה."
                    "</span></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**תקציב פרטיות: ε = {eps_count}**")
                st.divider()
                example = int(round(expected_city_size))
                lo_ex   = max(0, round(example - lap_margin))
                hi_ex   = round(example + lap_margin)
                st.markdown(
                    f"- 📏 **שגיאת דגימה:** ±{lap_margin:.0f} מצביעים (רווח סמך 95%)\n"
                    f"- 📊 **דוגמה:** עבור ספירה מדווחת של {example} מצביעים — "
                    f"הערך האמיתי נע בין {lo_ex} ל-{hi_ex} בסבירות 95%\n"
                    f"- **דיוק כולל: {lap_accuracy:.0%}**"
                )
                _static_bar(lap_accuracy, f"דיוק: {lap_accuracy:.0%}")

        st.markdown(
            "<div style='direction:rtl;text-align:right;margin-top:8px;margin-bottom:8px;padding:8px;"
            "background:#fffbe6;border-radius:8px;border:1px solid #ffe58f;'>"
            "<strong style='font-size:16px;'>"
            "💡 ε גבוה יותר = דיוק גבוה יותר, פרטיות נמוכה יותר.  "
            "ε נמוך יותר = פרטיות גבוהה יותר, שגיאה גדולה יותר.</strong></div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """
    Two navigation items (merged dashboard + voter list) and optional
    privacy settings.  Returns (current_page, eps_vote, eps_party, eps_count).
    """
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    nav_items = [
        ("dashboard", "📊  לוח ראשי"),
        ("voters",    "📋  רשימת מצביעים"),
    ]

    with st.sidebar:
        st.markdown("#### 🗳️ תפריטים", unsafe_allow_html=True)
        for page_key, label in nav_items:
            is_active = st.session_state.current_page == page_key
            open_tag  = '<div class="active-nav">' if is_active else "<div>"
            st.markdown(open_tag, unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        eps_vote  = DEFAULT_EPS_VOTE
        eps_party = DEFAULT_EPS_PARTY
        eps_count = DEFAULT_EPS_COUNT

        if SHOW_PRIVACY_SETTINGS:
            st.markdown("#### 🔒 הגדרות פרטיות")
            eps_vote  = st.slider("ε — סטטוס הצבעה (Binary RR)",
                                   0.1, 5.0, DEFAULT_EPS_VOTE,  step=0.1)
            eps_count = st.slider("ε — ספירות עיר (Laplace)",
                                   0.1, 5.0, DEFAULT_EPS_COUNT, step=0.1)
            st.divider()

        st.markdown("#### ⚙️ סימולציה")
        if st.button("⚡ בצע סימולציה לדוחות", use_container_width=True,
                     help="כל הפעילים הגישו את רשימותיהם"):
            bulk_simulate_unreported(eps_vote, eps_party)
            st.success("כל דוחות המצביעים נשלחו.")
        if st.button("🔄 אפס את כל הדוחות", use_container_width=True):
            st.session_state.reported_voted = {}
            st.session_state.reported_party = {}
            st.success("כל הדוחות נמחקו.")

    return st.session_state.current_page, eps_vote, eps_party, eps_count


# =============================================================================
# DATA SIMULATION
# =============================================================================

def simulate_voter_list(seed, n_voters, cities, party_names, organiser,
                         defection_rate):
    """
    בנה את רשימת המצביעים האמיתית.
    מחזיר DataFrame: voter_id, name, city, true_voted, true_party.
    """
    rng = np.random.default_rng(seed)
    first_names = [
        "יוסי", "מיכל", "דוד", "שרה", "אבי", "רחל", "נועה", "אמיר",
        "תמר", "גיל", "לי", "עומר", "דינה", "ניר", "מאיה", "יובל",
        "רון", "הלה", "שי", "אורית", "ידין", "ליאור", "כרמל", "בר",
    ]
    last_names = [
        "כהן", "לוי", "מזרחי", "פרץ", "ביטון",
        "אברהם", "דהן", "שמש", "פרידמן", "חסן",
    ]
    city_turnout = dict(zip(
        cities,
        np.clip(rng.beta(7, 3, size=len(cities)), 0.10, 0.97)
    ))
    k = len(party_names)
    defect_each = defection_rate / (k - 1)
    party_probs = [1 - defection_rate] + [defect_each] * (k - 1)
    rows = []
    for i in range(n_voters):
        city  = rng.choice(cities)
        fname = rng.choice(first_names)
        lname = rng.choice(last_names)
        voted = bool(rng.random() < city_turnout[city])
        party = rng.choice(party_names, p=party_probs)
        rows.append({
            "voter_id"  : i,
            "name"      : f"{fname} {lname}",
            "city"      : city,
            "true_voted": voted,
            "true_party": party,
        })
    return pd.DataFrame(rows)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if "voters" not in st.session_state:
        st.session_state.voters = simulate_voter_list(
            SEED, N_VOTERS, CITIES, PARTY_NAMES, ORGANISER, DEFECTION_RATE
        )
    if "reported_voted" not in st.session_state:
        st.session_state.reported_voted = {}
    if "reported_party" not in st.session_state:
        st.session_state.reported_party = {}


# =============================================================================
# DP REPORTING ACTIONS
# =============================================================================

def record_voter_report(voter_id, true_voted, true_party, eps_vote, eps_party):
    """החל רעש DP ושמור את הגרסה המוגנת — לא את האמת."""
    dp_voted = randomized_response(true_voted, eps_vote)
    dp_party = k_randomized_response(true_party, PARTY_NAMES, eps_party)
    st.session_state.reported_voted[voter_id] = dp_voted
    st.session_state.reported_party[voter_id] = dp_party


def bulk_simulate_unreported(eps_vote, eps_party):
    for _, row in st.session_state.voters.iterrows():
        vid = row["voter_id"]
        if vid not in st.session_state.reported_voted:
            record_voter_report(
                vid, row["true_voted"], row["true_party"], eps_vote, eps_party
            )


# =============================================================================
# AGGREGATE HELPERS
# =============================================================================

def compute_city_dp_counts(eps_count):
    df = st.session_state.voters
    rv = st.session_state.reported_voted
    rows = []
    for city in CITIES:
        city_ids       = df.loc[df.city == city, "voter_id"].tolist()
        n              = len(city_ids)
        reported_flags = [rv[vid] for vid in city_ids if vid in rv]
        raw_count      = sum(reported_flags)
        dp_count       = max(0, int(round(laplace_mechanism(raw_count, 1.0, eps_count))))
        true_count     = int(df.loc[df.city == city, "true_voted"].sum())
        rows.append({
            "עיר"              : city,
            "מגויסים"          : n,
            "הצביעו (DP)"      : dp_count,
            "הצביעו אמיתי"     : true_count,
            "דוחות שהתקבלו"    : len(reported_flags),
            "עדיין ממתינים"    : n - len(reported_flags),
        })
    return pd.DataFrame(rows)


def compute_party_dp_estimates(eps_party):
    rp = st.session_state.reported_party
    if not rp:
        return {p: 0 for p in PARTY_NAMES}
    reported_parties = list(rp.values())
    freqs = estimate_krr_frequency(reported_parties, PARTY_NAMES, eps_party)
    total = len(reported_parties)
    return {p: max(0, int(round(freqs[p] * total))) for p in PARTY_NAMES}


def compute_overall_turnout_estimate(eps_vote):
    rv = st.session_state.reported_voted
    if not rv:
        return 0.0, 0, 0
    flags     = list(rv.values())
    est_rate  = estimate_rr_frequency(flags, eps_vote)
    est_count = int(round(est_rate * len(flags)))
    return est_rate, est_count, len(flags)


# =============================================================================
# PLOT HELPERS  (all Hebrew strings wrapped with heb() for matplotlib)
# =============================================================================

def plot_city_bars(city_df):
    """תרשים עמודות אופקי אינטראקטיבי: הצביעו (DP) לעומת ממתינים לפי עיר."""
    cities  = city_df["עיר"].tolist()
    voted   = city_df["הצביעו (DP)"].tolist()
    pending = city_df["עדיין ממתינים"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cities, x=voted, orientation='h',
        name="הצביעו (משמר-פרטיות)", marker_color=STATUS_VOTED, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=cities, x=pending, orientation='h',
        name="מצביעים פוטנציאליים ממתינים", marker_color=STATUS_PENDING, opacity=0.60,
    ))
    fig.update_layout(
        barmode='stack',
        title=dict(text="נוכחות לפי עיר (משמר-פרטיות)", font=dict(size=13, color='#1a2340')),
        xaxis_title="מצביעים",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#1a2340')),
        height=320, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Heebo, sans-serif', color='#1a2340'),
        modebar=dict(bgcolor='rgba(240,244,248,0.9)', color='#1a2340', activecolor='#3b6cb7'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e0e4ec', zeroline=False,
                     tickfont=dict(color='#1a2340'), title_font=dict(color='#1a2340'),
                     linecolor='#c0ccde', tickcolor='#c0ccde')
    fig.update_yaxes(showgrid=False,
                     tickfont=dict(color='#1a2340'), linecolor='#c0ccde')
    return fig


def plot_missing_voters(city_df):
    """תרשים עמודות אופקי אינטראקטיבי: מצביעים שטרם גויסו לפי עיר."""
    sorted_df = city_df.sort_values("עדיין ממתינים", ascending=True)
    cities  = sorted_df["עיר"].tolist()
    missing = sorted_df["עדיין ממתינים"].tolist()
    total   = sorted_df["מגויסים"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cities, x=total, orientation='h',
        name='סה"כ מגויסים', marker_color='#d0ddf0', opacity=0.70,
    ))
    fig.add_trace(go.Bar(
        y=cities, x=missing, orientation='h',
        name="לא גויסו עדיין", marker_color=STATUS_MISSING, opacity=0.85,
    ))
    fig.update_layout(
        barmode='overlay',
        title=dict(text="מצביעים שטרם גויסו — לפי עיר", font=dict(size=13, color='#1a2340')),
        xaxis_title="מצביעים",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#1a2340')),
        height=320, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Heebo, sans-serif', color='#1a2340'),
        modebar=dict(bgcolor='rgba(240,244,248,0.9)', color='#1a2340', activecolor='#3b6cb7'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e0e4ec', zeroline=False,
                     tickfont=dict(color='#1a2340'), title_font=dict(color='#1a2340'),
                     linecolor='#c0ccde', tickcolor='#c0ccde')
    fig.update_yaxes(showgrid=False,
                     tickfont=dict(color='#1a2340'), linecolor='#c0ccde')
    return fig


def plot_party_estimates(party_counts):
    """תרשים עמודות אינטראקטיבי: הערכת הצבעות לפי מפלגה (k-RR)."""
    parties = list(party_counts.keys())
    counts  = [party_counts[p] for p in parties]
    colors  = [PARTY_COLORS.get(p, "#888") for p in parties]
    org_idx = parties.index(ORGANISER)

    fig = go.Figure(go.Bar(
        x=parties, y=counts,
        marker_color=colors,
        marker_line_color=['gold' if i == org_idx else 'black' for i in range(len(parties))],
        marker_line_width=[2.5 if i == org_idx else 0.8 for i in range(len(parties))],
        text=counts, textposition='outside',
    ))
    fig.update_layout(
        title=dict(
            text=f"הערכת DP — הצבעות לפי מפלגה (מסגרת זהב = {ORGANISER})",
            font=dict(size=13, color='#1a2340'),
        ),
        yaxis_title="מצביעים משוערים",
        height=320, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Heebo, sans-serif', color='#1a2340'),
        showlegend=False,
        modebar=dict(bgcolor='rgba(240,244,248,0.9)', color='#1a2340', activecolor='#3b6cb7'),
    )
    fig.update_yaxes(showgrid=True, gridcolor='#e0e4ec', zeroline=False,
                     tickfont=dict(color='#1a2340'), title_font=dict(color='#1a2340'),
                     linecolor='#c0ccde', tickcolor='#c0ccde')
    fig.update_xaxes(showgrid=False,
                     tickfont=dict(color='#1a2340'), linecolor='#c0ccde')
    return fig


# =============================================================================
# PAGE: MERGED DASHBOARD  (overview + cities + parties)
# =============================================================================

def page_dashboard(eps_vote, eps_party, eps_count):
    st.markdown('<div class="page-title">📊 לוח ראשי</div>',
                unsafe_allow_html=True)
    st.caption("סיכום הקמפיין — נוכחות, ערים, ומצביעים ממתינים.")

    n_total    = len(st.session_state.voters)
    n_reported = len(st.session_state.reported_voted)
    est_rate, est_count, _ = compute_overall_turnout_estimate(eps_vote)

    # ── Accuracy banner (shows impact of current ε selection) ──────────
    render_accuracy_banner(eps_vote, eps_count, n_reported)

    st.divider()

    # ── Top metric cards ────────────────────────────────────────────────
    n_pending = n_total - n_reported
    c1, c2, c3 = st.columns(3)
    c1.metric("מצביעים פוטנציאליים שהומרצו", f"{n_reported:,}",
              delta=f"{n_reported/n_total:.0%} מהרשימה")
    c2.metric("הצביעו (משמרת-פרטיות)",
              f"{est_count:,}" if n_reported > 0 else "—",
              delta=f"~{est_rate:.0%} מהמדווחים" if n_reported > 0 else None)
    c3.metric("מצביעים פוטנציאליים ממתינים",
              f"{n_pending:,}",
              delta=f"נותרו עוד {n_pending} מצביעים להמריץ" if n_pending > 0 else "כולם המרצו ✓",
              delta_color="inverse")

    if n_reported == 0:
        st.info("טרם הוגשו דוחות.  עבור לעמוד **רשימת מצביעים** "
                "לסימון מצביעים, או לחץ **בצע סימוציה לדוחות** בסרגל הצד.")
        return

    city_df      = compute_city_dp_counts(eps_count)
    party_counts = compute_party_dp_estimates(eps_party)

    # ── Row 1: city turnout | missing voters ────────────────────────────
    st.divider()
    st.markdown("#### נוכחות ומצביעים ממתינים לפי עיר")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_city_bars(city_df), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_missing_voters(city_df), use_container_width=True)

    st.divider()


# =============================================================================
# PAGE: VOTER LIST  (name on the right, buttons on the left)
# =============================================================================

def page_voter_list(eps_vote, eps_party):
    st.markdown('<div class="page-title">📋 רשימת מצביעים</div>',
                unsafe_allow_html=True)
    st.caption(
        "סמן מצביעים פוטנציאליים.  "
        "כל דיווח מוגן על ידי Randomized Response mechanism המספק פרטיות דיפרנציאלית.  "
    )

    df = st.session_state.voters
    rv = st.session_state.reported_voted

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        city_filter = st.selectbox("סנן לפי עיר", ["הכל"] + CITIES)
    with col_f2:
        status_filter = st.selectbox(
            "סנן לפי סטטוס דוח",
            ["הכל", "טרם דווח", "דווח — הצביע"],
        )
    with col_f3:
        search = st.text_input("חפש שם", "")

    filtered = df.copy()
    if city_filter != "הכל":
        filtered = filtered[filtered.city == city_filter]
    if search:
        filtered = filtered[filtered.name.str.contains(search, case=False)]
    if status_filter == "טרם דווח":
        filtered = filtered[~filtered.voter_id.isin(rv)]
    elif status_filter == "דווח — הצביע":
        filtered = filtered[filtered.voter_id.isin(rv)]

    st.markdown(f"**מוצגים {len(filtered)} מצביעים**")
    st.divider()

    for _, row in filtered.iterrows():
        vid     = row["voter_id"]
        already = vid in rv

        with st.container(border=True):
            # Layout (left → right): buttons | status | name
            # Hebrew reading order (right → left): name | status | buttons
            col_btns, col_status, col_name = st.columns([2, 2, 3])

            # Rightmost column: voter name and city
            with col_name:
                icon = "✅" if vid in rv else "⏳"
                st.markdown(f"**{icon}  {row['name']}**")
                st.caption(f"📍 {row['city']}")

            # Middle column: report status
            with col_status:
                if already:
                    st.markdown(
                        "<span style='color:green'>דווח: הצביע ✓</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption("(דוח מוגן DP נשמר)")
                else:
                    st.caption("טרם דווח")

            # Leftmost column: action buttons
            with col_btns:
                if not already:
                    if st.button("✅ הצביע", key=f"v_yes_{vid}",
                                 use_container_width=False):
                        record_voter_report(vid, True, row["true_party"],
                                            eps_vote, eps_party)
                        st.rerun()
                else:
                    if st.button("↩ בטל", key=f"v_undo_{vid}",
                                 width='content'):
                        del st.session_state.reported_voted[vid]
                        if vid in st.session_state.reported_party:
                            del st.session_state.reported_party[vid]
                        st.rerun()

    if len(filtered) == 0:
        st.info("אין מצביעים התואמים את הסינון הנוכחי.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_styles()
    init_session_state()
    render_top_bar(ACTIVIST_NAME)
    page, eps_vote, eps_party, eps_count = render_sidebar()

    if   page == "dashboard": page_dashboard(eps_vote, eps_party, eps_count)
    elif page == "voters":    page_voter_list(eps_vote, eps_party)


if __name__ == "__main__":
    main()
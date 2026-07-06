"""
app.py — לוח בחירות עם שמירת פרטיות  (Streamlit)
==================================================
הרצה:  streamlit run app.py
דרישות: voting_dp.py באותה תיקייה.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import population
import geo
from voting_dp import (
    randomized_response,
    estimate_rr_frequency,
    k_randomized_response,
    estimate_krr_frequency,
    rr_margin_of_error,
)

# The offline privacy–utility analysis (experiments/privacy_utility.py) is pure
# numpy/pandas (no matplotlib needed for the sweep itself); we reuse its Monte-
# Carlo sweep + aggregation so the UI's tradeoff tab matches the paper exactly.
_EXP_DIR = Path(__file__).resolve().parent / "experiments"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
import privacy_utility as pu  # noqa: E402


# =============================================================================
# UI TEXT  (Hebrew) — all constant, data-independent user-facing strings.
# -----------------------------------------------------------------------------
# Every title, button label, tooltip, section header and static message lives
# here, separated from the code, so the text is easy to edit and the app's
# language can be swapped by replacing this one block. Strings that interpolate
# runtime values (counts, ε, party names) stay inline at their call sites, since
# they depend on the data.
# =============================================================================

# ── Browser tab ─────────────────────────────────────────────────────────────
PAGE_TITLE = "ElectorApp — לוח DP"

# ── Sidebar navigation ──────────────────────────────────────────────────────
NAV_MENU_HEADER = "#### 🗳️ תפריטים"
NAV_DASHBOARD   = "📊  לוח ראשי"
NAV_VOTERS      = "📋  פנקס הבוחרים"
NAV_TRADEOFF    = "⚖️  ניתוח פרטיות-תועלת-עלות"

# ── Spinners / cache messages ───────────────────────────────────────────────
SPINNER_BUILD_POPULATION = "בונה אוכלוסיית בוחרים מנתוני אמת…"
SPINNER_RUN_SIM          = "מריץ סימולציה: מחיל הגנת פרטיות על הדוחות…"
SPINNER_RUN_TRADEOFF     = "מריץ סימולציית פרטיות–תועלת–עלות…"

# ── Accuracy banner ─────────────────────────────────────────────────────────
ACC_TITLE         = "הערכת דיוק הנתונים"   # expander header (shown with icon + level)
ACC_LEVEL_HIGH    = "דיוק גבוה (הגנת פרטיות חלשה)"
ACC_LEVEL_MEDIUM  = "דיוק בינוני"
ACC_LEVEL_LOW     = "דיוק נמוך (הגנת פרטיות חזקה)"
ACC_STATUS_HEADER = "📊 **סטטוס הצבעה** "
ACC_STATUS_TOOLTIP = (
    "Randomized Response (RR) — מנגנון פרטיות מקומית. "
    "כל דיווח מורעש לפני שמירתו: בהסתברות p הדיווח זהה לאמת, "
    "בהסתברות 1-p הוא מתהפך אקראית כדי להגן על פרטיות המשתמש."
)
ACC_CITY_HEADER = "🏙️ **ספירות עיר** "
ACC_CITY_TOOLTIP = (
    "הספירות מחושבות על ידי סכימת דיווחי ההצבעה השמורים (שכבר הורעשו בעת הקליטה)."
)
ACC_TIP_TEXT = (
    "💡 ε גבוה יותר = דיוק גבוה יותר, פרטיות נמוכה יותר.  "
    "ε נמוך יותר = פרטיות גבוהה יותר, שגיאה גדולה יותר."
)

# ── Table / DataFrame column headers (also shown in charts) ─────────────────
COL_CITY       = "עיר"
COL_RECRUITED  = "מגויסים"
COL_VOTED_DP   = "הצביעו (משמר-פרטיות)"
COL_VOTED_TRUE = "הצביעו אמיתי"
COL_REPORTS    = "דוחות שהתקבלו"
COL_PENDING    = "עדיין ממתינים"

# ── Charts ──────────────────────────────────────────────────────────────────
PLOT_CITY_VOTED_TRACE   = "הצביעו (משמר-פרטיות)"
PLOT_CITY_PENDING_TRACE = "מצביעים פוטנציאליים ממתינים"
PLOT_CITY_TITLE         = "נוכחות לפי עיר (משמר-פרטיות)"
PLOT_CITY_XAXIS         = "מצביעים"
PLOT_PARTY_YAXIS        = "מצביעים משוערים"
PLOT_TO_RECALL          = "תועלת: שיעור התומכים שטרם הצביעו אשר המפלגה הצליחה להמריץ להצביע, מתוך כלל המצביעים הפוטנציאלים (↑ טוב יותר)."
PLOT_TO_PRECISION       = "תועלת: שיעור פניות הפעילים שהצליחו להמריץ מצביעים פוטנציאלים, מתוך כלל הפניות שבוצעו (↑ טוב יותר)."
PLOT_TO_RISK            = "סיכון לבוחר (חשיפת סטאטוס ההצבעה): היתרון שיש לאדם חיצוני בניחוש סטאטוס ההצבעה מהסתכלות בדוח המצביעים שהודלף לעומת לניחוש כללי (↓ טוב יותר)"
PLOT_TO_XAXIS           = "תקציב פרטיות ε   (ε גבוה ⇐ פרטיות חלשה יותר)"
PLOT_TO_YAXIS           = "שיעור (0–1)"

# ── Dashboard page ──────────────────────────────────────────────────────────
DASH_PAGE_TITLE        = "📊 לוח ראשי"
DASH_SUBTITLE          = "סיכום הקמפיין — נוכחות, ערים, ומצביעים ממתינים."
DASH_SETTINGS_EXPANDER = "⚙️ הגדרות סימולציה"
DASH_EPS_SLIDER        = "ε — תקציב פרטיות (Randomized Response)"
DASH_FILL_SLIDER       = "שיעור מילוי הרשימה: אחוז המצביעים בפנקס הבוחרים שהסימולציה תמלא עבורם את סטאטוס ההצבעה"
DASH_FILL_HELP = (
    "הסימולציה מגרילה סטטוס הצבעה (משמר-פרטיות) רק לחלק זה מהרשימה "
    "(השאר נותרים ללא סטאטוס)."
)
DASH_RUN_BTN          = "⚡ הרץ סימולציה"
DASH_RUN_BTN_HELP     = "הצבת סטאטוס ההצבעה למצביעים פוטנציאליים מפנקס הבוחרים"
DASH_RESET_BTN        = "🔄 אפס את כל הדוחות"
DASH_RESET_DONE       = "כל הדוחות אופסו."
DASH_METRIC_MOBILIZED = "מצביעים פוטנציאליים שהומרצו"
DASH_METRIC_VOTED     = "הצביעו (משמרת-פרטיות)"
DASH_METRIC_PENDING   = "מצביעים פוטנציאליים ממתינים"
DASH_ALL_MOBILIZED    = "כל המצביעים הפוטנציאלים הומרצו ✓"
DASH_NO_REPORTS_INFO  = (
    "טרם הוגשו דוחות.  עבור לעמוד **רשימת מצביעים** לסימון מצביעים, "
    "או לחץ **הרץ סימולציה** למעלה."
)
DASH_CITY_CHART_HEADER = "#### נוכחות ומצביעים ממתינים לפי עיר"

# ── Map (geographic view under the per-city panel) ──────────────────────────
MAP_HEADER          = "#### מפת נוכחות ארצית (משמר-פרטיות)"
MAP_VIEW_LABEL      = "תצוגת מפה"
MAP_VIEW_AGG        = "מפת צבירה (הערכות DP)"
MAP_VIEW_DOTS       = "מפת רשומות מוגנות (נקודות)"
MAP_METRIC_LABEL    = "מדד לתצוגה"
MAP_METRIC_TURNOUT  = "אחוז הצבעה משוער"
MAP_METRIC_PENDING  = "מצביעים פוטנציאליים ממתינים"
MAP_METRIC_VOTED    = "הצביעו (משוער)"
MAP_DOT_VOTED       = "דווח: הצביע"
MAP_DOT_NOTVOTED    = "דווח: טרם הצביע"
MAP_AGG_TITLE       = ""
MAP_DOTS_TITLE      = (
    "כל נקודה היא רשומה משמרת-פרטיות (**מורעשת**) אחת."
)
MAP_MISSING_COORDS  = "⚠️ ל־{n} יישובים אין קואורדינטות והם אינם מוצגים על המפה (הם עדיין מופיעים בטבלה ובתרשים)."
MAP_NO_COORDS_FILE  = (
    "מפה אינה זמינה: קובץ הקואורדינטות `settlement_coords.csv` חסר. "
    "הרץ `python experiments/prepare_geo.py` כדי לייצר אותו."
)
MAP_DOTS_CAP        = 5000   # max individual dots rendered (random sample beyond this)
MAP_DOTS_CAPPED_MSG = "מוצגת דגימה אקראית של {shown:,} מתוך {total:,} רשומות מורעשות."

# ── Voter-list page ─────────────────────────────────────────────────────────
VOTERS_PAGE_TITLE = "📋 פנקס הבוחרים"
VOTERS_CAPTION = (
    "סמן מצביעים פוטנציאליים (השמות אינם אמיתיים).  "
    "כל דיווח מוגן על ידי מנגנון Randomized Response המספק פרטיות דיפרנציאלית (Differential Privacy).  "
)
VOTERS_FILTER_CITY           = "סנן לפי עיר"
VOTERS_FILTER_ALL            = "הכל"
VOTERS_FILTER_STATUS         = "סנן לפי סטטוס דוח"
VOTERS_STATUS_NOT_REPORTED   = "טרם דווח"
VOTERS_STATUS_REPORTED_VOTED = "דווח"
VOTERS_SEARCH                = "חפש שם (שם פרטי ושם משפחה)"
VOTERS_REPORTED_BADGE        = "<span style='color:green'>דווח: הצביע ✓</span>"
VOTERS_REPORTED_CAPTION      = "(דוח משמר-פרטיות נשמר)"
VOTERS_VOTE_BTN              = "✅ הצביע"
VOTERS_UNDO_BTN              = "↩ בטל"
VOTERS_NO_MATCH_INFO         = "אין מצביעים התואמים את הסינון הנוכחי."

# ── Privacy–utility–cost (tradeoff) page ────────────────────────────────────
TRADEOFF_PAGE_TITLE = "⚖️ פרטיות-תועלת-עלות"
TRADEOFF_INTRO = (
    "ניתוח השפעת רמת הפרטיות שנקבעה בסימולציה על הסיכון למצביע (פרמטר ε), התועלת הכללית למפלגה המתפעלת את המערכת והערכת העלות למפלגה."
)
TRADEOFF_PARAMS_EXPANDER     = "⚙️ פרמטרים של הסימולציה"
TRADEOFF_SEC_PARTY_POWER     = "**כוח המפלגה**"
TRADEOFF_FRAC_ACTIVISTS      = "אחוז התומכים שהם פעילים במפלגה (ימריצו את המצביעים הפוטנציאלים)"
TRADEOFF_FRAC_ACTIVISTS_HELP = "מספר הפעילים נגזר מגודל המפלגה: (אחוז × תומכים)."
TRADEOFF_CALLS               = "קיבולת הוצאת שיחות לכל פעיל"
TRADEOFF_CALLS_HELP          = "קובע את מספר השיחות הכולל שניתן להוציא למצביעים פוטנציאליים כדי להמריץ אותם (מספר המפעילים × קיבולת שיחות לכל פעיל)."
TRADEOFF_SEC_COST_RUNS       = "**עלות וכמות הרצות**"
TRADEOFF_COST                = "עלות לשיחה ($)"
TRADEOFF_COST_HELP           = "עלות הוצאת שיחה להמרצת מצביע פוטנציאלי."
TRADEOFF_REPEATS             = "מספר האיטרציות (דגימות חוזרות) בסימולציה"
TRADEOFF_REPEATS_HELP        = "מספר ההגרלות המבוצעות לכל הגדרת פרטיות ε."
TRADEOFF_SEC_POP_MODEL       = "**אוכלוסייה ומודל**"
TRADEOFF_FRACTION            = "אחוז דגימת אוכלוסייה"
TRADEOFF_FRACTION_HELP       = "גודל הדגימה מהאוכלוסיית הבוחרים (פנקס הבוחרים) לביצוע הסימולציה (דגימה קטנה יותר → רצועות שגיאה רחבות יותר)."
TRADEOFF_SUPPORT_NOISE       = "רעש מודל התמיכה"
TRADEOFF_SUPPORT_NOISE_HELP  = "מנבא תמיכה של מצביע פוטנציאלי במפלגה (הצביע למפלגה). 0 = כל מצביע פוטנציאלי יבציע למפלגה, ערך > 1 = מצביע עלול לא להצביע למפלגה."
TRADEOFF_SEC_CONSTRAINTS     = "##### אילוצי המפלגה"
TRADEOFF_MIN_RECALL          = "recall מינימלי"
TRADEOFF_NO_CONSTRAINT_HELP  = "0 = ללא אילוץ."
TRADEOFF_MIN_PRECISION       = "precision מינימלי"
TRADEOFF_MAX_LOSS            = "הפסד מקסימלי ($)"
TRADEOFF_MAX_LOSS_HELP       = "הפסד כספי מקסימלי משיחות מוטעות עקב דיווחי הצבעה משמרי פרטיות (ייתכן שמצביע כבר הצביע, אבל הומרץ בכל זאת, עקב דיווח הצבעה לא נכון). 0 = חישוב אוטומטי."
TRADEOFF_RUN_BTN             = "🚀 הרץ סימולציה"
TRADEOFF_CONFIGURE_INFO = (
    "הגדר פרמטרים ולחץ **הרץ סימולציה** כדי לחשב את הפשרה "
    "בין פרטיות, תועלת ועלות."
)
TRADEOFF_METRIC_PARTY        = "מפלגה"
TRADEOFF_METRIC_SUPPORTERS   = "מספר התומכים באוכלוסייה (מצביעים פוטנציאלים)"
TRADEOFF_METRIC_ACTIVISTS    = "פעילים × שיחות"
TRADEOFF_METRIC_BUDGET       = "תקציב כספי כולל נדרש"
TRADEOFF_NO_FEASIBLE_WARN = (
    "⚠️ אף ε אינו עומד בכל האילוצים. הקל את הדרישות "
    "(recall / precision / הפסד מקסימלי) או הגדל את התקציב."
)
TRADEOFF_EPS_SLIDER = "סמן ε על הגרף"
TRADEOFF_LEGEND_CAPTION = (
    "💡 לחיצה על פריט במקרא מסתירה / מציגה את העקומה המתאימה. "
    "ε גבוה = דיוק תפעולי גבוה אך פרטיות חלשה יותר לבוחר."
)
TRADEOFF_METRIC_RISK       = "סיכון לבוחר (חשיפת סטאטוס ההצבעה)"
TRADEOFF_METRIC_RISK_HELP =  "סיכון לבוחר (חשיפת סטאטוס ההצבעה): היתרון שיש לאדם חיצוני בניחוש סטאטוס ההצבעה מהסתכלות בדוח המצביעים שהודלף לעומת לניחוש כללי (↓ טוב יותר)"
TRADEOFF_METRIC_RECALL     = "Recall"
TRADEOFF_METRIC_RECALL_HELP = "שיעור התומכים שטרם הצביעו אשר המפלגה הצליחה להמריץ להצביע, מתוך כלל המצביעים הפוטנציאלים (↑ טוב יותר)."
TRADEOFF_METRIC_PRECISION  = "Precision"
TRADEOFF_METRIC_PRECISION_HELP = "שיעור פניות הפעילים שהצליחו להמריץ מצביעים פוטנציאלים, מתוך כלל הפניות שבוצעו (↑ טוב יותר)."
TRADEOFF_METRIC_COST       = "הפסד כספי משיחות מוטעות עקב דיווחי הצבעה משמרי פרטיות"


# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS
# =============================================================================

SEED = 42

# ── Population source (real CEC aggregates → synthetic individuals) ──────────
# The settlement list, cities, and party names are DERIVED from the real data
# (population.py) at session init — not hard-coded. Tune these knobs to trade
# fidelity against a responsive UI.
APP_SETTLEMENTS        = None   # None → population.DEFAULT_SETTLEMENTS (6 big + 6 small)
APP_FRACTION           = 1.0    # proportional down-sample (1.0 = full real size)
APP_MAX_PER_SETTLEMENT = 200    # cap per settlement so big cities stay responsive
APP_KEEP_TOP_PARTIES   = 8      # collapse minor parties into "אחר" (smaller k-RR)
APP_ORGANISER          = "הליכוד"  # party running the Elector (gold highlight).
                                   # None → largest party by support in the data.
VOTER_LIST_DISPLAY_CAP = 200    # max voter cards rendered at once (UI guard)

# ── ONE privacy budget ε (binary Randomized Response). No Laplace anywhere ──
# The same ε protects each capture AND every aggregate derived from it (counts,
# turnout) by DP post-processing. The optional party signal (k-RR) reuses this
# same ε — there is no longer a second, separate privacy knob in the UI.
DEFAULT_EPS = 1.0

# ── Partial fill of the voter list ──────────────────────────────────────────
# Realistic scenario: the Elector is only *partly* populated — activists have
# captured a voting status for only a fraction of the list; the rest are left
# with no status at all. The bulk simulation randomizes the status of only this
# fraction of the (still-unreported) voters. 1.0 = mark everyone (old behaviour).
APP_SIM_FILL_FRACTION = 0.5

# ── Privacy–utility–cost analysis (tradeoff tab) defaults ───────────────────
ANALYSIS_FRACTION       = 0.3    # population down-sample for the offline sweep
ANALYSIS_FRAC_ACTIVISTS = 0.05   # share of the party's supporters fielded as activists
ANALYSIS_CALLS          = 200    # calls each activist can make
ANALYSIS_COST_PER_CALL  = 0.5    # $ per activist contact (prices misdirected calls)
ANALYSIS_REPEATS        = 20     # Monte-Carlo repetitions per ε
ANALYSIS_MIN_RECALL     = 0.70   # party's minimum acceptable GOTV recall
ANALYSIS_MIN_PRECISION  = 0.85   # party's minimum acceptable GOTV precision
ANALYSIS_MAX_LOSS       = 1000.0 # max tolerated $ loss from misdirected calls (0 = auto)

# Colours are assigned to whatever parties the data yields (see assign_party_colors).
PARTY_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#5975A4", "#B07AA1",
]
STATUS_VOTED   = "#27ae60"
STATUS_PENDING = "#e74c3c"
STATUS_MISSING = "#e67e22"

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

SHOW_PRIVACY_SETTINGS = True
ACTIVIST_NAME         = f" משתמש: א. כהן  |  מטה תל-אביב  {APP_ORGANISER}"

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

    /* ── Captions and labels: black (not grey), slightly larger ──────── */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    [data-testid="stCaptionContainer"] small,
    small, caption {
        color: #1a2340 !important;
        direction: rtl !important;
        font-size: 0.95rem !important;
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

    /* ── Native help "?" icon — keep it adjacent to its label (RTL) ──────
       Streamlit shows a hover "?" icon whenever a widget/metric has help=.
       By default a full-width label paragraph (forced to width:100%, e.g.
       inside expanders) pushes the icon to the far edge, away from its text.
       We let the label text shrink to its content and add a small gap, so the
       icon sits just beside the label. Placed AFTER the expander rule above so
       its width:100% on label paragraphs is overridden here.
       Applies to every labelled widget and metric in the app.            */
    [data-testid="stWidgetLabel"],
    [data-testid="stMetricLabel"] {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        align-items: center !important;
        gap: 8px !important;          /* the gap between the "?" and the label */
    }
    [data-testid="stWidgetLabel"] > div:first-child,
    [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
        flex: 0 1 auto !important;    /* shrink to content so the icon stays next to it */
        width: auto !important;
    }
    /* The "?" hover target / icon itself never shrinks. */
    [data-testid="stTooltipHoverTarget"],
    [data-testid="stTooltipIcon"] {
        flex: 0 0 auto !important;
    }

    /* ── Centered loading spinner (used during simulations) ──────────── */
    .dp-spin-wrap {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        width: 100%; padding: 44px 0;
    }
    .dp-spinner {
        width: 48px; height: 48px;
        border: 5px solid #dde8f8;
        border-top-color: #3b6cb7;
        border-radius: 50%;
        animation: dp-spin 0.85s linear infinite;
    }
    .dp-spin-text {
        margin-top: 14px; color: #2d4a8a;
        font-weight: 600; font-size: 15px; text-align: center;
    }
    @keyframes dp-spin { to { transform: rotate(360deg); } }
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
                <div class="top-bar-subtitle"></div>
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


def centered_spinner_html(text):
    """A CSS-only spinner centred in its container (see .dp-spinner styles)."""
    return (f'<div class="dp-spin-wrap"><div class="dp-spinner"></div>'
            f'<div class="dp-spin-text">{text}</div></div>')


# =============================================================================
# DP ACCURACY BANNER
# =============================================================================

def render_accuracy_banner(eps_vote, n_reported):
    p_rr       = np.exp(eps_vote) / (1.0 + np.exp(eps_vote))
    flip       = 1.0 - p_rr

    n_total = len(st.session_state.voters)
    n_cities = max(1, len(st.session_state.cities))
    expected_city_size = max(1, int(round(n_total / n_cities)))
    # City / turnout counts are de-biased sums of the stored RR reports — the
    # same ε as capture, by post-processing, with NO Laplace added. Their error
    # is the RR sampling error; show the worst-case 95% margin for a typical city.
    count_margin   = rr_margin_of_error(expected_city_size, eps_vote)
    count_accuracy = max(0.0, 1.0 - count_margin / expected_city_size)

    if float(eps_vote) >= 2.0:
        level_icon, level_txt = "🟢", ACC_LEVEL_HIGH
    elif float(eps_vote) >= 1.0:
        level_icon, level_txt = "🟡", ACC_LEVEL_MEDIUM
    else:
        level_icon, level_txt = "🔴", ACC_LEVEL_LOW

    with st.expander(f"{level_icon}  {ACC_TITLE} — {level_txt}", expanded=False):
        # ── 1) Voting-status accuracy (the RR mechanism itself) ──────────
        with st.container(border=True):
            st.markdown(
                ACC_STATUS_HEADER
                + "<span class='info-tooltip'><span class='info-icon'>i</span>"
                  "<span class='tooltip-text'>"
                + ACC_STATUS_TOOLTIP
                + "</span></span>",
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

        # ── 2) City-count accuracy (post-processing of the same ε) ───────
        with st.container(border=True):
            st.markdown(
                ACC_CITY_HEADER
                + "<span class='info-tooltip'><span class='info-icon'>i</span>"
                  "<span class='tooltip-text'>"
                + ACC_CITY_TOOLTIP
                + "</span></span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**תקציב פרטיות: ε = {eps_vote}**  (אותו ε — עיבוד-המשך)")
            st.divider()
            example = expected_city_size
            lo_ex   = max(0, round(example - count_margin))
            hi_ex   = round(example + count_margin)
            st.markdown(
                f"- 📏 **שגיאת דגימה:** ±{count_margin:.0f} מצביעים "
                f"(רווח סמך 95%, עיר בגודל ~{example})\n"
                f"- 📊 **דוגמה:** עבור ספירה מוערכת של {example} מצביעים — "
                f"הערך האמיתי נע בין {lo_ex} ל-{hi_ex} בסבירות 95%\n"
                f"- **דיוק כולל: {count_accuracy:.0%}**"
            )
            _static_bar(count_accuracy, f"דיוק: {count_accuracy:.0%}")

        st.markdown(
            "<div style='direction:rtl;text-align:right;margin-top:8px;margin-bottom:8px;padding:8px;"
            "background:#fffbe6;border-radius:8px;border:1px solid #ffe58f;'>"
            "<strong style='font-size:16px;'>"
            + ACC_TIP_TEXT
            + "</strong></div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """
    Navigation only. The privacy budget ε and the simulation controls live in
    the page body (see `page_dashboard`), mirroring the tradeoff page layout.
    Returns the current page key.
    """
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    nav_items = [
        ("dashboard", NAV_DASHBOARD),
        ("voters",    NAV_VOTERS),
        ("tradeoff",  NAV_TRADEOFF),
    ]

    with st.sidebar:
        st.markdown(NAV_MENU_HEADER, unsafe_allow_html=True)
        for page_key, label in nav_items:
            is_active = st.session_state.current_page == page_key
            open_tag  = '<div class="active-nav">' if is_active else "<div>"
            st.markdown(open_tag, unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.current_page


# =============================================================================
# DATA SOURCE  (real CEC aggregates → synthetic individual voters)
# =============================================================================

@st.cache_data(show_spinner=SPINNER_BUILD_POPULATION)
def build_population(settlements, fraction, max_per_settlement,
                     keep_top_parties, seed):
    """
    Synthetic individual-level voter population grounded in the real CEC
    results (see population.py). Cached so it is built once per parameter set,
    not on every Streamlit rerun. Returns the schema population.SCHEMA, with a
    `latent_support` column (modelled preference) used as the party signal.
    """
    return population.synthesize_population(
        settlements=settlements,
        fraction=fraction,
        max_per_settlement=max_per_settlement,
        keep_top_parties=keep_top_parties,
        seed=seed,
    )


def assign_party_colors(parties):
    """Map each party label to a stable colour from PARTY_PALETTE."""
    return {p: PARTY_PALETTE[i % len(PARTY_PALETTE)] for i, p in enumerate(parties)}


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if "voters" not in st.session_state:
        pop = build_population(
            APP_SETTLEMENTS, APP_FRACTION, APP_MAX_PER_SETTLEMENT,
            APP_KEEP_TOP_PARTIES, SEED,
        )
        st.session_state.voters       = pop
        # Cities and parties are DERIVED from the data, not hard-coded.
        st.session_state.cities       = population.settlement_names(pop)
        st.session_state.party_names  = population.party_candidates(pop)
        # Pin the organising party if configured and present; else largest by support.
        party_names = st.session_state.party_names
        st.session_state.organiser    = (
            APP_ORGANISER if APP_ORGANISER in party_names else party_names[0]
        )
        st.session_state.party_colors = assign_party_colors(party_names)
    if "reported_voted" not in st.session_state:
        st.session_state.reported_voted = {}
    if "reported_party" not in st.session_state:
        st.session_state.reported_party = {}
    # Single privacy budget ε, shared across pages. Pre-seeding the key lets the
    # in-body slider (key="eps") use it as its initial value.
    st.session_state.setdefault("eps", DEFAULT_EPS)
    # Fraction of the list the bulk simulation populates with a status (the rest
    # are left without one). Pre-seeded so the in-body slider (key="fill_fraction")
    # uses it as its initial value.
    st.session_state.setdefault("fill_fraction", APP_SIM_FILL_FRACTION)


# =============================================================================
# DP REPORTING ACTIONS
# =============================================================================

def record_voter_report(voter_id, true_voted, party, eps):
    """
    החל רעש DP ושמור את הגרסה המוגנת — לא את האמת.

    A single ε protects both the participation status (binary RR) and the
    optional party signal (k-RR). `party` is `latent_support` (defined for every
    individual and always a real candidate) — never `true_party`, which is <NA>
    for non-voters and "פסול" for invalid ballots.
    """
    dp_voted = randomized_response(true_voted, eps)
    dp_party = k_randomized_response(party, st.session_state.party_names, eps)
    st.session_state.reported_voted[voter_id] = dp_voted
    st.session_state.reported_party[voter_id] = dp_party


def bulk_simulate_unreported(eps, fill_fraction=1.0):
    """
    Simulate a *partial* fill of the Elector: only a random `fill_fraction` of the
    not-yet-reported voters get a DP-protected voting status; the rest are left
    with no status at all (the realistic case where the system is only partly
    populated). The DP mechanism is still applied at the point of capture — only
    the noised value is stored, never the truth.
    """
    df          = st.session_state.voters
    pending_ids = [vid for vid in df["voter_id"].tolist()
                   if vid not in st.session_state.reported_voted]
    n_fill = int(round(max(0.0, min(1.0, fill_fraction)) * len(pending_ids)))
    if n_fill >= len(pending_ids):
        fill_ids = set(pending_ids)
    else:
        rng = np.random.default_rng(SEED)
        fill_ids = set(rng.choice(pending_ids, size=n_fill, replace=False).tolist())

    for _, row in df.iterrows():
        vid = row["voter_id"]
        if vid in fill_ids:
            record_voter_report(
                vid, row["true_voted"], row["latent_support"], eps
            )


# =============================================================================
# AGGREGATE HELPERS
# =============================================================================

def compute_city_dp_counts(eps_vote):
    df = st.session_state.voters
    rv = st.session_state.reported_voted
    rows = []
    for city in st.session_state.cities:
        city_ids       = df.loc[df.city == city, "voter_id"].tolist()
        n              = len(city_ids)
        reported_flags = [rv[vid] for vid in city_ids if vid in rv]
        # De-bias the sum of the stored RR reports to recover the true "voted"
        # count. Summing/de-biasing the already-perturbed reports is
        # post-processing of the same capture-time ε — no Laplace noise added.
        if reported_flags:
            est_rate  = estimate_rr_frequency(reported_flags, eps_vote)
            dp_count  = max(0, int(round(est_rate * len(reported_flags))))
        else:
            dp_count  = 0
        true_count     = int(df.loc[df.city == city, "true_voted"].sum())
        rows.append({
            COL_CITY       : city,
            COL_RECRUITED  : n,
            COL_VOTED_DP   : dp_count,
            COL_VOTED_TRUE : true_count,
            COL_REPORTS    : len(reported_flags),
            COL_PENDING    : n - len(reported_flags),
        })
    return pd.DataFrame(rows)


def compute_party_dp_estimates(eps):
    rp = st.session_state.reported_party
    party_names = st.session_state.party_names
    if not rp:
        return {p: 0 for p in party_names}
    reported_parties = list(rp.values())
    freqs = estimate_krr_frequency(reported_parties, party_names, eps)
    total = len(reported_parties)
    return {p: max(0, int(round(freqs[p] * total))) for p in party_names}


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
    cities  = city_df[COL_CITY].tolist()
    voted   = city_df[COL_VOTED_DP].tolist()
    pending = city_df[COL_PENDING].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cities, x=voted, orientation='h',
        name=PLOT_CITY_VOTED_TRACE, marker_color=STATUS_VOTED, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=cities, x=pending, orientation='h',
        name=PLOT_CITY_PENDING_TRACE, marker_color=STATUS_PENDING, opacity=0.60,
    ))
    fig.update_layout(
        barmode='stack',
        title=dict(text=PLOT_CITY_TITLE, font=dict(size=13, color='#1a2340')),
        xaxis_title=PLOT_CITY_XAXIS,
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


def _map_layout(fig):
    """Shared MapLibre layout: Israel-framed, token-free tiles, light theme."""
    fig.update_layout(
        map=dict(style="carto-positron",
                 center=dict(lat=geo.ISRAEL_CENTER[0], lon=geo.ISRAEL_CENTER[1]),
                 zoom=geo.ISRAEL_ZOOM),
        height=460, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Heebo, sans-serif', color='#1a2340'),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,0.7)', font=dict(color='#1a2340')),
        modebar=dict(bgcolor='rgba(240,244,248,0.9)', color='#1a2340', activecolor='#3b6cb7'),
    )
    return fig


def plot_settlement_bubble_map(city_df, metric):
    """
    One bubble per settlement at its centroid; colour = a de-biased DP aggregate,
    size ∝ settlement size. Reads only `city_df` (already de-biased by
    compute_city_dp_counts) — post-processing of the stored RR reports, no extra
    privacy budget. Returns (figure, n_settlements_without_coords).
    """
    latlon = geo.city_latlon()

    lats, lons, sizes, values, texts, hover = [], [], [], [], [], []
    n_missing = 0
    for _, row in city_df.iterrows():
        city = row[COL_CITY]
        if city not in latlon:
            n_missing += 1
            continue
        reports = row[COL_REPORTS]
        if metric == MAP_METRIC_TURNOUT:
            val = (row[COL_VOTED_DP] / reports) if reports else 0.0
            hov = f"{val:.0%}"
        elif metric == MAP_METRIC_PENDING:
            val = row[COL_PENDING]
            hov = f"{int(val):,}"
        else:  # MAP_METRIC_VOTED
            val = row[COL_VOTED_DP]
            hov = f"{int(val):,}"
        lat, lon = latlon[city]
        lats.append(lat); lons.append(lon)
        sizes.append(row[COL_RECRUITED])
        values.append(val)
        texts.append(city)
        hover.append(hov)

    colorscale = "YlOrRd" if metric == MAP_METRIC_PENDING else "YlGn"
    n_max = max(sizes) if sizes else 1
    marker_sizes = [8 + 34 * (s / n_max) ** 0.5 for s in sizes]

    fig = go.Figure(go.Scattermap(
        lat=lats, lon=lons, mode='markers',
        marker=dict(
            size=marker_sizes,
            color=values, colorscale=colorscale, showscale=True,
            colorbar=dict(title=dict(text=metric, font=dict(color='#1a2340')),
                          tickfont=dict(color='#1a2340')),
            opacity=0.85,
        ),
        text=texts, customdata=hover,
        hovertemplate="<b>%{text}</b><br>" + metric + ": %{customdata}<extra></extra>",
    ))
    return _map_layout(fig), n_missing


def plot_voter_dot_map():
    """
    One dot per REPORTED voter, jittered around its settlement centroid, coloured
    by the STORED (RR-perturbed) status — never `true_voted`. This is exactly the
    noised record a leak would expose: individually deniable, yet accurate in
    aggregate. Returns (figure_or_None, note_or_None).
    """
    rv = st.session_state.reported_voted
    df = st.session_state.voters
    latlon = geo.city_latlon()

    rep = df.loc[df.voter_id.isin(rv), ["voter_id", "city"]].copy()
    rep = rep[rep.city.isin(latlon)]
    if rep.empty:
        return None, None

    note = None
    total = len(rep)
    if total > MAP_DOTS_CAP:
        rep = rep.sample(MAP_DOTS_CAP, random_state=SEED)
        note = MAP_DOTS_CAPPED_MSG.format(shown=MAP_DOTS_CAP, total=total)
    rep["voted"] = rep.voter_id.map(rv)

    rng = np.random.default_rng(SEED)
    lat_v, lon_v, lat_n, lon_n = [], [], [], []
    for city, grp in rep.groupby("city", sort=False):
        lat, lon = latlon[city]
        la, lo = geo.jitter(lat, lon, len(grp), rng)
        flags = grp["voted"].to_numpy()
        lat_v.extend(la[flags]);  lon_v.extend(lo[flags])
        lat_n.extend(la[~flags]); lon_n.extend(lo[~flags])

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=lat_n, lon=lon_n, mode='markers', name=MAP_DOT_NOTVOTED,
        marker=dict(size=6, color=STATUS_PENDING, opacity=0.55),
        hovertemplate=MAP_DOT_NOTVOTED + "<extra></extra>",
    ))
    fig.add_trace(go.Scattermap(
        lat=lat_v, lon=lon_v, mode='markers', name=MAP_DOT_VOTED,
        marker=dict(size=6, color=STATUS_VOTED, opacity=0.75),
        hovertemplate=MAP_DOT_VOTED + "<extra></extra>",
    ))
    return _map_layout(fig), note


def render_city_map(city_df):
    """Geographic view under the per-city panel: aggregate bubbles or DP dots."""
    if not geo.coords_available():
        st.info(MAP_NO_COORDS_FILE)
        return

    st.markdown(MAP_HEADER)
    view = st.radio(MAP_VIEW_LABEL, [MAP_VIEW_AGG, MAP_VIEW_DOTS],
                    horizontal=True, key="map_view")

    if view == MAP_VIEW_AGG:
        metric = st.selectbox(
            MAP_METRIC_LABEL,
            [MAP_METRIC_TURNOUT, MAP_METRIC_PENDING, MAP_METRIC_VOTED],
            key="map_metric",
        )
        fig, n_missing = plot_settlement_bubble_map(city_df, metric)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(MAP_AGG_TITLE)
        if n_missing:
            st.caption(MAP_MISSING_COORDS.format(n=n_missing))
    else:
        fig, note = plot_voter_dot_map()
        if fig is None:
            st.info(DASH_NO_REPORTS_INFO)
            return
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(MAP_DOTS_TITLE)
        if note:
            st.caption(note)


def plot_party_estimates(party_counts):
    """תרשים עמודות אינטראקטיבי: הערכת הצבעות לפי מפלגה (k-RR)."""
    organiser = st.session_state.organiser
    party_colors = st.session_state.party_colors
    parties = list(party_counts.keys())
    counts  = [party_counts[p] for p in parties]
    colors  = [party_colors.get(p, "#888") for p in parties]
    org_idx = parties.index(organiser)

    fig = go.Figure(go.Bar(
        x=parties, y=counts,
        marker_color=colors,
        marker_line_color=['gold' if i == org_idx else 'black' for i in range(len(parties))],
        marker_line_width=[2.5 if i == org_idx else 0.8 for i in range(len(parties))],
        text=counts, textposition='outside',
    ))
    fig.update_layout(
        title=dict(
            text=f"הערכת DP — הצבעות לפי מפלגה (מסגרת זהב = {organiser})",
            font=dict(size=13, color='#1a2340'),
        ),
        yaxis_title=PLOT_PARTY_YAXIS,
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

def page_dashboard():
    st.markdown(f'<div class="page-title">{DASH_PAGE_TITLE}</div>',
                unsafe_allow_html=True)
    st.markdown(DASH_SUBTITLE)
    st.info(DASH_NO_REPORTS_INFO)

    # ── Controls (ε slider + simulation actions) — in the page body, ────
    #    mirroring the privacy-utility-cost page layout. ──────────────────
    with st.expander(DASH_SETTINGS_EXPANDER, expanded=True):
        col_eps, col_run = st.columns([2, 1])
        with col_eps:
            if SHOW_PRIVACY_SETTINGS:
                st.slider(DASH_EPS_SLIDER, 0.1, 5.0, step=0.1, key="eps")
            st.slider(
                DASH_FILL_SLIDER,
                0.0, 1.0, step=0.05, key="fill_fraction",
                help=DASH_FILL_HELP,
                format="%.2f",
            )
        with col_run:
            run_sim = st.button(DASH_RUN_BTN, use_container_width=True,
                                type="primary", help=DASH_RUN_BTN_HELP)
            reset_sim = st.button(DASH_RESET_BTN, use_container_width=True)

    eps = st.session_state.eps


    if run_sim:
        spinner = st.empty()
        spinner.markdown(
            centered_spinner_html(SPINNER_RUN_SIM),
            unsafe_allow_html=True,
        )
        bulk_simulate_unreported(eps, st.session_state.fill_fraction)
        spinner.empty()
        n_now = len(st.session_state.reported_voted)
        st.success(
            f"הסימולציה הסתיימה — נקלטו דוחות עבור {n_now:,} מצביעים "
            f"({st.session_state.fill_fraction:.0%} מהרשימה); השאר נותרו ללא סטטוס."
        )
    if reset_sim:
        st.session_state.reported_voted = {}
        st.session_state.reported_party = {}
        st.success(DASH_RESET_DONE)

    n_total    = len(st.session_state.voters)
    n_reported = len(st.session_state.reported_voted)
    est_rate, est_count, _ = compute_overall_turnout_estimate(eps)

    # ── Accuracy banner (shows impact of current ε selection) ──────────
    render_accuracy_banner(eps, n_reported)

    st.divider()

    # ── Top metric cards ────────────────────────────────────────────────
    n_pending = n_total - n_reported
    c1, c2, c3 = st.columns(3)
    c1.metric(DASH_METRIC_MOBILIZED, f"{n_reported:,}",
              delta=f"{n_reported/n_total:.0%} מהרשימה")
    c2.metric(DASH_METRIC_VOTED,
              f"{est_count:,}" if n_reported > 0 else "—",
              delta=f"~{est_rate:.0%} מהמדווחים" if n_reported > 0 else None)
    c3.metric(DASH_METRIC_PENDING,
              f"{n_pending:,}",
              delta=f"נותרו עוד {n_pending} מצביעים להמריץ" if n_pending > 0 else DASH_ALL_MOBILIZED,
              delta_color="inverse")

    if n_reported == 0:
        return

    city_df = compute_city_dp_counts(eps)

    # ── Per-city panel: bar chart and map side-by-side ──────────────────
    #    Left: chased (voted, DP) vs. still-pending per city.
    #    Right: geographic view (aggregate bubbles, default, or DP-record dots).
    st.divider()
    col_bars, col_map = st.columns(2, gap="large")
    with col_bars:
        st.markdown(DASH_CITY_CHART_HEADER)
        st.plotly_chart(plot_city_bars(city_df), use_container_width=True)
    with col_map:
        render_city_map(city_df)

    st.divider()


# =============================================================================
# PAGE: VOTER LIST  (name on the right, buttons on the left)
# =============================================================================

def page_voter_list():
    eps = st.session_state.eps
    st.markdown(f'<div class="page-title">{VOTERS_PAGE_TITLE}</div>',
                unsafe_allow_html=True)
    st.info(VOTERS_CAPTION)

    df = st.session_state.voters
    rv = st.session_state.reported_voted

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        city_filter = st.selectbox(VOTERS_FILTER_CITY,
                                    [VOTERS_FILTER_ALL] + st.session_state.cities)
    with col_f2:
        status_filter = st.selectbox(
            VOTERS_FILTER_STATUS,
            [VOTERS_FILTER_ALL, VOTERS_STATUS_NOT_REPORTED, VOTERS_STATUS_REPORTED_VOTED],
        )
    with col_f3:
        search = st.text_input(VOTERS_SEARCH, "")

    filtered = df.copy()
    if city_filter != VOTERS_FILTER_ALL:
        filtered = filtered[filtered.city == city_filter]
    if search:
        filtered = filtered[filtered.name.str.contains(search, case=False)]
    if status_filter == VOTERS_STATUS_NOT_REPORTED:
        filtered = filtered[~filtered.voter_id.isin(rv)]
    elif status_filter == VOTERS_STATUS_REPORTED_VOTED:
        filtered = filtered[filtered.voter_id.isin(rv)]

    # The synthetic population can be large; cap how many cards we render so the
    # page stays responsive. Filters above narrow the list to find specific voters.
    total_match = len(filtered)
    shown = filtered.head(VOTER_LIST_DISPLAY_CAP)
    if total_match > len(shown):
        st.markdown(
            f"**מוצגים {len(shown)} מתוך {total_match} מצביעים** — "
            f"השתמש בסינון כדי לצמצם את הרשימה."
        )
    else:
        st.markdown(f"**מוצגים {total_match} מצביעים**")
    st.divider()

    for _, row in shown.iterrows():
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
                    st.markdown(VOTERS_REPORTED_BADGE, unsafe_allow_html=True)
                    st.caption(VOTERS_REPORTED_CAPTION)
                else:
                    st.caption(VOTERS_STATUS_NOT_REPORTED)

            # Leftmost column: action buttons
            with col_btns:
                if not already:
                    if st.button(VOTERS_VOTE_BTN, key=f"v_yes_{vid}",
                                 use_container_width=False):
                        record_voter_report(vid, True, row["latent_support"], eps)
                        st.rerun()
                else:
                    if st.button(VOTERS_UNDO_BTN, key=f"v_undo_{vid}",
                                 width='content'):
                        del st.session_state.reported_voted[vid]
                        if vid in st.session_state.reported_party:
                            del st.session_state.reported_party[vid]
                        st.rerun()

    if total_match == 0:
        st.info(VOTERS_NO_MATCH_INFO)


# =============================================================================
# PRIVACY–UTILITY–COST ANALYSIS  (offline sweep reused from experiments/)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_analysis_population(fraction, keep_top_parties, support_noise, seed):
    """
    Population for the offline sweep — same default settlements as the dashboard
    but WITHOUT the per-settlement cap, so party sizes stay realistic. A noisy
    support model (>0) makes the party's predictor imperfect (research Q A3).
    """
    support_model = ("observed" if not support_noise or support_noise <= 0
                     else population.make_noisy_support_model(support_noise))
    return population.synthesize_population(
        settlements=None,
        fraction=fraction,
        keep_top_parties=keep_top_parties,
        support_model=support_model,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def run_analysis(fraction, keep_top_parties, support_noise, organiser, repeats,
                 frac_activists, calls_per_activist, cost_per_call, seed):
    """
    Run the full ε sweep and return (agg, organiser, n_pop, n_supporters,
    n_activists, effective_budget). Cached on every argument, so re-tweaking the
    ε highlight slider is instant — only changing a parameter re-runs the sweep.
    """
    pop = build_analysis_population(fraction, keep_top_parties, support_noise, seed)
    candidates = population.party_candidates(pop)
    org = organiser if organiser in candidates else candidates[0]

    n_supporters = int((pop["latent_support"] == org).sum())
    n_activists = max(1, round(frac_activists * n_supporters))
    effective_budget = n_activists * calls_per_activist

    raw = pu.run_sweep(pop, org, candidates, pu.DEFAULT_EPSILONS,
                       repeats=repeats, budget=effective_budget, seed=seed + 1)
    agg = pu.aggregate(raw)
    # Price the privacy-attributable misdirected calls (same column the script writes).
    agg["cost_misdirected_calls"] = agg["wasted_due_to_privacy"] * cost_per_call
    return agg, org, len(pop), n_supporters, n_activists, effective_budget


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_tradeoff(agg, organiser, cost_per_call, min_recall, min_precision,
                  max_loss, optimal_eps, feasible_left, selected_eps):
    """
    Interactive Plotly reproduction of the paper's `_combined_rrerr` figure.

    LEFT axis (rate 0–1):  recall ↑, precision ↑, voter-risk ↓ — each with a 95%
    RR-sampling band (±1.96σ). RIGHT axis (USD): cost of misdirected calls ↓.
    Constraint lines (min recall / min precision / max loss), the party-feasible
    ε region (green band) and the privacy-optimal feasible ε (★ + dashed line)
    are drawn on top. Legend entries toggle each curve (with its band).
    """
    finite = agg[np.isfinite(agg["epsilon"])].sort_values("epsilon")
    x = finite["epsilon"].to_numpy(dtype=float)
    span = float(x.max() - x.min()) or 1.0
    x_lo, x_hi = x.min() - 0.04 * span, x.max() + 0.04 * span

    C_RISK, C_REC, C_PREC, C_COST, C_FEAS = (
        "#C44E52", "#3b6cb7", "#55A868", "#DD8452", "#2E8B57")

    fig = go.Figure()

    def band(y, e, color, group, axis="y"):
        fig.add_trace(go.Scatter(x=x, y=y + e, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False,
                                 legendgroup=group, yaxis=axis))
        fig.add_trace(go.Scatter(x=x, y=y - e, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_hex_to_rgba(color, 0.15),
                                 hoverinfo="skip", showlegend=False,
                                 legendgroup=group, yaxis=axis))

    def curve(y, e, color, name, group, symbol, dash=None, axis="y", fmt=".3f"):
        band(y, e, color, group, axis)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=name, legendgroup=group,
            line=dict(color=color, dash=dash, width=2),
            marker=dict(color=color, symbol=symbol, size=8), yaxis=axis,
            hovertemplate="ε=%{x:g}<br>" + name + ": %{y:" + fmt + "}<extra></extra>"))

    rec = finite["recall_mean"].to_numpy()
    prec = finite["precision_mean"].to_numpy()
    risk = finite["status_disclosure_advantage"].to_numpy()
    cost = finite["cost_misdirected_calls"].to_numpy()
    rec_e = 1.96 * finite["recall_std"].to_numpy()
    prec_e = 1.96 * finite["precision_std"].to_numpy()
    risk_e = 1.96 * finite["status_attacker_std"].to_numpy()
    cost_e = 1.96 * finite["wasted_std"].to_numpy() * cost_per_call

    curve(rec, rec_e, C_REC, PLOT_TO_RECALL, "rec", "triangle-down")
    curve(prec, prec_e, C_PREC, PLOT_TO_PRECISION, "prec", "triangle-up")
    curve(risk, risk_e, C_RISK, PLOT_TO_RISK, "risk",
          "square", dash="dash")
    curve(cost, cost_e, C_COST, f"עלות: שיחות מוטעות ↓ (@ ${cost_per_call:g})",
          "cost", "diamond", axis="y2", fmt=",.0f")

    # ── party-feasible ε region + privacy-optimal ε (★) ──────────────────
    if optimal_eps is not None and feasible_left is not None:
        fig.add_vrect(x0=feasible_left, x1=x_hi, fillcolor=_hex_to_rgba(C_FEAS, 0.12),
                      line_width=0, layer="below")
        fig.add_vline(x=optimal_eps, line=dict(color=C_FEAS, dash="dash", width=2))
        fig.add_trace(go.Scatter(
            x=[optimal_eps], y=[0.985], mode="markers+text",
            marker=dict(symbol="star", size=20, color=C_FEAS,
                        line=dict(color="white", width=1)),
            text=[f" ε={optimal_eps:g}"], textposition="middle right",
            textfont=dict(color=C_FEAS, size=13),
            name=f"★ ε אופטימלי לפרטיות = {optimal_eps:g}",
            hovertemplate=f"ε אופטימלי (הכי פרטי שעדיין ישים) = {optimal_eps:g}<extra></extra>"))

    # ── ε highlight from the slider ──────────────────────────────────────
    if selected_eps is not None:
        fig.add_vline(x=selected_eps, line=dict(color="#666", dash="dot", width=1.5),
                      annotation_text=f"ε נבחר = {selected_eps:g}",
                      annotation_position="top",
                      annotation_font=dict(color="#666", size=11))

    # ── constraint threshold lines ───────────────────────────────────────
    if min_recall:
        fig.add_hline(y=min_recall, line=dict(color=C_REC, dash="dot", width=1),
                      annotation_text=f"recall מינ' {min_recall:g}",
                      annotation_position="bottom left",
                      annotation_font=dict(color=C_REC, size=11))
    if min_precision:
        fig.add_hline(y=min_precision, line=dict(color=C_PREC, dash="dot", width=1),
                      annotation_text=f"precision מינ' {min_precision:g}",
                      annotation_position="top left",
                      annotation_font=dict(color=C_PREC, size=11))
    if max_loss is not None:
        fig.add_trace(go.Scatter(
            x=[x_lo, x_hi], y=[max_loss, max_loss], mode="lines", yaxis="y2",
            line=dict(color=C_COST, dash="dot", width=1),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=x_hi, y=max_loss, yref="y2", text=f"הפסד מקס' ${max_loss:,.0f}",
                           font=dict(color=C_COST, size=11), showarrow=False,
                           xanchor="right", yanchor="bottom")

    fig.update_layout(
        title=dict(text=f"פרטיות · תועלת · עלות  ({organiser})",
                   font=dict(size=16, color="#1a2340")),
        xaxis=dict(
            title=PLOT_TO_XAXIS,
            tickmode="array", tickvals=list(x), ticktext=[f"{v:g}" for v in x],
            range=[x_lo, x_hi], showgrid=True, gridcolor="#e0e4ec", zeroline=False,
            tickfont=dict(color="#1a2340"), title_font=dict(color="#1a2340"),
            linecolor="#c0ccde", tickcolor="#c0ccde"),
        yaxis=dict(
            title=PLOT_TO_YAXIS, range=[-0.02, 1.02], showgrid=True,
            gridcolor="#e0e4ec", zeroline=False, tickfont=dict(color="#1a2340"),
            title_font=dict(color="#1a2340"), linecolor="#c0ccde", tickcolor="#c0ccde"),
        yaxis2=dict(
            title=f"עלות שיחות מוטעות (USD @ ${cost_per_call:g})", overlaying="y",
            side="right", rangemode="tozero", showgrid=False,
            tickfont=dict(color=C_COST), title_font=dict(color=C_COST),
            linecolor=C_COST, tickcolor=C_COST),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5,
                    groupclick="togglegroup", font=dict(color="#1a2340")),
        height=580, margin=dict(l=20, r=20, t=60, b=130),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Heebo, sans-serif", color="#1a2340"),
        hovermode="x unified",
        modebar=dict(bgcolor="rgba(240,244,248,0.9)", color="#1a2340", activecolor="#3b6cb7"),
    )
    return fig


# =============================================================================
# PAGE: PRIVACY–UTILITY–COST TRADEOFF
# =============================================================================

def page_tradeoff():
    st.markdown(f'<div class="page-title">{TRADEOFF_PAGE_TITLE}</div>',
                unsafe_allow_html=True)
    st.markdown(TRADEOFF_INTRO)
    st.info(TRADEOFF_CONFIGURE_INFO)

    with st.expander(TRADEOFF_PARAMS_EXPANDER, expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(TRADEOFF_SEC_PARTY_POWER)
            frac_activists = st.slider(
                TRADEOFF_FRAC_ACTIVISTS, 0.01, 0.50, ANALYSIS_FRAC_ACTIVISTS, 0.01,
                help=TRADEOFF_FRAC_ACTIVISTS_HELP)
            calls_per_activist = st.number_input(
                TRADEOFF_CALLS, min_value=1, max_value=1_000_000,
                value=ANALYSIS_CALLS, step=50,
                help=TRADEOFF_CALLS_HELP)
        with c2:
            st.markdown(TRADEOFF_SEC_COST_RUNS)
            cost_per_call = st.number_input(
                TRADEOFF_COST, min_value=0.0, max_value=100.0,
                value=ANALYSIS_COST_PER_CALL, step=0.1,
                help=TRADEOFF_COST_HELP)
            repeats = st.slider(
                TRADEOFF_REPEATS, 1, 50, ANALYSIS_REPEATS,
                help=TRADEOFF_REPEATS_HELP)
        with c3:
            st.markdown(TRADEOFF_SEC_POP_MODEL)
            fraction = st.slider(
                TRADEOFF_FRACTION, 0.05, 1.0, ANALYSIS_FRACTION, 0.05,
                help=TRADEOFF_FRACTION_HELP)
            support_noise = st.slider(
                TRADEOFF_SUPPORT_NOISE, 0.0, 1.0, 0.0, 0.05,
                help=TRADEOFF_SUPPORT_NOISE_HELP)

        st.markdown(TRADEOFF_SEC_CONSTRAINTS)
        d1, d2, d3 = st.columns(3)
        with d1:
            min_recall = st.slider(TRADEOFF_MIN_RECALL, 0.0, 1.0, ANALYSIS_MIN_RECALL, 0.05,
                                   help=TRADEOFF_NO_CONSTRAINT_HELP)
        with d2:
            min_precision = st.slider(TRADEOFF_MIN_PRECISION, 0.0, 1.0,
                                      ANALYSIS_MIN_PRECISION, 0.05,
                                      help=TRADEOFF_NO_CONSTRAINT_HELP)
        with d3:
            max_loss = st.number_input(
                TRADEOFF_MAX_LOSS, min_value=0.0, max_value=1e9,
                value=ANALYSIS_MAX_LOSS, step=100.0,
                help=TRADEOFF_MAX_LOSS_HELP)

        run = st.button(TRADEOFF_RUN_BTN, use_container_width=True, type="primary")

    if run:
        st.session_state.tradeoff_params = dict(
            fraction=fraction, support_noise=support_noise, repeats=int(repeats),
            frac_activists=frac_activists, calls_per_activist=int(calls_per_activist),
            cost_per_call=cost_per_call, min_recall=min_recall,
            min_precision=min_precision, max_loss=max_loss,
        )

    if "tradeoff_params" not in st.session_state:
        return

    p = st.session_state.tradeoff_params
    spinner = st.empty()
    spinner.markdown(
        centered_spinner_html(SPINNER_RUN_TRADEOFF),
        unsafe_allow_html=True,
    )
    agg, org, n_pop, n_sup, n_act, eff_budget = run_analysis(
        p["fraction"], APP_KEEP_TOP_PARTIES, p["support_noise"], APP_ORGANISER,
        p["repeats"], p["frac_activists"], p["calls_per_activist"],
        p["cost_per_call"], SEED,
    )
    spinner.empty()

    # ── constraints + feasible region + privacy-optimal ε ───────────────
    mr = p["min_recall"] if p["min_recall"] > 0 else None
    mp = p["min_precision"] if p["min_precision"] > 0 else None
    finite = agg[np.isfinite(agg["epsilon"])].sort_values("epsilon").reset_index(drop=True)
    if p["max_loss"] > 0:
        ml = p["max_loss"]
    else:
        fc = finite["cost_misdirected_calls"]
        ml = pu._round_sig(0.5 * (float(fc.min()) + float(fc.max()))) if len(fc) else None

    feas = np.ones(len(finite), dtype=bool)
    if mr is not None:
        feas &= finite["recall_mean"].to_numpy() >= mr
    if mp is not None:
        feas &= finite["precision_mean"].to_numpy() >= mp
    if ml is not None:
        feas &= finite["cost_misdirected_calls"].to_numpy() <= ml

    xv = finite["epsilon"].to_numpy(dtype=float)
    span = float(xv.max() - xv.min()) or 1.0
    if feas.any():
        star = int(np.flatnonzero(feas).min())
        optimal_eps = float(xv[star])
        feasible_left = (float(xv.min() - 0.04 * span) if star == 0
                         else float(0.5 * (xv[star - 1] + xv[star])))
    else:
        optimal_eps, feasible_left = None, None

    # ── campaign-capacity summary ────────────────────────────────────────
    monetary_budget = eff_budget * p["cost_per_call"]
    s1, s2, s3, s4 = st.columns(4)

    s1.metric(TRADEOFF_METRIC_SUPPORTERS, f"{n_sup:,}")
    s2.metric(TRADEOFF_METRIC_ACTIVISTS, f"{n_act:,} × {p['calls_per_activist']:,}")
    s3.metric(TRADEOFF_METRIC_BUDGET, f"${monetary_budget:,.0f}")
    s4.metric(TRADEOFF_METRIC_PARTY, org)


    # ── ε highlight slider ───────────────────────────────────────────────
    st.markdown(TRADEOFF_LEGEND_CAPTION)

    eps_opts = [float(v) for v in xv]
    default_eps = optimal_eps if optimal_eps in eps_opts else eps_opts[len(eps_opts) // 2]
    selected_eps = st.select_slider(
        TRADEOFF_EPS_SLIDER, options=eps_opts, value=default_eps,
        format_func=lambda v: f"{v:g}",
    )

    fig = plot_tradeoff(agg, org, p["cost_per_call"], mr, mp, ml,
                        optimal_eps, feasible_left, selected_eps)
    st.plotly_chart(fig, use_container_width=True)

    if optimal_eps is not None:
        st.success(
            f"💡 ⭐**ε אופטימלי לפרטיות = {optimal_eps:g}** — הערך הפרטי ביותר "
            f"שעדיין עומד בכל אילוצי המפלגה הפעילים."
        )
    else:
        st.warning(TRADEOFF_NO_FEASIBLE_WARN)

    # ── read-out at the selected ε ───────────────────────────────────────
    row = finite[finite["epsilon"] == selected_eps].iloc[0]
    st.markdown(f"#### ערכים ב-ε = {selected_eps:g}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(TRADEOFF_METRIC_RISK, f"{row['status_disclosure_advantage']:.2f}",
              help=TRADEOFF_METRIC_RISK_HELP)
    m2.metric(TRADEOFF_METRIC_RECALL, f"{row['recall_mean']:.0%}",
              help=TRADEOFF_METRIC_RECALL_HELP)
    m3.metric(TRADEOFF_METRIC_PRECISION, f"{row['precision_mean']:.0%}",
              help=TRADEOFF_METRIC_PRECISION_HELP)
    m4.metric(TRADEOFF_METRIC_COST, f"${row['cost_misdirected_calls']:,.0f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_styles()
    init_session_state()
    render_top_bar(ACTIVIST_NAME.split(":")[-1])
    page = render_sidebar()

    if   page == "dashboard": page_dashboard()
    elif page == "voters":    page_voter_list()
    elif page == "tradeoff":  page_tradeoff()


if __name__ == "__main__":
    main()
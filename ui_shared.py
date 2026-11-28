"""Look and navigation shared by the three dashboard pages.

Streamlit's automatic page list is hidden in .streamlit/config.toml.
The sidebar offers the same delivery choice on every page and links to research.
"""
import streamlit as st

THEME_CSS = """
    <style>
    :root {
        --ink: #111827;
        --ink-2: #374151;
        --muted: #6B7280;
        --line: #E5E7EB;
        --line-soft: #F1F2F5;
        --surface: #FFFFFF;
        --canvas: #F6F7FB;
        --brand: #EA580C;
        --brand-soft: #FFF1E7;
        --route: #2563EB;
        --good: #15803D;
        --good-soft: #DCFCE7;
        --bad: #B91C1C;
        --bad-soft: #FEE2E2;
        --warn-soft: #FEF3C7;
    }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    /* Compact option buttons so 3-4 choices fit on one line in the sidebar */
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
        padding-left: 9px !important; padding-right: 9px !important; min-width: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p { font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* ---------- Sidebar ---------- */
    .brand { display:flex; align-items:center; gap:10px; margin-bottom: 0.4rem; }
    .brand-logo {
        width: 36px; height: 36px; border-radius: 10px; background: var(--brand);
        display:flex; align-items:center; justify-content:center; font-size: 19px;
    }
    .brand-name { font-weight: 800; font-size: 1.02rem; color: var(--ink); line-height: 1.1; }
    .brand-sub { font-size: 0.75rem; color: var(--muted); }
    .step-head {
        display:flex; align-items:center; gap:9px; margin: 1.1rem 0 0.35rem;
        font-weight: 700; font-size: 0.95rem; color: var(--ink);
    }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%; background: var(--brand-soft); color: var(--brand);
        display:inline-flex; align-items:center; justify-content:center; font-size: 0.75rem; font-weight: 800;
    }
    .model-hint { font-size: 0.8rem; color: var(--muted); margin-top: -0.45rem; margin-bottom: 0.4rem; }

    /* ---------- Hero ---------- */
    .hero h1 {
        font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; color: var(--ink);
        margin: 0; padding: 0; line-height: 1.15;
    }
    .hero p { color: var(--muted); font-size: 0.98rem; margin: 0.35rem 0 0; max-width: 760px; }
    .eyebrow {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
        color: var(--brand); margin-bottom: 0.3rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--surface); padding: 5px; border-radius: 12px;
        border: 1px solid var(--line); width: fit-content; max-width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px; padding: 0 18px; border-radius: 8px; background: transparent;
    }
    .stTabs [data-baseweb="tab"] p { font-weight: 600; font-size: 0.92rem; }
    .stTabs [aria-selected="true"] { background: var(--brand-soft); }
    .stTabs [aria-selected="true"] p { color: var(--brand); }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Chips ---------- */
    .chips { display:flex; flex-wrap:wrap; gap:8px; margin: 0.2rem 0 1rem; }
    .chip {
        display:inline-flex; align-items:center; gap:6px; background: var(--surface);
        border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px;
        font-size: 0.83rem; color: var(--ink-2); font-weight: 500;
    }

    /* ---------- KPI cards ---------- */
    .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 1.1rem; }
    .kpi {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    }
    .kpi-label { font-size: 0.82rem; color: var(--muted); font-weight: 600; display:flex; align-items:center; gap:7px; }
    .kpi-value { font-size: 1.75rem; font-weight: 800; color: var(--ink); margin-top: 6px; letter-spacing: -0.01em; }
    .kpi-value.text { font-size: 1.2rem; line-height: 1.3; }
    .kpi-sub { font-size: 0.8rem; margin-top: 4px; color: var(--muted); }
    .kpi-sub.good { color: var(--good); font-weight: 600; }
    .kpi-sub.bad { color: var(--bad); font-weight: 600; }

    /* ---------- Cards / panels ---------- */
    .panel {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
        padding: 18px 18px 8px;
    }
    .panel-title { font-weight: 700; font-size: 1rem; color: var(--ink); margin-bottom: 2px; }
    .panel-sub { font-size: 0.8rem; color: var(--muted); margin-bottom: 12px; }
    .note {
        display:flex; gap:10px; align-items:flex-start; border-radius: 12px; padding: 10px 14px;
        font-size: 0.84rem; line-height: 1.45; color: var(--ink-2); margin: 0 0 12px;
        background: #EFF6FF; border: 1px solid #BFDBFE;
    }
    .note.warm { background: var(--brand-soft); border-color: #FED7AA; }
    .note b { color: var(--ink); }
    .note-icon { font-size: 1rem; line-height: 1.3; }
    .real-sim { display:flex; flex-wrap:wrap; gap: 6px 16px; font-size: 0.8rem; color: var(--muted); margin: 8px 2px 0; }
    .real-sim .tag { font-weight: 700; padding: 1px 8px; border-radius: 999px; margin-right: 4px; font-size: 0.72rem; }
    .real-sim .tag.real { background: var(--good-soft); color: var(--good); }
    .real-sim .tag.sim { background: var(--warn-soft); color: #92400E; }
    .map-frame { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: var(--surface); }

    /* ---------- Trip timeline ---------- */
    .stops { position: relative; margin: 6px 0 10px; }
    .stop { display:flex; gap: 12px; position: relative; padding-bottom: 16px; }
    .stop:not(:last-child)::before {
        content: ""; position: absolute; left: 15px; top: 32px; bottom: 0; width: 2px; background: var(--line);
    }
    .stop-icon {
        flex: 0 0 32px; height: 32px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
        font-size: 15px; background: var(--line-soft); border: 1px solid var(--line);
    }
    .stop-icon.rider { background: #FFEDD5; border-color: #FED7AA; }
    .stop-icon.store { background: #DCFCE7; border-color: #BBF7D0; }
    .stop-icon.home { background: #DBEAFE; border-color: #BFDBFE; }
    .stop-body { flex: 1; min-width: 0; }
    .stop-row { display:flex; justify-content: space-between; align-items: baseline; gap: 8px; }
    .stop-title { font-weight: 650; font-size: 0.9rem; color: var(--ink); }
    .stop-time { font-weight: 700; font-size: 0.85rem; color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
    .stop-meta { font-size: 0.78rem; color: var(--muted); margin-top: 1px; overflow-wrap: anywhere; }
    .pill { display:inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 8px; border-radius: 999px; margin-left: 4px; }
    .pill.good { background: var(--good-soft); color: var(--good); }
    .pill.bad { background: var(--bad-soft); color: var(--bad); }

    .facts { border-top: 1px solid var(--line-soft); margin-top: 4px; padding-top: 8px; }
    .fact { display:flex; justify-content: space-between; padding: 6px 0; font-size: 0.84rem; border-bottom: 1px solid var(--line-soft); gap: 10px; }
    .fact:last-child { border-bottom: none; }
    .fact span:first-child { color: var(--muted); }
    .fact span:last-child { color: var(--ink); font-weight: 600; text-align: right; }

    .section-title { font-size: 1.15rem; font-weight: 750; color: var(--ink); margin: 1.4rem 0 0.15rem; }
    .section-sub { font-size: 0.88rem; color: var(--muted); margin-bottom: 0.8rem; }

    div[data-testid="stExpander"] details { background: var(--surface); border-radius: 12px; }
    .main iframe, section.stMain iframe { border-radius: 14px; border: 1px solid var(--line) !important; }

    /* ---------- Page switcher (shared by both pages) ---------- */
    .nav-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--muted); margin: 0.9rem 0 0.35rem;
    }
    /* Give the delivery choice the same full-width, 44px targets as the research link. */
    section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stButtonGroup"] [aria-label="Choose a delivery"]) {
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"]:has([aria-label="Choose a delivery"]) {
        width: 100%; margin-bottom: 12px;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [aria-label="Choose a delivery"] {
        display: flex; width: 100%; max-width: 100% !important; min-height: 44px;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [aria-label="Choose a delivery"] button {
        flex: 1 1 0; min-height: 44px; justify-content: center; white-space: nowrap;
        padding-left: 4px !important; padding-right: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; margin-bottom: 6px;
        background: var(--surface); display: flex; align-items: center; min-height: 44px;
        width: 100%; box-sizing: border-box;
    }
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { border-color: #FED7AA; }
    .nav-current {
        border: 1.5px solid var(--brand); background: var(--brand-soft); border-radius: 10px;
        padding: 8px 11px; margin-bottom: 8px;
    }
    .nav-current-title {
        display:flex; align-items:center; gap: 6px; font-weight: 700; font-size: 0.92rem; color: var(--ink);
    }
    .nav-here {
        margin-left: auto; font-size: 0.66rem; font-weight: 700; color: var(--brand); background: var(--surface);
        border-radius: 999px; padding: 1px 8px; white-space: nowrap;
    }
    .nav-desc { font-size: 0.78rem; color: var(--ink-2); margin-top: 2px; }
    .howto { display:flex; flex-wrap:wrap; gap: 10px; margin: 0.9rem 0 0.2rem; }
    .howto-step {
        display:flex; align-items:center; gap: 8px; background: var(--surface); border: 1px solid var(--line);
        border-radius: 999px; padding: 5px 14px 5px 6px; font-size: 0.85rem; color: var(--ink-2);
    }
    .howto-step .step-num { flex: 0 0 22px; }
    </style>

"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def delivery_mode_switch(current: str | None) -> None:
    """Choose one or several deliveries from the sidebar on any page."""
    selected = st.segmented_control(
        "Choose a delivery",
        ("One delivery", "Several deliveries"),
        default=current,
        key=f"sidebar_delivery_mode_{(current or 'data').replace(' ', '_').lower()}",
    )
    if selected and selected != current:
        st.switch_page("app.py" if selected == "One delivery" else "pages/2_Multi_Order_D3QN.py")


def sidebar_header(current: str) -> None:
    """Brand and consistent project navigation. Call inside the sidebar."""
    st.markdown(
        '<div class="brand"><div class="brand-logo">🛵</div><div>'
        '<div class="brand-name">Delivery Route Planner</div>'
        '<div class="brand-sub">AI route planning demo</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Explore the project</div>', unsafe_allow_html=True)
    mode = {
        "app.py": "One delivery",
        "pages/2_Multi_Order_D3QN.py": "Several deliveries",
    }.get(current)
    delivery_mode_switch(mode)
    if current == "pages/3_Data_and_Experiments.py":
        st.markdown(
            '<div class="nav-current"><div class="nav-current-title">🔬 Data & experiments'
            '<span class="nav-here">You are here</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.page_link(
            "pages/3_Data_and_Experiments.py",
            label="Data & experiments",
            icon="🔬",
            help="How the shared data was prepared and the models were tested.",
        )
    st.divider()


def html_kpi(icon: str, label: str, value: str, sub: str = "", tone: str = "", text: bool = False) -> str:
    """Render a shared summary card for either delivery mode."""
    return (
        f'<div class="kpi"><div class="kpi-label"><span>{icon}</span>{label}</div>'
        f'<div class="kpi-value{" text" if text else ""}">{value}</div><div class="kpi-sub {tone}">{sub}</div></div>'
    )


def route_tabs():
    """The two route pages use the same navigation labels and ordering."""
    return st.tabs(["🗺️  1 · Try a route", "📊  2 · Compare results"])


def step_head(num: int, text: str) -> None:
    st.markdown(f'<div class="step-head"><span class="step-num">{num}</span>{text}</div>', unsafe_allow_html=True)


def page_hero(eyebrow: str, title: str, text: str, steps: list) -> None:
    """Page title, one-sentence purpose and a numbered 'how to use' strip."""
    howto = "".join(f'<span class="howto-step"><span class="step-num">{i}</span>{s}</span>'
                    for i, s in enumerate(steps, 1))
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p>'
        f'<div class="howto">{howto}</div></div>',
        unsafe_allow_html=True,
    )

import streamlit as st

def apply_custom_style():
    """Injects modern CSS for shadows, rounded corners, and interactive elements."""
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        
        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e6e6e6;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label { color: #6b7280 !important; font-size: 0.9rem !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-weight: 600 !important; }

        /* HEADER CARD */
        .header-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            padding: 2rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
        }
        .header-card h1 { color: white !important; margin: 0; font-weight: 700; }
        .header-card p { color: #a5b4fc !important; margin-top: 0.5rem; font-size: 1.1rem; }

        /* RISK HERO CARDS */
        .risk-card {
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .risk-card:hover { transform: translateY(-5px); }
        
        .winner-card { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
        .loser-card { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
        
        .risk-title { font-size: 1rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .risk-ticker { font-size: 3rem; font-weight: 800; margin: 10px 0; }
        .risk-metric { font-size: 1.5rem; font-weight: 600; background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; display: inline-block; }
        .risk-desc { margin-top: 10px; font-size: 0.9rem; opacity: 0.9; }

        /* AI SUGGESTION BOX */
        .ai-box {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 5px solid #22c55e;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            color: #14532d;
        }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    st.markdown("""
        <div class="header-card">
            <h1>Risk Intelligence</h1>
            <p>Sharpe Ratio Maximization & Algorithmic Reallocation</p>
        </div>
    """, unsafe_allow_html=True)

def display_risk_hero(best_row, worst_row):
    """Displays the massive green/red cards for best/worst assets."""
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
        <div class="risk-card winner-card">
            <div class="risk-title">🏆 Efficiency King</div>
            <div class="risk-ticker">{best_row['ticker']}</div>
            <div class="risk-metric">Sharpe {best_row['sharpe']:.2f}</div>
            <div class="risk-desc">Highest Risk-Adjusted Return</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="risk-card loser-card">
            <div class="risk-title">⚠️ Drag on Portfolio</div>
            <div class="risk-ticker">{worst_row['ticker']}</div>
            <div class="risk-metric">Sharpe {worst_row['sharpe']:.2f}</div>
            <div class="risk-desc">Lowest Risk-Adjusted Return</div>
        </div>
        """, unsafe_allow_html=True)
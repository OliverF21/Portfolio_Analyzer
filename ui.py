import streamlit as st

def apply_custom_style():
    """Injects modern CSS for shadows, rounded corners, and interactive elements."""
    st.markdown("""
        <style>
        /* MAIN CONTAINER BACKGROUND */
        .main {
            background-color: #f8f9fa;
        }
        
        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-color: #6366f1;
        }

        /* DATAFRAMES */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        /* CUSTOM HEADER CARD */
        .header-card {
            background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
            padding: 2rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
        }
        .header-card h1 {
            color: white !important;
            margin: 0;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        .header-card p {
            color: #e0e7ff;
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    """Renders the top banner."""
    st.markdown("""
        <div class="header-card">
            <h1>Portfolio Analyst Pro</h1>
            <p>Advanced Conservative Risk Analytics & Performance Tracking</p>
        </div>
    """, unsafe_allow_html=True)

    
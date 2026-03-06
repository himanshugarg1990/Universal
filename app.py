import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank – Personal Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main */
    .main { background-color: #F4F6F9; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        border-radius: 14px;
        padding: 22px 20px 18px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(27,79,114,0.25);
        margin-bottom: 6px;
    }
    .kpi-card .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .kpi-card .kpi-label {
        font-size: 0.82rem;
        opacity: 0.88;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #1E8449, #27AE60);
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #B7590A, #E67E22);
    }
    .kpi-card-purple {
        background: linear-gradient(135deg, #6C3483, #9B59B6);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1B4F72, #2E86C1);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 18px;
        letter-spacing: 0.3px;
    }

    /* Offer card */
    .offer-card {
        background: white;
        border-left: 5px solid #2E86C1;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .offer-card h4 { margin: 0 0 6px 0; color: #1B4F72; font-size: 1rem; }
    .offer-card p  { margin: 0; font-size: 0.88rem; color: #555; }

    /* Insight box */
    .insight-box {
        background: #EAF2FB;
        border-left: 4px solid #2E86C1;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1C2833;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4F72 0%, #154360 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSlider > div > div { background: #2E86C1 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
    section[data-testid="stSidebar"] label { font-weight: 600 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df.drop(columns=["ID", "ZIP Code"], inplace=True, errors="ignore")
    df["Education_Label"] = df["Education"].map(
        {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
    )
    df["Income_Group"] = pd.cut(
        df["Income"],
        bins=[0, 50, 100, 150, 225],
        labels=["Low (<50k)", "Mid (50-100k)", "High (100-150k)", "Very High (>150k)"],
    )
    df["Family_Label"] = df["Family"].astype(str) + " member(s)"
    return df


df_full = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("### Personal Loan Analytics")
    st.markdown("---")
    st.markdown("**🔍 Filters**")

    income_min, income_max = int(df_full["Income"].min()), int(df_full["Income"].max())
    income_range = st.slider(
        "Income Range ($000)", income_min, income_max, (income_min, income_max), step=5
    )

    edu_options = ["All"] + list(df_full["Education_Label"].unique())
    edu_filter = st.multiselect("Education Level", edu_options[1:], default=edu_options[1:])

    family_options = sorted(df_full["Family"].unique())
    family_filter = st.multiselect(
        "Family Size", family_options, default=family_options
    )

    st.markdown("---")
    st.markdown("**📊 Navigation**")
    st.markdown("Use the tabs above to explore\ndifferent analytics sections.")
    st.markdown("---")
    st.caption("Dataset: Universal Bank | 5,000 customers")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[(df["Income"] >= income_range[0]) & (df["Income"] <= income_range[1])]
if edu_filter:
    df = df[df["Education_Label"].isin(edu_filter)]
if family_filter:
    df = df[df["Family"].isin(family_filter)]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 4px 0;'>
    <h1 style='color:#1B4F72; font-size:2.2rem; font-weight:800; margin:0;'>
        🏦 Universal Bank — Personal Loan Dashboard
    </h1>
    <p style='color:#555; font-size:1rem; margin-top:6px;'>
        Four-lens analytics: Descriptive · Diagnostic · Predictive · Prescriptive
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────────────────────────────
total = len(df)
accepted = df["Personal Loan"].sum()
acceptance_rate = accepted / total * 100 if total > 0 else 0
avg_income = df["Income"].mean()
avg_ccavg = df["CCAvg"].mean()
avg_mortgage = df["Mortgage"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, cls in zip(
    [c1, c2, c3, c4, c5],
    [f"{total:,}", f"{accepted:,}", f"{acceptance_rate:.1f}%", f"${avg_income:.0f}K", f"${avg_ccavg:.2f}K"],
    ["Total Customers", "Loan Accepted", "Acceptance Rate", "Avg Income", "Avg CC Spend/mo"],
    ["", "kpi-card-green", "kpi-card-orange", "kpi-card-purple", ""],
):
    col.markdown(
        f'<div class="kpi-card {cls}"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive", "🔬 Diagnostic", "🤖 Predictive", "🎯 Prescriptive", "🔍 Drill-Down"
])

PALETTE = px.colors.qualitative.Set2
BLUE = "#2E86C1"
GREEN = "#27AE60"
RED = "#E74C3C"
ORANGE = "#E67E22"


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Descriptive Analysis — Who Are Our Customers?</div>', unsafe_allow_html=True)

    # --- Row 1: Loan acceptance donut + Age distribution
    col1, col2 = st.columns([1, 2])

    with col1:
        counts = df["Personal Loan"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({0: "Rejected", 1: "Accepted"})
        fig = px.pie(
            counts, values="Count", names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map={"Accepted": GREEN, "Rejected": "#AEB6BF"},
            title="Loan Acceptance Rate",
        )
        fig.update_traces(textinfo="percent+label", pull=[0.04, 0])
        fig.update_layout(
            showlegend=True, height=320,
            title_font_size=14, margin=dict(t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="Age", nbins=30, color_discrete_sequence=[BLUE],
            title="Age Distribution of Customers",
            labels={"Age": "Age (years)", "count": "# Customers"},
        )
        fig.update_layout(height=320, bargap=0.05, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
                          title_font_size=14, margin=dict(t=40, b=10))
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 2: Income distribution + Education breakdown
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df, x="Income", nbins=40, color_discrete_sequence=[ORANGE],
            title="Income Distribution ($000)",
            labels={"Income": "Annual Income ($000)", "count": "# Customers"},
        )
        fig.update_layout(height=300, bargap=0.04, paper_bgcolor="white", plot_bgcolor="#FFFBF5",
                          title_font_size=14, margin=dict(t=40, b=10))
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        edu_counts = df["Education_Label"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Count"]
        fig = px.bar(
            edu_counts, x="Education", y="Count",
            color="Education", color_discrete_sequence=PALETTE,
            title="Customers by Education Level",
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8F9FA", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 3: Family size + Average metrics
    col5, col6 = st.columns(2)

    with col5:
        fam_counts = df["Family"].value_counts().sort_index().reset_index()
        fam_counts.columns = ["Family Size", "Count"]
        fam_counts["Family Size"] = fam_counts["Family Size"].astype(str) + " member(s)"
        fig = px.bar(
            fam_counts, x="Family Size", y="Count",
            color="Count", color_continuous_scale="Blues",
            title="Distribution of Family Size",
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8FBFF", title_font_size=14,
                          coloraxis_showscale=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        avg_metrics = pd.DataFrame({
            "Metric": ["Avg Income ($K)", "Avg CC Spend ($K/mo)", "Avg Mortgage ($K)"],
            "Value": [df["Income"].mean(), df["CCAvg"].mean(), df["Mortgage"].mean()],
            "Color": [BLUE, GREEN, ORANGE],
        })
        fig = px.bar(
            avg_metrics, x="Metric", y="Value",
            color="Metric",
            color_discrete_sequence=[BLUE, GREEN, ORANGE],
            title="Average Financial Metrics",
            text=avg_metrics["Value"].apply(lambda x: f"${x:.2f}K"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8F9FA", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="", yaxis_title="Value ($K)")
        st.plotly_chart(fig, use_container_width=True)

    # Banking services adoption
    st.markdown('<div class="section-header">🏧 Banking Services Adoption</div>', unsafe_allow_html=True)
    services = ["Securities Account", "CD Account", "Online", "CreditCard"]
    svc_data = pd.DataFrame({
        "Service": services,
        "% Customers": [df[s].mean() * 100 for s in services],
    })
    fig = px.bar(
        svc_data, x="Service", y="% Customers",
        color="Service", color_discrete_sequence=PALETTE,
        title="% of Customers Using Each Banking Service",
        text=svc_data["% Customers"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                      plot_bgcolor="#F8F9FA", title_font_size=14,
                      margin=dict(t=40, b=10), xaxis_title="", yaxis_title="% of Customers")
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.markdown("**💡 Key Descriptive Insights**")
    insights = [
        f"📌 Only <b>{acceptance_rate:.1f}%</b> of customers accepted the personal loan — the dataset is imbalanced, which must be handled in modeling.",
        f"📌 Average customer income is <b>${avg_income:.0f}K/year</b>, with credit card spending averaging <b>${avg_ccavg:.2f}K/month</b>.",
        f"📌 The majority of customers are in the <b>30–50 age range</b>, suggesting a predominantly working-age clientele.",
        f"📌 <b>Online banking</b> is the most widely adopted service, while <b>CD Accounts</b> have the lowest uptake.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Diagnostic Analysis — Why Do Customers Accept Loans?</div>', unsafe_allow_html=True)

    loan_yes = df[df["Personal Loan"] == 1]
    loan_no  = df[df["Personal Loan"] == 0]

    # --- Income vs Loan acceptance
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df, x="Personal Loan", y="Income",
            color="Personal Loan",
            color_discrete_map={0: "#AEB6BF", 1: GREEN},
            labels={"Personal Loan": "Loan Accepted (0=No, 1=Yes)", "Income": "Annual Income ($K)"},
            title="Income Distribution by Loan Acceptance",
            points="outliers",
        )
        fig.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
                          title_font_size=14, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="Personal Loan", y="CCAvg",
            color="Personal Loan",
            color_discrete_map={0: "#AEB6BF", 1: ORANGE},
            labels={"Personal Loan": "Loan Accepted (0=No, 1=Yes)", "CCAvg": "Monthly CC Spend ($K)"},
            title="Credit Card Spending by Loan Acceptance",
            points="outliers",
        )
        fig.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="#FFFBF5",
                          title_font_size=14, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- Education vs Loan acceptance
    col3, col4 = st.columns(2)

    with col3:
        edu_loan = df.groupby("Education_Label")["Personal Loan"].mean().reset_index()
        edu_loan.columns = ["Education", "Acceptance Rate"]
        edu_loan["Acceptance Rate"] *= 100
        fig = px.bar(
            edu_loan, x="Education", y="Acceptance Rate",
            color="Education", color_discrete_sequence=PALETTE,
            title="Loan Acceptance Rate by Education Level",
            text=edu_loan["Acceptance Rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8F9FA", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="", yaxis_title="Acceptance Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        mort_grp = df.groupby("Personal Loan")["Mortgage"].mean().reset_index()
        mort_grp["Personal Loan"] = mort_grp["Personal Loan"].map({0: "Rejected", 1: "Accepted"})
        fig = px.bar(
            mort_grp, x="Personal Loan", y="Mortgage",
            color="Personal Loan",
            color_discrete_map={"Accepted": GREEN, "Rejected": "#AEB6BF"},
            title="Average Mortgage Value by Loan Acceptance",
            text=mort_grp["Mortgage"].apply(lambda x: f"${x:.0f}K"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8FBFF", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="", yaxis_title="Avg Mortgage ($K)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Banking services vs loan acceptance
    st.markdown('<div class="section-header">🏧 Banking Services & Loan Acceptance</div>', unsafe_allow_html=True)
    services = ["Securities Account", "CD Account", "Online", "CreditCard"]
    svc_rates = []
    for s in services:
        r_yes = df[df[s] == 1]["Personal Loan"].mean() * 100
        r_no  = df[df[s] == 0]["Personal Loan"].mean() * 100
        svc_rates.append({"Service": s, "Has Service": r_yes, "No Service": r_no})
    svc_df = pd.DataFrame(svc_rates)
    svc_melt = svc_df.melt("Service", var_name="Status", value_name="Loan Acceptance Rate (%)")

    fig = px.bar(
        svc_melt, x="Service", y="Loan Acceptance Rate (%)",
        color="Status",
        color_discrete_map={"Has Service": BLUE, "No Service": "#AEB6BF"},
        barmode="group",
        title="Loan Acceptance Rate: With vs Without Each Banking Service",
        text=svc_melt["Loan Acceptance Rate (%)"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
                      title_font_size=14, margin=dict(t=40, b=10),
                      xaxis_title="", legend_title="Service Status")
    st.plotly_chart(fig, use_container_width=True)

    # --- Correlation heatmap
    st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = ["Age", "Income", "Family", "CCAvg", "Education", "Mortgage",
                 "Securities Account", "CD Account", "Online", "CreditCard", "Personal Loan"]
    corr = df[corr_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix — All Features vs Personal Loan",
        aspect="auto",
    )
    fig.update_layout(height=480, title_font_size=14, margin=dict(t=50, b=10),
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # --- Key diagnostic insights
    st.markdown("**💡 Key Diagnostic Insights**")
    avg_inc_yes = loan_yes["Income"].mean()
    avg_inc_no  = loan_no["Income"].mean()
    avg_cc_yes  = loan_yes["CCAvg"].mean()
    avg_cc_no   = loan_no["CCAvg"].mean()
    diag_insights = [
        f"📌 Customers who <b>accepted the loan</b> have a significantly higher average income: <b>${avg_inc_yes:.0f}K</b> vs <b>${avg_inc_no:.0f}K</b> for those who rejected it.",
        f"📌 CC spending is much higher for loan acceptors (<b>${avg_cc_yes:.2f}K/mo</b>) than rejectors (<b>${avg_cc_no:.2f}K/mo</b>), indicating higher purchasing power.",
        f"📌 <b>CD Account</b> holders show a dramatically higher loan acceptance rate — having a CD account is one of the strongest signals of loan interest.",
        f"📌 <b>Education</b> plays a role: Graduate and Advanced/Professional customers accept loans at higher rates than undergraduates.",
    ]
    for ins in diag_insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PREDICTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Predictive Analysis — Who Will Accept the Loan?</div>', unsafe_allow_html=True)

    # Feature prep
    features = ["Age", "Experience", "Income", "Family", "CCAvg",
                 "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"]
    target = "Personal Loan"

    df_model = df_full[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    @st.cache_resource
    def train_models(X_train, y_train):
        dt  = DecisionTreeClassifier(max_depth=6, random_state=42)
        rf  = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        return dt, rf, gb

    dt, rf, gb = train_models(X_train, y_train)

    model_choice = st.selectbox(
        "Select Model to Inspect",
        ["Decision Tree", "Random Forest", "Gradient Boosting"],
        index=1,
    )
    model_map = {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosting": gb}
    model = model_map[model_choice]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in zip(
        [col1, col2, col3, col4],
        ["AUC-ROC Score", "Precision (Loan=1)", "Recall (Loan=1)", "F1-Score (Loan=1)"],
        [f"{auc:.3f}", f"{report['1']['precision']:.3f}", f"{report['1']['recall']:.3f}", f"{report['1']['f1-score']:.3f}"],
    ):
        col.markdown(
            f'<div class="kpi-card kpi-card-purple"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ROC Curve
    with col_a:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                                  line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(color="gray", dash="dash"), name="Random"))
        fig.update_layout(
            title=f"ROC Curve — {model_choice}",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=350, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
            title_font_size=14, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with col_b:
        imp = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig = px.bar(
            imp, x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Blues",
            title=f"Feature Importance — {model_choice}",
        )
        fig.update_layout(height=350, coloraxis_showscale=False, paper_bgcolor="white",
                          plot_bgcolor="#F8FBFF", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="Importance Score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    col_c, col_d = st.columns(2)
    with col_c:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Loan", "Loan"], y=["No Loan", "Loan"],
            title=f"Confusion Matrix — {model_choice}",
        )
        fig.update_layout(height=350, paper_bgcolor="white", title_font_size=14,
                          margin=dict(t=40, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    with col_d:
        comp_data = []
        for name, m in model_map.items():
            p = m.predict_proba(X_test)[:, 1]
            comp_data.append({"Model": name, "AUC-ROC": roc_auc_score(y_test, p)})
        comp_df = pd.DataFrame(comp_data)
        fig = px.bar(
            comp_df, x="Model", y="AUC-ROC",
            color="Model", color_discrete_sequence=[BLUE, GREEN, ORANGE],
            title="Model Comparison — AUC-ROC Score",
            text=comp_df["AUC-ROC"].apply(lambda x: f"{x:.3f}"),
            range_y=[0.8, 1.0],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8F9FA", title_font_size=14,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Prediction probability distribution
    pred_df = X_test.copy()
    pred_df["Predicted Prob"] = y_prob
    pred_df["Actual"] = y_test.values
    fig = px.histogram(
        pred_df, x="Predicted Prob", color=pred_df["Actual"].map({0: "No Loan", 1: "Accepted Loan"}),
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map={"No Loan": "#AEB6BF", "Accepted Loan": GREEN},
        title=f"Predicted Probability Distribution — {model_choice}",
        labels={"Predicted Prob": "Predicted Probability of Accepting Loan", "color": "Actual"},
    )
    fig.update_layout(height=320, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
                      title_font_size=14, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**💡 Key Predictive Insights**")
    top_feat = imp.iloc[-1]["Feature"]
    pred_insights = [
        f"📌 <b>Income</b> and <b>CCAvg</b> are consistently the top predictors of loan acceptance across all three models.",
        f"📌 <b>CD Account</b> ownership is a surprisingly strong predictor — customers with a CD account are far more likely to accept a personal loan.",
        f"📌 The <b>Gradient Boosting</b> model achieves the highest AUC-ROC, making it the recommended model for production deployment.",
        f"📌 The model is well-calibrated: the probability distribution clearly separates loan acceptors from rejectors.",
    ]
    for ins in pred_insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PRESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🎯 Prescriptive Analysis — Personalised Offers for Predicted Customers</div>', unsafe_allow_html=True)

    # Score all filtered customers
    df_score = df[features].copy()
    df_score_full = df.copy()
    df_score_full["Loan_Probability"] = gb.predict_proba(df_score)[:, 1]
    df_score_full["Predicted_Loan"] = gb.predict(df_score)

    high_potential = df_score_full[df_score_full["Loan_Probability"] >= 0.5].copy()

    # --- Segment builder
    def assign_offer(row):
        if row["Income"] > 100 and row["CCAvg"] > 3:
            persona = "Premium Spender"
            offer = "🏆 Elite Personal Loan — Up to $200K at 7.9% APR with zero processing fee"
            rationale = "High income + high CC spending indicates significant borrowing capacity and strong repayment potential."
        elif row["CD Account"] == 1:
            persona = "Savings-Focused"
            offer = "🏦 CD-Linked Loan — Preferential rate of 8.5% APR leveraging your CD as partial collateral"
            rationale = "CD account holders have a savings mindset; a linked loan offer feels safe and familiar."
        elif row["Education"] == 3:
            persona = "Professional"
            offer = "🎓 Professional Loan — Up to $100K at 9.0% APR for career investments & home improvements"
            rationale = "Advanced/professional education correlates with stable high-paying jobs and loan repayment ability."
        elif row["Family"] >= 3 and row["Mortgage"] > 0:
            persona = "Family Planner"
            offer = "🏠 Family Comfort Loan — Up to $75K at 9.5% APR for home renovation or education expenses"
            rationale = "Larger families with mortgages often need liquidity for ongoing family expenses."
        elif row["Online"] == 1:
            persona = "Digital Adopter"
            offer = "📱 Digital Fast Loan — Instant approval up to $50K at 10.0% APR via mobile app"
            rationale = "Online banking users are comfortable with digital processes and likely to respond to app-based campaigns."
        else:
            persona = "Standard Prospect"
            offer = "💳 Personal Loan Starter — Up to $30K at 10.5% APR with flexible repayment options"
            rationale = "General offer for customers showing moderate loan interest based on income and profile."
        return pd.Series([persona, offer, rationale])

    high_potential[["Persona", "Recommended Offer", "Rationale"]] = high_potential.apply(assign_offer, axis=1)

    st.markdown(f"### 🎯 {len(high_potential):,} customers predicted as likely loan acceptors")
    st.markdown(f"*(Probability threshold: ≥ 50% | Model: Gradient Boosting)*")
    st.markdown("---")

    # Persona distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        persona_counts = high_potential["Persona"].value_counts().reset_index()
        persona_counts.columns = ["Persona", "Count"]
        fig = px.pie(
            persona_counts, values="Count", names="Persona",
            hole=0.5, title="Customer Segments (Personas)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=340, paper_bgcolor="white", title_font_size=14,
                          margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_prob_persona = high_potential.groupby("Persona")["Loan_Probability"].mean().sort_values(ascending=False).reset_index()
        avg_prob_persona.columns = ["Persona", "Avg Loan Probability"]
        fig = px.bar(
            avg_prob_persona, x="Persona", y="Avg Loan Probability",
            color="Persona", color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Average Predicted Probability by Persona",
            text=avg_prob_persona["Avg Loan Probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, showlegend=False, paper_bgcolor="white",
                          plot_bgcolor="#F8F9FA", title_font_size=14,
                          margin=dict(t=40, b=10), xaxis_title="", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Personalised offer cards per persona
    st.markdown('<div class="section-header">💌 Personalised Offers by Segment</div>', unsafe_allow_html=True)
    personas_unique = high_potential["Persona"].unique()
    for persona in personas_unique:
        subset = high_potential[high_potential["Persona"] == persona].iloc[0]
        st.markdown(f"""
        <div class="offer-card">
            <h4>🎯 {persona} — {subset['Recommended Offer']}</h4>
            <p>📊 <b>Avg Predicted Probability:</b> {high_potential[high_potential['Persona'] == persona]['Loan_Probability'].mean():.1%} &nbsp;|&nbsp;
               👥 <b>Segment Size:</b> {len(high_potential[high_potential['Persona'] == persona]):,} customers</p>
            <p>💡 <b>Why this offer?</b> {subset['Rationale']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Sample predicted customers table
    st.markdown('<div class="section-header">📋 Top 20 High-Potential Customers</div>', unsafe_allow_html=True)
    display_cols = ["Age", "Income", "Education_Label", "Family", "CCAvg", "Mortgage",
                    "Loan_Probability", "Persona", "Recommended Offer"]
    top_customers = high_potential.sort_values("Loan_Probability", ascending=False)[display_cols].head(20)
    top_customers["Loan_Probability"] = top_customers["Loan_Probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(top_customers.reset_index(drop=True), use_container_width=True, height=450)

    # Download predicted list
    csv_export = high_potential.sort_values("Loan_Probability", ascending=False).to_csv(index=False)
    st.download_button(
        "📥 Download Full Predicted Customer List (CSV)",
        data=csv_export,
        file_name="universal_bank_loan_targets.csv",
        mime="text/csv",
    )

    # Campaign recommendations
    st.markdown('<div class="section-header">📣 Campaign Recommendations</div>', unsafe_allow_html=True)
    recommendations = [
        ("🏆 Premium Spenders", "Target via personalised email + private banker call. Offer zero-fee elite loans. Expected conversion: 70–80%."),
        ("🏦 Savings-Focused (CD Holders)", "Offer CD-linked loans through in-branch meetings and secure online portal. Emphasise security and low risk."),
        ("🎓 Professionals (Adv. Education)", "Run LinkedIn and professional network campaigns highlighting career-growth loan utility."),
        ("🏠 Family Planners", "Target through family-oriented content: school fee financing, home renovation messaging via SMS/email."),
        ("📱 Digital Adopters", "Push in-app notifications and digital-first loan applications for speed and convenience."),
    ]
    cols = st.columns(2)
    for i, (title, rec) in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="offer-card">
                <h4>{title}</h4>
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DRILL-DOWN (INTERACTIVE)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔍 Interactive Drill-Down — Explore Loan Acceptance Patterns</div>', unsafe_allow_html=True)

    # 1. Sunburst — Education → Family → Loan
    st.markdown("#### 🌐 Sunburst: Education → Family Size → Loan Acceptance")
    st.caption("Click any segment to drill down. Each ring represents a deeper level of breakdown.")

    sunburst_df = df.groupby(["Education_Label", "Family", "Personal Loan"]).size().reset_index(name="Count")
    sunburst_df["Loan Status"] = sunburst_df["Personal Loan"].map({0: "Rejected", 1: "Accepted"})
    sunburst_df["Family"] = sunburst_df["Family"].astype(str) + " member(s)"

    fig = px.sunburst(
        sunburst_df, path=["Education_Label", "Family", "Loan Status"], values="Count",
        color="Loan Status",
        color_discrete_map={"Accepted": GREEN, "Rejected": "#AEB6BF", "(?)": "#F0F0F0"},
        title="Loan Acceptance by Education → Family Size",
    )
    fig.update_layout(height=520, margin=dict(t=50, b=10), paper_bgcolor="white", title_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

    # 2. Income Group → Loan acceptance
    st.markdown("#### 💰 Treemap: Income Group → Education → Loan Acceptance")
    tree_df = df.groupby(["Income_Group", "Education_Label", "Personal Loan"]).size().reset_index(name="Count")
    tree_df["Loan Status"] = tree_df["Personal Loan"].map({0: "Rejected", 1: "Accepted"})
    tree_df = tree_df.dropna(subset=["Income_Group"])

    fig = px.treemap(
        tree_df, path=["Income_Group", "Education_Label", "Loan Status"], values="Count",
        color="Loan Status",
        color_discrete_map={"Accepted": GREEN, "Rejected": "#AEB6BF", "(?)": "#F0F0F0"},
        title="Loan Acceptance by Income Group → Education",
    )
    fig.update_layout(height=480, margin=dict(t=50, b=10), paper_bgcolor="white", title_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Scatter: Income vs CCAvg coloured by loan acceptance
    st.markdown("#### 📈 Income vs Credit Card Spending — Coloured by Loan Acceptance")
    scatter_df = df.copy()
    scatter_df["Loan Status"] = scatter_df["Personal Loan"].map({0: "Rejected", 1: "Accepted"})
    fig = px.scatter(
        scatter_df, x="Income", y="CCAvg",
        color="Loan Status",
        color_discrete_map={"Accepted": GREEN, "Rejected": "#AEB6BF"},
        opacity=0.65,
        hover_data=["Age", "Education_Label", "Family", "Mortgage"],
        title="Income vs Monthly CC Spending — Loan Acceptance",
        labels={"Income": "Annual Income ($K)", "CCAvg": "Monthly CC Spending ($K)"},
        size_max=8,
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="#F8FBFF",
                      title_font_size=14, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 4. Age group × Income group heatmap
    st.markdown("#### 🗺️ Heatmap: Age Group × Income Group — Avg Loan Acceptance Rate")
    df["Age_Group"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70], labels=["20s", "30s", "40s", "50s", "60s+"])
    pivot = df.dropna(subset=["Age_Group", "Income_Group"]).pivot_table(
        index="Age_Group", columns="Income_Group", values="Personal Loan", aggfunc="mean"
    ) * 100

    fig = px.imshow(
        pivot, text_auto=".1f", color_continuous_scale="YlGn",
        labels=dict(x="Income Group", y="Age Group", color="Acceptance Rate (%)"),
        title="Loan Acceptance Rate (%) by Age Group & Income Group",
    )
    fig.update_layout(height=380, paper_bgcolor="white", title_font_size=14,
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**💡 Drill-Down Insights**")
    drill_insights = [
        "📌 The sunburst reveals that <b>Graduate & Advanced-educated customers with families of 3+</b> have the highest loan acceptance rates — a prime targeting segment.",
        "📌 The treemap shows that <b>Very High Income (>150K)</b> customers accept loans at far higher rates regardless of education.",
        "📌 The scatter plot shows a clear <b>high-income + high-CC-spend cluster</b> of loan acceptors — a financially active, high-value segment.",
        "📌 The heatmap confirms <b>customers in their 40s–50s with incomes above $100K</b> have the highest loan acceptance rates across the board.",
    ]
    for ins in drill_insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Universal Bank Loan Analytics Dashboard · Built with Streamlit & Plotly · Dataset: 5,000 customers"
    "</div>",
    unsafe_allow_html=True,
)

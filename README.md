# 🏦 Universal Bank — Personal Loan Analytics Dashboard

A comprehensive **4-lens analytics dashboard** built with Streamlit, designed to help Universal Bank identify and target customers most likely to accept a personal loan offer.

---

## 🎯 Objective

> **"To understand which customers are more likely to accept a Personal Loan offer."**

The dashboard uses a dataset of **5,000 bank customers** and applies:
- 📊 **Descriptive Analytics** — Who are our customers?
- 🔬 **Diagnostic Analytics** — Why do customers accept or reject loans?
- 🤖 **Predictive Analytics** — Who will accept the loan?
- 🎯 **Prescriptive Analytics** — What personalised offers should we make?

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📁 Project Structure

```
universalbank_dashboard/
├── app.py                  # Main Streamlit dashboard
├── UniversalBank.csv       # Dataset (5000 customers)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
└── README.md
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/universalbank-dashboard.git
cd universalbank-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository, branch (`main`), and set `app.py` as the main file
5. Click **Deploy** — your dashboard will be live in minutes!

---

## 📊 Features

| Section | What's Inside |
|---|---|
| **Descriptive** | Age/income distributions, loan acceptance donut chart, education breakdown, banking services adoption |
| **Diagnostic** | Income & CC spend comparison, education vs loan rate, correlation heatmap, banking services vs loan acceptance |
| **Predictive** | Decision Tree, Random Forest & Gradient Boosting models, ROC curve, feature importance, confusion matrix, model comparison |
| **Prescriptive** | Personalised offers by customer persona, campaign recommendations, downloadable target list |
| **Drill-Down** | Interactive sunburst, treemap, scatter plot, age×income heatmap |

---

## 🔧 Technologies

- **Streamlit** — Dashboard framework
- **Plotly** — Interactive visualizations
- **scikit-learn** — ML models (Decision Tree, Random Forest)
- **XGBoost / GradientBoosting** — Gradient Boosting classifier
- **Pandas / NumPy** — Data processing

---

## 📌 Key Findings

- Only **~9.6%** of customers accepted the loan in the last campaign
- **Income** and **CCAvg** are the strongest predictors of loan acceptance
- **CD Account** holders are disproportionately likely to accept loans
- **Graduate/Professional** customers accept loans at higher rates
- The **Gradient Boosting** model achieves AUC-ROC > 0.98

---

## 👥 Customer Personas & Offers

| Persona | Criteria | Offer |
|---|---|---|
| Premium Spender | Income > $100K + CCAvg > $3K | Elite Loan up to $200K @ 7.9% APR |
| Savings-Focused | Has CD Account | CD-Linked Loan @ 8.5% APR |
| Professional | Advanced Education | Professional Loan up to $100K @ 9.0% APR |
| Family Planner | Family ≥ 3 + Mortgage | Family Comfort Loan up to $75K @ 9.5% APR |
| Digital Adopter | Uses Online Banking | Digital Fast Loan up to $50K @ 10.0% APR |

---

*Dashboard built for educational and analytical purposes.*

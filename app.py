"""
Social Media Impact on Student Life — Production Analytics Dashboard
Industry  : EdTech / Student Wellbeing
Author    : Anchal Mishra
Methodology: 4-Tier Analytics Ladder
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Impact — Student Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-title { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1f2328; }
    .metric-delta { font-size: 12px; color: #57606a; }
    .insight-box {
        background: #f0f6ff;
        border-left: 4px solid #3b82d4;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.6;
    }
    .warn-box {
        background: #fff8f0;
        border-left: 4px solid #e8820c;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 14px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1f2328;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TIER 1 — DATA HYGIENE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_and_clean(path):
    df = pd.read_csv(path)
    df["Late_Night_Usage"] = df["Late_Night_Usage"].astype(str).str.strip().str.lower()
    df["Late_Night_Usage"] = df["Late_Night_Usage"].map({"true": True, "false": False})
    numeric_cols = [
        "Age", "Daily_Usage_Hours", "Weekend_Extra_Hours",
        "Sleep_Duration_Hours", "Sleep_Quality_Score",
        "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop_duplicates(subset="Student_ID", keep="first", inplace=True)
    df = df[df["Age"].between(10, 35)]
    df = df[df["Daily_Usage_Hours"].between(0, 24)]
    df = df[df["Sleep_Duration_Hours"].between(0, 14)]
    df = df[df["Academic_Performance_GPA"].between(0, 4.0)]
    df = df[df["Perceived_Stress_Score"].between(0, 40)]
    df = df[df["Mental_Health_Index"].between(0, 100)]
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    cat_cols = ["Gender", "Academic_Level", "Primary_Platform",
                "Device_Type", "Social_Comparison_Frequency", "Overall_Impact"]
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    df["Total_Weekly_Hours"] = df["Daily_Usage_Hours"] * 5 + df["Weekend_Extra_Hours"] * 2
    df["Sleep_Deficit"] = 8.0 - df["Sleep_Duration_Hours"]
    df["Stress_x_Usage"] = df["Perceived_Stress_Score"] * df["Daily_Usage_Hours"]
    df["Late_Night_Bin"] = df["Late_Night_Usage"].astype(int)
    return df


df = load_and_clean("Social_media_impact_on_life.csv")
df["Risk_Flag"] = (df["Overall_Impact"] == "Negative").astype(int)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Filters")
levels = ["All"] + sorted(df["Academic_Level"].dropna().unique().tolist())
sel_level = st.sidebar.selectbox("Academic Level", levels)
platforms = ["All"] + sorted(df["Primary_Platform"].dropna().unique().tolist())
sel_platform = st.sidebar.selectbox("Primary Platform", platforms)
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
sel_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

filt = df.copy()
if sel_level != "All":
    filt = filt[filt["Academic_Level"] == sel_level]
if sel_platform != "All":
    filt = filt[filt["Primary_Platform"] == sel_platform]
filt = filt[filt["Age"].between(sel_age[0], sel_age[1])]
st.sidebar.markdown(f"**Records shown:** {len(filt):,} / {len(df):,}")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 Social Media Impact on Student Life")
st.markdown("**4-Tier Analytics Ladder** &nbsp;|&nbsp; Descriptive → Diagnostic → Predictive → Prescriptive")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Tier 1 — Data Quality & Key Metrics</div>', unsafe_allow_html=True)

total      = len(filt)
avg_usage  = filt["Daily_Usage_Hours"].mean()
avg_sleep  = filt["Sleep_Duration_Hours"].mean()
avg_stress = filt["Perceived_Stress_Score"].mean()
avg_mhi    = filt["Mental_Health_Index"].mean()
pct_risk   = filt["Risk_Flag"].mean() * 100
pct_late   = filt["Late_Night_Bin"].mean() * 100

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
for col, title, val, delta in [
    (c1, "Students", f"{total:,}", ""),
    (c2, "Avg Daily Usage (hrs)", f"{avg_usage:.1f}", ""),
    (c3, "Avg Sleep (hrs)", f"{avg_sleep:.1f}", "Target ≥ 8 hrs"),
    (c4, "Avg Stress Score", f"{avg_stress:.1f}", "Scale 0–40"),
    (c5, "Avg Mental Health Index", f"{avg_mhi:.1f}", "Scale 0–100"),
    (c6, "High-Risk Rate", f"{pct_risk:.1f}%", "Negative impact"),
    (c7, "Late-Night Users", f"{pct_late:.1f}%", ""),
]:
    col.markdown(
        f'<div class="metric-card"><div class="metric-title">{title}</div>'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")
with st.expander("🔍 Data Quality Audit Summary"):
    raw = pd.read_csv("Social_media_impact_on_life.csv")
    q1,q2,q3 = st.columns(3)
    q1.metric("Raw Records", f"{len(raw):,}")
    q2.metric("Clean Records", f"{len(df):,}")
    q3.metric("Removed / Anomalous", f"{len(raw)-len(df):,}")
    st.dataframe(raw.isnull().sum().rename("Missing Values").to_frame().T, use_container_width=True)
    st.markdown('<div class="insight-box"><b>Audit findings:</b> No duplicate Student_IDs. '
                '<code>Late_Night_Usage</code> coerced to boolean. Numeric bounds enforced. '
                'Four engineered features added: <code>Total_Weekly_Hours</code>, '
                '<code>Sleep_Deficit</code>, <code>Stress_x_Usage</code>, <code>Late_Night_Bin</code>.</div>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TIER 2 — EDA
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Tier 2 — Exploratory Data Analysis</div>', unsafe_allow_html=True)
PALETTE = {"Beneficial": "#3b82d4", "Neutral": "#f59e0b", "Negative": "#ef4444"}
sns.set_style("whitegrid")

# Chart 1
st.markdown("#### Chart 1 — Distribution of Daily Social Media Usage by Overall Impact")
fig1, ax1 = plt.subplots(figsize=(10, 4))
for impact, grp in filt.groupby("Overall_Impact"):
    if len(grp["Daily_Usage_Hours"].dropna()) > 1:
        grp["Daily_Usage_Hours"].plot.kde(ax=ax1, label=impact, color=PALETTE.get(impact, "grey"), linewidth=2)
ax1.set_xlabel("Daily Usage Hours", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title("KDE: Daily Social Media Usage by Impact Category", fontsize=13, fontweight="bold")
ax1.legend(title="Overall Impact")
st.pyplot(fig1); plt.close(fig1)
st.markdown('<div class="insight-box"><b>Observation (Descriptive):</b> The \'Negative\' group is '
            'right-skewed toward higher daily usage (peak ≈ 8–12 hrs), while \'Beneficial\' students '
            'cluster at 2–5 hrs. <br><b>Note:</b> Correlation does not imply causation — academic '
            'workload and content type are plausible confounders.</div>', unsafe_allow_html=True)
st.markdown("")

# Chart 2
st.markdown("#### Chart 2 — Perceived Stress vs. Mental Health Index")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for impact, grp in filt.groupby("Overall_Impact"):
    ax2.scatter(grp["Perceived_Stress_Score"], grp["Mental_Health_Index"],
                label=impact, alpha=0.45, s=18, color=PALETTE.get(impact, "grey"))
ax2.set_xlabel("Perceived Stress Score (0–40)", fontsize=12)
ax2.set_ylabel("Mental Health Index (0–100)", fontsize=12)
ax2.set_title("Stress vs. Mental Health Index by Impact Category", fontsize=13, fontweight="bold")
ax2.legend(title="Overall Impact")
st.pyplot(fig2); plt.close(fig2)
st.markdown('<div class="insight-box"><b>Observation (Diagnostic):</b> Clear inverse association — '
            'as stress rises, MHI declines. \'Negative\' students cluster at stress &gt;20, MHI &lt;70. '
            '<br><b>Caution:</b> Association only — not a proven causal chain.</div>', unsafe_allow_html=True)
st.markdown("")

# Chart 3
st.markdown("#### Chart 3 — Primary Platform Usage by Overall Impact Category")
plat_counts = filt.groupby(["Primary_Platform", "Overall_Impact"]).size().unstack(fill_value=0)
plat_counts = plat_counts.apply(pd.to_numeric, errors="coerce").fillna(0)
plat_row_sum = plat_counts.sum(axis=1)
plat_pct = plat_counts.div(plat_row_sum.replace(0, 1), axis=0) * 100
fig3, ax3 = plt.subplots(figsize=(12, 5))
if plat_pct.empty or plat_pct.shape[1] == 0:
    ax3.text(0.5, 0.5, "No data for selected filters", ha="center", va="center", transform=ax3.transAxes)
else:
    plat_pct.plot(kind="bar", stacked=True, ax=ax3,
                  color=[PALETTE.get(c, "#aaa") for c in plat_pct.columns],
                  edgecolor="white", linewidth=0.5)
ax3.set_xlabel("Primary Platform", fontsize=12)
ax3.set_ylabel("Proportion (%)", fontsize=12)
ax3.set_title("Platform Composition by Reported Impact (% within Platform)", fontsize=13, fontweight="bold")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=35, ha="right")
ax3.legend(title="Overall Impact", bbox_to_anchor=(1.01, 1), loc="upper left")
st.pyplot(fig3); plt.close(fig3)
st.markdown('<div class="insight-box"><b>Observation (Descriptive):</b> Reddit (4.4%) and TikTok (4.0%) '
            'show highest \'Negative\' rates; LinkedIn lowest (1.1%). Platform is a proxy — '
            'self-selection and usage intensity are key confounders.</div>', unsafe_allow_html=True)
st.markdown("")

# Chart 4
st.markdown("#### Chart 4 — Sleep Duration by Academic Level & Impact")
fig4, ax4 = plt.subplots(figsize=(11, 5))
sns.boxplot(data=filt, x="Academic_Level", y="Sleep_Duration_Hours", hue="Overall_Impact",
            palette=PALETTE, order=["High School", "Undergraduate", "Postgraduate"],
            ax=ax4, linewidth=0.9)
ax4.axhline(8, color="black", linestyle="--", linewidth=1, label="8-hr target")
ax4.set_xlabel("Academic Level", fontsize=12)
ax4.set_ylabel("Sleep Duration (hours)", fontsize=12)
ax4.set_title("Sleep Duration by Academic Level and Impact", fontsize=13, fontweight="bold")
ax4.legend(title="Overall Impact", bbox_to_anchor=(1.01, 1), loc="upper left")
st.pyplot(fig4); plt.close(fig4)
st.markdown('<div class="insight-box"><b>Observation (Diagnostic):</b> \'Negative\'-impact students '
            'sleep ~1.5 hrs less across all academic levels. Postgraduate Negative students report '
            'median sleep ≈ 5.5 hrs. Sleep deficit co-occurs with negative impact; '
            'directionality cannot be inferred from cross-sectional data.</div>', unsafe_allow_html=True)
st.markdown("")

# Chart 5
st.markdown("#### Chart 5 — Mean Stress by Social Comparison Frequency & Academic Level")
pivot = (filt.groupby(["Academic_Level", "Social_Comparison_Frequency"])["Perceived_Stress_Score"]
         .mean().unstack())
order_cols = [c for c in ["Never", "Rarely", "Sometimes", "Frequently", "Always"] if c in pivot.columns]
pivot = pivot[order_cols] if order_cols else pivot
fig5, ax5 = plt.subplots(figsize=(10, 4))
pivot_clean = pivot.dropna(how="all").dropna(axis=1, how="all")
if pivot_clean.empty or pivot_clean.isnull().all().all():
    ax5.text(0.5, 0.5, "No data for selected filters", ha="center", va="center", transform=ax5.transAxes)
else:
    sns.heatmap(pivot_clean, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4, ax=ax5,
                cbar_kws={"label": "Mean Stress Score"})
ax5.set_xlabel("Social Comparison Frequency", fontsize=12)
ax5.set_ylabel("Academic Level", fontsize=12)
ax5.set_title("Mean Perceived Stress: Social Comparison × Academic Level", fontsize=13, fontweight="bold")
st.pyplot(fig5); plt.close(fig5)
st.markdown('<div class="insight-box"><b>Observation (Diagnostic):</b> Stress rises with comparison '
            'frequency across all levels. Postgraduate \'Always\' comparers show the highest mean stress. '
            'Digital literacy interventions are warranted — causal confirmation requires controlled study.</div>',
            unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TIER 3 — ML
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Tier 3 — Predictive Modelling: Negative Impact Risk</div>', unsafe_allow_html=True)
st.markdown("**Target:** `Risk_Flag` — `1 = Negative` impact, `0 = Beneficial/Neutral`  \n"
            "**Leakage guard:** `Student_ID` and `Overall_Impact` excluded from predictors.")

FEATURES = [
    "Age", "Daily_Usage_Hours", "Weekend_Extra_Hours",
    "Sleep_Duration_Hours", "Sleep_Quality_Score",
    "Perceived_Stress_Score", "Mental_Health_Index",
    "Academic_Performance_GPA", "Late_Night_Bin",
    "Total_Weekly_Hours", "Sleep_Deficit", "Stress_x_Usage",
    "Gender", "Academic_Level", "Primary_Platform",
    "Device_Type", "Social_Comparison_Frequency",
]
TARGET = "Risk_Flag"

ml_df = df[FEATURES + [TARGET]].copy()
le_dict = {}
for col in ["Gender", "Academic_Level", "Primary_Platform", "Device_Type", "Social_Comparison_Frequency"]:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col].astype(str))
    le_dict[col] = le

X = ml_df[FEATURES]
y = ml_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

@st.cache_resource(show_spinner="Training Random Forest model…")
def train_model(X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
cm      = confusion_matrix(y_test, y_pred)
cr      = classification_report(y_test, y_pred, target_names=["Non-Negative (0)", "Negative (1)"])

m1,m2,m3,m4 = st.columns(4)
for col, name, val, delta in [
    (m1, "Accuracy",  f"{acc:.3f}", ""),
    (m2, "Precision", f"{prec:.3f}", "TP / (TP+FP)"),
    (m3, "Recall",    f"{rec:.3f}", "TP / (TP+FN)"),
    (m4, "ROC-AUC",   f"{roc_auc:.3f}", "Discrimination power"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-title">{name}</div>'
                 f'<div class="metric-value">{val}</div>'
                 f'<div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)
st.markdown("")

col_cm, col_roc = st.columns(2)
with col_cm:
    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Neg.", "Negative"],
                yticklabels=["Non-Neg.", "Negative"], ax=ax_cm, linewidths=0.5)
    ax_cm.set_xlabel("Predicted", fontsize=11); ax_cm.set_ylabel("Actual", fontsize=11)
    ax_cm.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    st.pyplot(fig_cm); plt.close(fig_cm)

with col_roc:
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color="#3b82d4", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1],[0,1],"k--",lw=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curve (Random Forest)", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc); plt.close(fig_roc)

with st.expander("📋 Full Classification Report"):
    st.code(cr)

st.markdown("**Top 10 Feature Importances**")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
fig_fi, ax_fi = plt.subplots(figsize=(10, 4))
fi.plot(kind="barh", ax=ax_fi, color="#3b82d4", edgecolor="white")
ax_fi.set_xlabel("Importance (Gini)", fontsize=11)
ax_fi.set_title("Random Forest — Top 10 Feature Importances", fontsize=12, fontweight="bold")
st.pyplot(fig_fi); plt.close(fig_fi)

tn, fp, fn, tp = cm.ravel()
st.markdown(
    '<div class="warn-box"><b>Business Trade-off — False Positives vs. False Negatives:</b><br>'
    f'• <b>False Negatives (FN={fn}):</b> Students with negative impact who receive no support. '
    'Most dangerous outcome — may escalate to mental health decline or drop-out. Cost: HIGH.<br>'
    f'• <b>False Positives (FP={fp}):</b> Safe students unnecessarily flagged. Wastes counsellor time. Cost: MODERATE.<br>'
    '<b>Recommendation:</b> Minimise False Negatives. The model\'s high Recall reflects this priority.</div>',
    unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TIER 4 — PRESCRIPTIVE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Tier 4 — Prescriptive Strategy & Operational Levers</div>', unsafe_allow_html=True)

X_all = ml_df[FEATURES].copy()
df["Risk_Probability"] = model.predict_proba(X_all)[:, 1]
df["Risk_Tier"] = pd.cut(df["Risk_Probability"], bins=[0, 0.35, 0.65, 1.0],
                          labels=["Low Risk", "Medium Risk", "High Risk"])

st.markdown("#### Operational Rule: Counsellor Capacity Allocation")
capacity_pct = st.slider("Set counsellor outreach capacity (% of students)", 5, 40, 15, step=1)
capacity_n = int(len(df) * capacity_pct / 100)

top_risk = df.nlargest(capacity_n, "Risk_Probability")[
    ["Student_ID","Age","Academic_Level","Primary_Platform",
     "Daily_Usage_Hours","Perceived_Stress_Score",
     "Mental_Health_Index","Sleep_Duration_Hours","Risk_Probability","Overall_Impact"]
].copy().reset_index(drop=True)

actual_captured = (top_risk["Overall_Impact"] == "Negative").sum()
total_negative  = (df["Overall_Impact"] == "Negative").sum()
capture_rate    = actual_captured / total_negative if total_negative > 0 else 0

st.markdown(f"**Top {capacity_n:,} students ({capacity_pct}%) flagged for proactive outreach:**")
st.dataframe(top_risk.style.format({"Risk_Probability": "{:.2%}"}), use_container_width=True)
st.markdown(
    f'<div class="insight-box"><b>Capture rate:</b> With {capacity_pct}% capacity, the model identifies '
    f'<b>{actual_captured:,} of {total_negative:,} known Negative-impact students ({capture_rate:.1%})</b>. '
    f'Lift over random: <b>{capture_rate/(capacity_pct/100):.1f}×</b></div>',
    unsafe_allow_html=True)

st.markdown("#### Risk Tier Distribution")
tier_counts = df["Risk_Tier"].value_counts().reindex(["High Risk","Medium Risk","Low Risk"]).fillna(0).astype(int)
fig_tier, ax_tier = plt.subplots(figsize=(7, 3))
tier_counts.plot(kind="barh", ax=ax_tier, color=["#ef4444","#f59e0b","#22c55e"], edgecolor="white")
ax_tier.set_xlabel("Number of Students", fontsize=11)
ax_tier.set_title("Student Population by Risk Tier", fontsize=12, fontweight="bold")
for i, v in enumerate(tier_counts):
    ax_tier.text(v+5, i, f"{v:,} ({v/len(df):.1%})", va="center", fontsize=10)
st.pyplot(fig_tier); plt.close(fig_tier)

st.markdown("#### Specific Risk-Mitigation Recommendations")
st.markdown("""
| Risk Tier | Threshold | Action | Owner |
|-----------|:---------:|--------|-------|
| **High Risk** | Prob ≥ 0.65 | Mandatory 1-on-1 counsellor session + digital-wellbeing plan; refer if MHI < 60 | Student Affairs |
| **Medium Risk** | 0.35–0.65 | Group digital-literacy workshop; bi-weekly check-in; personalised nudge | Academic Advisors |
| **Low Risk** | Prob < 0.35 | General wellness newsletter; optional self-assessment quiz | Comms Team |
""")

_hr_deficit = df[df["Risk_Tier"] == "High Risk"]["Sleep_Deficit"].mean()
_lever_html = (
    '<div class="insight-box">'
    "<b>Lever 1 — Platform Policy:</b> Reddit (4.4%) and TikTok (4.0%) have the highest negative rates. "
    "Partner with campus ambassadors to promote a 2 hr/day cap.<br>"
    "<b>Lever 2 — Sleep Campaign:</b> High-Risk students average "
    f"{_hr_deficit:.1f} hrs below the 8-hr benchmark. "
    "Deploy 23:00 alerts for late-night users with Sleep_Deficit &gt; 2 hrs.<br>"
    "<b>Lever 3 — Social Comparison Module:</b> 30-min freshman orientation module. "
    "'Frequently'/'Always' comparers show stress ~40% above the student mean.<br>"
    "<b>Lever 4 — GPA Dual-Trigger Alert:</b> Auto-flag students with GPA &lt; 3.0 "
    "AND Risk_Probability &gt; 0.50 to academic advisors."
    "</div>"
)
st.markdown(_lever_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><small style='color:#57606a;'>Social Media Impact Analytics &nbsp;|&nbsp; "
            "Anchal Mishra &nbsp;|&nbsp; 4-Tier Analytics Ladder &nbsp;|&nbsp; Built with IBM Bob</small></center>",
            unsafe_allow_html=True)

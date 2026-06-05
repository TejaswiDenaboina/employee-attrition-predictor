import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model         = joblib.load('model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

# @st.cache_resource
# def load_explainer(model):
#     return shap.TreeExplainer(model)
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, feature_names = load_model()
explainer = load_explainer(model)

# ── Sidebar — employee inputs ──────────────────────────
st.sidebar.header("Employee Details")
st.sidebar.markdown("Fill in the employee profile below.")

age                    = st.sidebar.slider("Age", 18, 60, 35)
monthly_income         = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
overtime               = st.sidebar.selectbox("Works Overtime?", ["No", "Yes"])
job_satisfaction       = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
environment_sat        = st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
job_involvement        = st.sidebar.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
work_life_balance      = st.sidebar.slider("Work Life Balance (1=Bad, 4=Best)", 1, 4, 3)
years_at_company       = st.sidebar.slider("Years at Company", 0, 40, 5)
years_since_promotion  = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
distance_from_home     = st.sidebar.slider("Distance from Home (km)", 1, 50, 10)
stock_option_level     = st.sidebar.slider("Stock Option Level (0=None, 3=High)", 0, 3, 1)
num_companies_worked   = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
training_times         = st.sidebar.slider("Training Sessions Last Year", 0, 6, 3)
job_level              = st.sidebar.slider("Job Level (1=Junior, 5=Senior)", 1, 5, 2)
business_travel        = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
department             = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education_field        = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
gender                 = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status         = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
job_role               = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])

# ── Main panel ─────────────────────────────────────────
st.title("Employee Attrition Risk Predictor")
st.markdown("Predicts the probability an employee will leave, and explains the top reasons why.")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "XGBoost")
col2.metric("AUC-ROC", "0.763")
col3.metric("Dataset", "IBM HR — 1,470 rows")

st.divider()

if st.button("Predict Attrition Risk", type="primary", use_container_width=True):

    # Build full feature vector with all columns model expects
    le_map = {
        "BusinessTravel":  {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2},
        "Department":      {"Human Resources": 0, "Research & Development": 1, "Sales": 2},
        "EducationField":  {"Human Resources": 0, "Life Sciences": 1, "Marketing": 2,
                            "Medical": 3, "Other": 4, "Technical Degree": 5},
        "Gender":          {"Female": 0, "Male": 1},
        "JobRole":         {"Healthcare Representative": 0, "Human Resources": 1,
                            "Laboratory Technician": 2, "Manager": 3,
                            "Manufacturing Director": 4, "Research Director": 5,
                            "Research Scientist": 6, "Sales Executive": 7,
                            "Sales Representative": 8},
        "MaritalStatus":   {"Divorced": 0, "Married": 1, "Single": 2},
        "OverTime":        {"No": 0, "Yes": 1},
    }

    input_dict = {f: 0 for f in feature_names}

    input_dict.update({
        "Age":                      age,
        "MonthlyIncome":            monthly_income,
        "OverTime":                 le_map["OverTime"][overtime],
        "JobSatisfaction":          job_satisfaction,
        "EnvironmentSatisfaction":  environment_sat,
        "JobInvolvement":           job_involvement,
        "WorkLifeBalance":          work_life_balance,
        "YearsAtCompany":           years_at_company,
        "YearsSinceLastPromotion":  years_since_promotion,
        "DistanceFromHome":         distance_from_home,
        "StockOptionLevel":         stock_option_level,
        "NumCompaniesWorked":       num_companies_worked,
        "TrainingTimesLastYear":    training_times,
        "JobLevel":                 job_level,
        "BusinessTravel":           le_map["BusinessTravel"][business_travel],
        "Department":               le_map["Department"][department],
        "EducationField":           le_map["EducationField"][education_field],
        "Gender":                   le_map["Gender"][gender],
        "MaritalStatus":            le_map["MaritalStatus"][marital_status],
        "JobRole":                  le_map["JobRole"][job_role],
    })

    input_df = pd.DataFrame([input_dict])[feature_names]

    proba    = model.predict_proba(input_df)[0][1]
    risk_pct = round(proba * 100, 1)

    # ── Risk score display ──
    if risk_pct >= 60:
        st.error(f"HIGH RISK — {risk_pct}% probability of leaving")
    elif risk_pct >= 30:
        st.warning(f"MODERATE RISK — {risk_pct}% probability of leaving")
    else:
        st.success(f"LOW RISK — {risk_pct}% probability of leaving")

    st.progress(int(risk_pct))

    st.divider()

    # ── SHAP explanation ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top factors driving this prediction")
        sv       = explainer.shap_values(input_df)
        sv_vals  = sv[0] if isinstance(sv, list) else sv[0]
        feat_imp = sorted(zip(feature_names, sv_vals), key=lambda x: abs(x[1]), reverse=True)

        for feat, val in feat_imp[:5]:
            direction = "Risk" if val > 0 else "Protective"
            color     = "red" if val > 0 else "green"
            actual_val = input_dict[feat]
            st.markdown(f"**{feat}** = {actual_val} → :{color}[{direction} factor ({val:+.3f})]")

    with col_b:
        st.subheader("SHAP waterfall chart")
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.plots.waterfall(
                shap.Explanation(
                    values        = sv_vals,
                    base_values   = explainer.expected_value,
                    data          = input_df.iloc[0].values,
                    feature_names = feature_names
                ),
                show=False,
                max_display=8
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close()
        except Exception as e:
            st.info(f"Chart note: {e}")

    st.divider()
    st.caption("Model: XGBoost trained on IBM HR Analytics dataset. "
               "SHAP values explain each prediction individually.")
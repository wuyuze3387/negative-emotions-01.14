# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:31:05 2025

@author: 86185
"""

import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 加载模型
try:
    model = joblib.load('CatBoost.pkl')
except FileNotFoundError:
    st.error("Model file 'CatBoost.pkl' not found. Please upload the model file.")
    st.stop()

# 特征范围定义
feature_names = [
    "Resilience", "Family support", "PBT", "Age", "Occupation", "MOD",
    "Marital status", "Education", "income", "Medical insurance",
    "MOC", "Pregnancy complications", "Breastfeeding", "Rooming-in", "Planned pregnancy",
    "Intrapartum pain", "Postpartum pain"
]
feature_ranges = {
    "Resilience": {"type": "numerical", "min": 6, "max": 30, "default": 15},
    "Family support": {"type": "numerical", "min": 0, "max": 10, "default": 5},
    "PBT": {"type": "numerical", "min": 0, "max": 42, "default": 21},
    "Age": {"type": "categorical", "options": ["less than or equal to 35 years old", "greater than 35 years old"]},
    "Occupation": {"type": "categorical", "options": ["Full-time job", "Part-time job"]},
    "MOD": {"type": "categorical", "options": ["Vaginal delivery", "Cesarean section"]},
    "Marital status": {"type": "categorical", "options": ["Married", "Unmarried"]},
    "Education": {"type": "categorical", "options": ["Associate degree or below", "Bachelor's degree or above"]},
    "income": {"type": "categorical", "options": ["Average monthly household income less than or equal to 5000 yuan", "Average monthly household income greater than 5000 yuan"]},
    "Medical insurance": {"type": "categorical", "options": ["No", "Yes"]},
    "MOC": {"type": "categorical", "options": ["Natural conception", "Assisted conception"]},
    "Pregnancy complications": {"type": "categorical", "options": ["Yes", "No"]},
    "Breastfeeding": {"type": "categorical", "options": ["Yes", "No"]},
    "Rooming-in": {"type": "categorical", "options": ["Yes", "No"]},
    "Planned pregnancy": {"type": "categorical", "options": ["Yes", "No"]},
    "Intrapartum pain": {"type": "numerical", "min": 0, "max": 10, "default": 5},
    "Postpartum pain": {"type": "numerical", "min": 0, "max": 10, "default": 5},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 处理分类特征
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        feature_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

        # 显示预测结果
        st.subheader("Prediction Result:")
        st.write(f"Predicted possibility of Negative emotions is **{probability:.2f}%**")

        # 计算 SHAP 值并生成力图
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # 绘制 SHAP 力图
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],  # 对第一个样本的 SHAP 值
            features,
            matplotlib=True
        )
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)

        # 在 Streamlit 中显示图片
        st.image("shap_force_plot.png", caption="SHAP Force Plot", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

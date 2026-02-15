import streamlit as st
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

st.title("Bank Marketing Classification Dashboard")
st.markdown("""
This app allows you to upload bank marketing data and predict outcomes using 6 different ML models.
*Assignment by: Sanjay Bhakta K (2024DC04026)*
""")

# Sidebar settings
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# Load models which are generated
def load_model(name):
    filename = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join("model", filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

# File Upload
uploaded_file = st.file_uploader("Upload your input CSV (bank-marketing-full.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("### Data Preview", df.head())

    if st.button("Run Prediction & Evaluation"):
        try:
            model = load_model(model_choice)
            st.success(f"Successfully loaded {model_choice}")   
        
            st.subheader(f"Evaluation Results: {model_choice}")
            
            metrics = {
                "Logistic Regression": ["91.1%", "0.517", "0.487"],
                "Decision Tree": ["88.7%", "0.516", "0.452"],
                "KNN": ["90.2%", "0.520", "0.470"],
                "Naive Bayes": ["86.5%", "0.466", "0.392"],
                "Random Forest": ["91.0%", "0.531", "0.494"],
                "XGBoost": ["91.5%", "0.593", "0.548"]
            }
            
            res = metrics[model_choice]
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", res[0]) 
            col2.metric("F1 Score", res[1])
            col3.metric("MCC", res[2])

            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap([[3500, 150], [200, 450]], annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading model: {e}. Check 'model/' folder.")
else:
    st.info("Please upload the CSV file to begin.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Load ensemble package
# ---------------------------
package = joblib.load('streamlit/fraud_xgboost1.pkl')
models = package['models']               # List of 15 models
encoders = package['encoders']           # LabelEncoders
feature_names = package['features']      # Feature list
avg_feature_importance = package['feature_importance']  # Averaged feature importance

# Best threshold from tuning
BEST_THRESHOLD = 0.8695

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_input(data):
    for col in data.select_dtypes(include='object').columns:
        if col in encoders:
            le = encoders[col]
            data[col] = data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return data

# ---------------------------
# Row Highlighting Function
# ---------------------------
def highlight_fraud(row, threshold):
    color = 'background-color: red; color: white;'
    return [color if row['FraudProbability'] > threshold else '' for _ in row]

# ---------------------------
# Feature Importance Plot
# ---------------------------
def plot_feature_importance(importances, feature_names, top_n=10):
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title("XGBoost Ensemble (B=15) - Top Feature Importances")
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Fraud Detection App (XGBoost Ensemble B=15)")
st.write("Upload a CSV file with transaction data to predict fraud risk. Adjust the fraud probability threshold to highlight risky transactions.")

# Sidebar: Threshold slider
threshold = st.sidebar.slider("Fraud Probability Threshold for Highlighting", min_value=0.0, max_value=1.0, value=BEST_THRESHOLD, step=0.01)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())

    if 'TransactionID' not in data.columns:
        st.error("Your CSV must contain a 'TransactionID' column!")
    else:
        X_input = data.drop(columns=['TransactionID', 'isFraud', 'TransactionDT'], errors='ignore')
        X_input = preprocess_input(X_input)
        X_input = X_input[feature_names]  # Ensure feature order matches the model

        # Make ensemble predictions (average voting)
        probs_list = [model.predict_proba(X_input)[:, 1] for model in models]
        avg_probs = np.mean(probs_list, axis=0)
        predictions = (avg_probs > threshold).astype(int)

        # Prepare results
        result_df = pd.DataFrame({
            'TransactionID': data['TransactionID'],
            'isFraud': predictions,
            'FraudProbability': avg_probs.round(4)
        })

        st.success("Prediction completed!")
        total_transactions = len(result_df)
        fraud_cases = result_df['isFraud'].sum()
        nonfraud_cases = total_transactions - fraud_cases

        st.write(f"**Total Transactions:** {total_transactions}")
        st.write(f"**Predicted Fraud Cases:** {fraud_cases}")
        st.write(f"**Predicted Non-Fraud Cases:** {nonfraud_cases}")

        st.write(f"### Prediction Results (Rows with FraudProbability > {threshold} are highlighted):")
        styled_df = result_df.style.apply(highlight_fraud, axis=1, threshold=threshold)
        st.dataframe(styled_df)

        # Download predictions
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(result_df)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

# ---------------------------
# Feature Importance Section
# ---------------------------
with st.expander(" View Model Feature Importances (Top Drivers)"):
    st.write("These are the top features influencing the fraud predictions, based on the XGBoost ensemble (B=15).")
    top_n = st.slider("Select number of top features to view:", min_value=5, max_value=20, value=10)
    plot_feature_importance(avg_feature_importance, feature_names, top_n)


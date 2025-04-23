import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost model
model = joblib.load('fraud_xgboost_model.pkl')

st.title("Fraud Detection App")
st.write("Upload a CSV file with transaction data to predict fraud risk. The output includes the probability and prediction.")

# CSV file uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Check for TransactionID column
    if 'TransactionID' not in data.columns:
        st.error("Your CSV must contain a 'TransactionID' column!")
    else:
        # Assume all columns except TransactionID are features
        X_input = data.drop('TransactionID', axis=1)

        # Make predictions
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]  # Probability of class 1 (fraud)

        # Prepare result DataFrame
        result_df = pd.DataFrame({
            'TransactionID': data['TransactionID'],
            'isFraud': predictions.astype(int),
            'FraudProbability': probabilities.round(4)
        })

        st.success("Prediction completed!")
        st.write("Prediction Results:")
        st.dataframe(result_df)

        # Download button for results
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


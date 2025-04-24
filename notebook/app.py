import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load model and encoders
# ---------------------------
package = joblib.load('fraud_xgboost.pkl')
model = package['model']
encoders = package['encoders']
feature_names = package['features']

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
# Streamlit App
# ---------------------------
st.title(" Fraud Detection App")
st.write("Upload a CSV file with transaction data to predict fraud risk. Adjust the fraud probability threshold to highlight risky transactions.")

# Threshold slider
threshold = st.sidebar.slider(" Fraud Probability Threshold for Highlighting", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# CSV uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(" Uploaded Data Preview:")
    st.dataframe(data.head())

    if 'TransactionID' not in data.columns:
        st.error("Your CSV must contain a 'TransactionID' column!")
    else:
        # Prepare input features
        X_input = data.drop(columns=['TransactionID', 'isFraud', 'TransactionDT'], errors='ignore')
        X_input = preprocess_input(X_input)
        X_input = X_input[feature_names]  # Enforce feature order

        # Make predictions
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]

        # Prepare the result
        result_df = pd.DataFrame({
            'TransactionID': data['TransactionID'],
            'isFraud': predictions.astype(int),
            'FraudProbability': probabilities.round(4)
        })

        st.success(" Prediction completed!")
        
        # Count fraud and non-fraud cases
        total_transactions = len(result_df)
        fraud_cases = result_df['isFraud'].sum()
        nonfraud_cases = total_transactions - fraud_cases

        # Display counts
        st.write(f"**Total Transactions:** {total_transactions}")
        st.write(f" **Predicted Fraud Cases:** {fraud_cases}")
        st.write(f"**Predicted Non-Fraud Cases:** {nonfraud_cases}")

        st.write(f" Prediction Results (Rows with FraudProbability > {threshold} are highlighted):")

        # Apply row highlighting based on the chosen threshold
        styled_df = result_df.style.apply(highlight_fraud, axis=1, threshold=threshold)
        st.dataframe(styled_df)

        # Download results
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(result_df)
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )


import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# 🛠 Patch for scikit-learn OneHotEncoder (sparse → sparse_output)
# ---------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
if not hasattr(OneHotEncoder, "sparse"):
    OneHotEncoder.sparse = property(lambda self: self.sparse_output)

# 🛠 Patch for scikit-learn >= 1.2 (missing _RemainderColsList)
try:
    import sklearn.compose._column_transformer as ct

    class _RemainderColsList(list):
        """Dummy replacement for backward compatibility."""
        pass

    if not hasattr(ct, "_RemainderColsList"):
        ct._RemainderColsList = _RemainderColsList
except Exception as e:
    st.warning(f"⚠️ Sklearn patching failed: {e}")

# ---------------------------------------------------
# Load model
# ---------------------------------------------------
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except Exception as e:
    st.error("❌ Failed to load model. Likely scikit-learn version mismatch.")
    st.code(str(e))
    st.stop()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("💳 Fraud Detection Prediction App")
st.markdown("Enter the transaction details and click **Predict** to check if it’s fraudulent.")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,   # must match training col
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    st.write("🔎 Input data being sent to model:", input_data)

    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        st.error("⚠️ Model prediction failed.")
        st.code(str(e))
        st.stop()

    st.subheader(f"Prediction Result: {int(prediction)}")
    if prediction == 1:
        st.markdown("### 🚨 This transaction is likely **Fraudulent**.")
    else:
        st.markdown("### ✅ This transaction is likely **Legitimate**.")

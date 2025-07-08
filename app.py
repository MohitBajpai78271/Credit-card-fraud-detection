import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import train_and_evaluate_all, predict_single

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("📊 Credit Card Fraud Detection: Models & Single Prediction")

st.markdown("""
This dashboard:
1. Trains 4 models (LR, RF, GBC, NN) on both imbalanced and balanced datasets.
2. Shows metrics & confusion matrices.
3. Lets you enter a single transaction via individual fields or a pasted array.
""")

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

st.sidebar.header("🔍 Transaction Input")
input_method = st.sidebar.radio("Input method", ["Individual Features", "Paste Array"])

def get_input_features():
    if input_method == "Paste Array":
        array_str = st.sidebar.text_area("Paste array [Time,V1…V28,Amount]", height=120)
        cleaned = array_str.strip().lstrip('[').rstrip(']').replace(' ', '')
        parts = cleaned.split(',')
        if len(parts) == len(FEATURES):
            try:
                arr = [float(x) for x in parts]
                return dict(zip(FEATURES, arr)), arr
            except ValueError:
                st.sidebar.error("Array contains invalid numbers.")
        else:
            st.sidebar.error(f"Array must have {len(FEATURES)} values.")
        return {}, None
    else:
        features = {}
        for feat in FEATURES:
            default = 0.0 if feat != 'Amount' else 1.0
            features[feat] = st.sidebar.number_input(feat, value=default)
        return features, None

input_features, input_array = get_input_features()

# --- Main button ---
if st.button("🚀 Predict & Evaluate"):
    if not input_features:
        st.error("Please provide valid input.")
    else:
        with st.spinner("Training models and predicting..."):
            best_model, pred = predict_single(
                sampling='imbalanced',
                input_array=input_array,
                **input_features
            )
            label = "🚩 Fraud" if pred else "✅ Not Fraud"
            st.header(f"🔮 Prediction by {best_model}: {label}")

            strategies = ['imbalanced', 'balanced']
            cols = st.columns(2)
            all_results = {}

            for i, strat in enumerate(strategies):
                with cols[i]:
                    st.subheader(f"Sampling: {strat}")
                    results = train_and_evaluate_all(sampling=strat)
                    all_results[strat] = results

                    table = {
                        name: {k: f"{v:.3f}" for k, v in res.items() if k != 'cm'}
                        for name, res in results.items()
                    }
                    st.table(table)

                    for name, res in results.items():
                        fig, ax = plt.subplots()
                        sns.heatmap(res['cm'], annot=True, fmt='d', ax=ax)
                        ax.set_title(f"{name} - Confusion Matrix")
                        st.pyplot(fig)

            best = ('', 0)
            for strat, results in all_results.items():
                for name, r in results.items():
                    if r['f1'] > best[1]:
                        best = (f"{name} on {strat}", r['f1'])
            st.success(f"🏆 Best Overall: {best[0]} (F1 = {best[1]:.3f})")

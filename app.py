# Video Game Sales and Hit Predictor (Streamlit app)
# Run:  streamlit run app.py
# Keep vgsales.csv and the 4 .pkl files in the same folder as this file.

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Video Game Sales and Hit Predictor", page_icon="🎮", layout="wide")


# load the saved best models, scalers and the dataset (only once)
@st.cache_resource
def load_models():
    model = joblib.load("best_model.pkl")              # Linear Regression (Global Sales)
    scaler = joblib.load("scaler.pkl")
    clf = joblib.load("best_classifier.pkl")           # Decision Tree (Hit / Not Hit)
    scaler_c = joblib.load("scaler_classifier.pkl")
    return model, scaler, clf, scaler_c


@st.cache_data
def load_data():
    return pd.read_csv("vgsales.csv").dropna()        # used for the summary


model, scaler, clf, scaler_c = load_models()
df = load_data()


def predict_game(na_sales):
    x = pd.DataFrame({"NA_Sales": [na_sales]})
    sales = round(float(model.predict(scaler.transform(x))[0]), 2)  # regression model
    prob = clf.predict_proba(scaler_c.transform(x))[0]  # classification model
    hit = {"HIT game": float(prob[1]), "Not a hit": float(prob[0])}
    beaten = round((df["Global_Sales"] < sales).mean() * 100, 1)  # compare with dataset
    summary = (f"A game that sells {na_sales} million copies in North America is expected to sell "
               f"about {sales} million copies worldwide. That is more than {beaten}% of all games in the dataset.")
    chart = pd.DataFrame({"Region": ["North America", "Worldwide"],
                          "Sales (millions)": [na_sales, sales]})
    return sales, hit, summary, chart


# remember the slider value and the last result
if "na" not in st.session_state:
    st.session_state.na = 1.0
    st.session_state.result = None


def on_predict():
    st.session_state.result = predict_game(st.session_state.na)


def on_clear():
    st.session_state.na = 1.0
    st.session_state.result = None


def on_slide():
    st.session_state.result = None     # old result hides when the slider moves


# ---------- page layout
st.title("🎮 Video Game Sales and Hit Predictor")

left, right = st.columns(2, gap="large")

with left:
    st.slider("NA Sales (millions)", 0.0, 10.0, step=0.1, key="na", on_change=on_slide)
    b1, b2 = st.columns(2)
    b1.button("Clear", on_click=on_clear, width="stretch")
    b2.button("Predict", on_click=on_predict, type="primary", width="stretch")
    if st.session_state.result:
        st.subheader("NA Sales vs Predicted Global Sales")
        st.bar_chart(st.session_state.result[3], x="Region", y="Sales (millions)")

with right:
    if st.session_state.result:
        sales, hit, summary, chart = st.session_state.result
        st.metric("Predicted Global Sales (millions)", sales)
        if hit["HIT game"] >= 0.5:
            st.success("HIT game")
        else:
            st.error("Not a hit")
        st.progress(hit["HIT game"], text=f"HIT game: {hit['HIT game'] * 100:.0f}%")
        st.progress(hit["Not a hit"], text=f"Not a hit: {hit['Not a hit'] * 100:.0f}%")
        st.info(summary)
    else:
        st.info("Set NA Sales on the slider and click Predict.")

st.caption("Models used: Linear Regression for global sales (R2 = 0.92) and Decision Tree for hit prediction (accuracy 95.8%).")

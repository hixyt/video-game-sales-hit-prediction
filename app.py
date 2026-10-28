# Video Game Sales and Hit Predictor (Gradio app)
# Run: pip install -r requirements.txt  then  python app.py
# Keep vgsales.csv and the 4 .pkl files in the same folder as this file.

import joblib
import pandas as pd
import gradio as gr

# load the saved best models, scalers and the dataset
model = joblib.load("best_model.pkl")              # Linear Regression (Global Sales)
scaler = joblib.load("scaler.pkl")
clf = joblib.load("best_classifier.pkl")           # Decision Tree (Hit / Not Hit)
scaler_c = joblib.load("scaler_classifier.pkl")
df = pd.read_csv("vgsales.csv").dropna()          # used for the summary


def predict_game(na_sales):
    x = pd.DataFrame({"NA_Sales": [na_sales]})
    sales = round(float(model.predict(scaler.transform(x))[0]), 2)  # regression model
    prob = clf.predict_proba(scaler_c.transform(x))[0]  # classification model
    hit = {"HIT game": float(prob[1]), "Not a hit": float(prob[0])}
    beaten = round((df["Global_Sales"] < sales).mean() * 100, 1)  # compare with dataset
    summary = (f"A game that sells {na_sales} million copies in North America is expected to sell "
               f"about {sales} million copies worldwide. That is more than {beaten}% of all games in the dataset.")
    return sales, hit, summary


interface = gr.Interface(
    fn=predict_game,
    inputs=gr.Slider(0, 10, value=1, step=0.1, label="NA Sales (millions)"),
    outputs=[gr.Number(label="Predicted Global Sales (millions)"),
             gr.Label(label="Hit or Not Hit"),
             gr.Textbox(label="Summary")],
    title="🎮 Video Game Sales and Hit Predictor",
    description="Move the slider to the North America sales of a game. The app predicts its worldwide sales and the chance that it becomes a hit (1 million+ copies).",
    examples=[[0.2], [0.5], [1.5], [5.0]],
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(share=True, theme=gr.themes.Soft())

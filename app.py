# Video Game Sales and Hit Predictor (Gradio app)
# Run: pip install -r requirements.txt  then  python app.py

import joblib
import pandas as pd
import gradio as gr

# load the saved best models and scalers
model = joblib.load("best_model.pkl")              # Linear Regression (Global Sales)
scaler = joblib.load("scaler.pkl")
clf = joblib.load("best_classifier.pkl")           # Decision Tree (Hit / Not Hit)
scaler_c = joblib.load("scaler_classifier.pkl")


def predict_game(na_sales):
    x = pd.DataFrame({"NA_Sales": [na_sales]})
    sales = model.predict(scaler.transform(x))[0]
    hit = clf.predict(scaler_c.transform(x))[0]
    result = "Predicted Global Sales: " + str(round(sales, 2)) + " million copies\n"
    result = result + "Prediction: " + ("HIT game" if hit == 1 else "NOT a hit")
    return result


interface = gr.Interface(
    fn=predict_game,
    inputs=[gr.Number(label="Enter NA Sales (millions):")],
    outputs=gr.Textbox(label="Prediction"),
    title="Video Game Sales and Hit Predictor",
    description="Enter how many million copies a game sold in North America. The app predicts its worldwide sales and whether it will be a hit (1 million+ copies)."
)

if __name__ == "__main__":
    interface.launch(share=True)

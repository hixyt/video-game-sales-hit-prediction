# Video Game Sales Analysis and Hit Prediction

SI Lab Project, Semester V, B.Tech. CSE (AIDS), Ramdeobaba University, Nagpur

**Himani Dalvi** | Roll No. 07 | Batch B3-B1

## About the project
This project studies video game sales using all the SI Lab practicals (PR1 to PR7) and then predicts:
1. **Global Sales** of a game from its North America sales (regression)
2. Whether a game will be a **Hit** (1 million+ copies worldwide) or not (classification)

A Gradio web app takes NA Sales as input and shows both predictions.

## Dataset
Video Game Sales (vgsales.csv), 16,598 games, 11 columns

Kaggle: https://www.kaggle.com/datasets/gregorysmith/video-game-sales

## Files
| File | What it is |
|---|---|
| B3_B1_07_SIPROJECT.ipynb | Colab notebook with all practicals, models and the GUI |
| si_project_code.py | Complete project code (PR1 to PR7) in one Python file |
| app.py | Gradio GUI application |
| best_model.pkl, scaler.pkl | Best regression model (Linear Regression) and its scaler |
| best_classifier.pkl, scaler_classifier.pkl | Best classification model (Decision Tree) and its scaler |
| requirements.txt | Python libraries needed |
| vgsales.csv | Dataset (same as the Kaggle file) |

## How to run
**In Google Colab:** open the notebook, upload vgsales.csv, then Runtime > Run all. The last cell gives a public Gradio link.

**On your computer:**
```
pip install -r requirements.txt
python app.py
```

## Results
| Task | Best model | Result |
|---|---|---|
| Classification (Hit / Not Hit) | Decision Tree | Accuracy 95.8%, F1-Score 0.82 |
| Regression (Global Sales) | Linear Regression | R2 0.92, RMSE 0.58 |

## Tools
Python, Google Colab, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels, Scikit-learn, Requests, BeautifulSoup, Joblib, Gradio

# ==========================================================
# SI LAB PROJECT: VIDEO GAME SALES ANALYSIS AND HIT PREDICTION
# Himani Dalvi | Roll No. 07 | Batch B3-B1
# Dataset: vgsales.csv (Kaggle: https://www.kaggle.com/datasets/gregorysmith/video-game-sales)
# How to run: keep vgsales.csv in the same folder, then run:  python si_project_code.py
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------- PR1 - WEB SCRAPING (collecting video game data from a website)

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_best-selling_video_games"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
print("Status code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")
print("Page title:", soup.title.text)
print("Heading:", soup.find("h1").text)

#Extract the best-selling games table
table = soup.find("table", class_="wikitable")
header = [th.get_text(strip=True) for th in table.find("tr").find_all("th")]
print(header)

n = len(header) - 1     # all columns except Rank
games = []

for row in table.find_all("tr")[1:]:
    cols = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
    if len(cols) >= n:
        games.append(cols[-n:])     # tied games have no Rank cell, so take the last n cells

scraped = pd.DataFrame(games, columns=header[1:])
print("Games scraped:", len(scraped))

#Clean the scraped data and save to CSV
scraped = scraped.rename(columns={header[1]: "Title", header[2]: "Sales"})
scraped["Sales"] = scraped["Sales"].str.replace(r"\[.*?\]", "", regex=True)   # remove [a] notes
scraped["Sales"] = pd.to_numeric(scraped["Sales"], errors="coerce")
scraped = scraped.dropna(subset=["Sales"])

scraped.to_csv("best_selling_games.csv", index=False)
print(scraped[["Title", "Sales"]].head(10))
print("Average sales of these top games (millions):", scraped["Sales"].mean())


# ---------- LOAD DATASET (Kaggle: Video Game Sales, vgsales.csv)

df = pd.read_csv("vgsales.csv")
print(df.shape)
print(df.head())


# ---------- PREPROCESSING

print(df.columns)

df.info()

print("Missing values:")
print(df.isnull().sum())


# ---------- CLEANING

df = df.dropna()

df["Year"] = df["Year"].astype(int)
df = df.drop_duplicates()

print("Clean data shape:", df.shape)
print("Missing values left:", df.isnull().sum().sum())


# ---------- summary

print("Total games:", len(df))
print("Numerical columns: Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales")
print("Categorical columns: Platform, Genre, Publisher")
print()
print(df["Genre"].value_counts())


# ---------- PR2


# ---------- Global Sales statistics

g = df["Global_Sales"]
print("Mean=", g.mean())
print("Median=", g.median())
print("Mode=", g.mode()[0])
print("Variance=", g.var())
print("Std Dev=", g.std())
print("Range=", g.max() - g.min())
print("IQR=", g.quantile(0.75) - g.quantile(0.25))

print(df[["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].describe())

#Quartiles, min, max and skewness
Q1 = g.quantile(0.25)
Q2 = g.quantile(0.50)
Q3 = g.quantile(0.75)
print("Q1 =", Q1, " Q2 =", Q2, " Q3 =", Q3)
print("Min =", g.min(), " Max =", g.max())
print("Skewness =", g.skew())


# ---------- Genre-wise average sales

print(df.groupby("Genre")["Global_Sales"].mean().sort_values(ascending=False))


# ---------- Histogram of Global Sales

plt.hist(df["Global_Sales"], bins=50, edgecolor="black", range=(0, 5))
plt.title("Histogram of Global Sales (up to 5 million)")
plt.xlabel("Global Sales (millions)")
plt.ylabel("Frequency")
plt.show()


# ---------- Boxplot of Sales by Genre

plt.figure(figsize=(12, 5))
sns.boxplot(x="Genre", y="Global_Sales", data=df)
plt.ylim(0, 5)
plt.xticks(rotation=45)
plt.title("Boxplot of Global Sales by Genre")
plt.show()


# ---------- Top 10 Games by Sales

top10 = df.nlargest(10, "Global_Sales")
plt.figure(figsize=(10, 5))
plt.bar(top10["Name"], top10["Global_Sales"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Games by Global Sales")
plt.xlabel("Game")
plt.ylabel("Global Sales (millions)")
plt.show()


# ---------- Region-wise Sales Share

region = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
plt.pie(region, labels=region.index, autopct="%1.1f%%")
plt.title("Sales Share by Region")
plt.show()


# ---------- Scatter Plot

plt.scatter(df["NA_Sales"], df["Global_Sales"])
plt.title("NA Sales vs Global Sales")
plt.xlabel("NA Sales (millions)")
plt.ylabel("Global Sales (millions)")
plt.show()


# ---------- Line Chart

per_year = df.groupby("Year")["Global_Sales"].agg(["count", "sum"])

plt.figure(figsize=(10, 5))
plt.plot(per_year.index, per_year["count"], marker="o", label="Games released")
plt.plot(per_year.index, per_year["sum"], marker="o", label="Total Global Sales (millions)")
plt.title("Games Released and Global Sales per Year")
plt.xlabel("Year")
plt.legend()
plt.show()

print("Year with most games:", per_year["count"].idxmax())
print("Year with highest sales:", per_year["sum"].idxmax())


# ---------- Sampling and Distributions


# ---------- Compare population and samples

# 1. Simple Random Sampling
sample1 = df.sample(n=100, random_state=42)

# 2. Systematic Sampling
k = len(df) // 100
sample4 = df.iloc[::k]

# 3. Stratified Sampling
sample5 = df.groupby("Genre", group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=42))

# 4. Cluster Sampling
random_years = pd.Series(df["Year"].unique()).sample(n=3, random_state=42)
sample6 = df[df["Year"].isin(random_years)]

# Create the comparison table
comparison = pd.DataFrame({
    "Statistics": ["Size", "Mean", "Median", "Std Dev"],
    "Population": [len(df), df["Global_Sales"].mean(), df["Global_Sales"].median(), df["Global_Sales"].std()],
    "Simple Random": [len(sample1), sample1["Global_Sales"].mean(), sample1["Global_Sales"].median(), sample1["Global_Sales"].std()],
    "Systematic": [len(sample4), sample4["Global_Sales"].mean(), sample4["Global_Sales"].median(), sample4["Global_Sales"].std()],
    "Stratified": [len(sample5), sample5["Global_Sales"].mean(), sample5["Global_Sales"].median(), sample5["Global_Sales"].std()],
    "Cluster": [len(sample6), sample6["Global_Sales"].mean(), sample6["Global_Sales"].median(), sample6["Global_Sales"].std()]
})
print(comparison)


# ---------- Percentage error in mean

population_mean = df["Global_Sales"].mean()
sample_mean = sample1["Global_Sales"].mean()

error = abs((sample_mean - population_mean) / population_mean) * 100
print("Percentage Error in Mean:", error)


# ---------- Normal distribution

M = df["Global_Sales"].mean()
S = df["Global_Sales"].std()

norm = np.random.normal(M, S, len(df))

plt.figure(figsize=(6, 4))
plt.hist(norm, bins=30, edgecolor="black")
plt.title("Normal Distribution")
plt.xlabel("Global Sales")
plt.ylabel("Frequency")
plt.show()


# ---------- Binomial distribution

p = (df["Global_Sales"] >= 1).mean()
print("Probability of a hit game:", p)

binomial = np.random.binomial(n=5, p=p, size=1000)

plt.figure(figsize=(6, 4))
plt.hist(binomial, bins=11, edgecolor="black")
plt.title("Binomial Distribution")
plt.xlabel("Number of Hit Games out of 5")
plt.ylabel("Frequency")
plt.show()


# ---------- Poisson distribution

lambda_value = len(df) / df["Year"].nunique()
print("Lambda =", lambda_value)

poisson = np.random.poisson(lam=lambda_value, size=1000)

plt.figure(figsize=(6, 4))
plt.hist(poisson, bins=15, edgecolor="black")
plt.title("Poisson Distribution")
plt.xlabel("Number of Games per Year")
plt.ylabel("Frequency")
plt.show()

#Binomial probability using scipy
from scipy.stats import binom

p_hit = (df["Global_Sales"] >= 1).mean()
print("P(exactly 1 hit game out of 5) =", binom.pmf(1, 5, p_hit))
print("P(no hit game out of 5) =", binom.pmf(0, 5, p_hit))


# ---------- Uniform distribution

#If games were released equally in every year
uniform = np.random.uniform(df["Year"].min(), df["Year"].max(), len(df))

plt.figure(figsize=(6, 4))
plt.hist(uniform, bins=20, edgecolor="black")
plt.title("Uniform Distribution")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.show()

print("Uniform Mean Year:", uniform.mean())
print("Actual Mean Year:", df["Year"].mean())


# ---------- Distribution comparison: population vs samples

plt.figure(figsize=(8, 5))
plt.hist(df["Global_Sales"], bins=30, range=(0, 3), alpha=0.5, density=True, label="Population")
plt.hist(sample1["Global_Sales"], bins=30, range=(0, 3), alpha=0.5, density=True, label="Simple Random")
plt.hist(sample5["Global_Sales"], bins=30, range=(0, 3), alpha=0.5, density=True, label="Stratified")
plt.title("Population vs Samples")
plt.xlabel("Global Sales (millions)")
plt.ylabel("Density")
plt.legend()
plt.show()


# ---------- Probability distribution

probability = df["Year"].value_counts(normalize=True).sort_index()
print(probability.tail(10))

probability.plot(kind="bar", figsize=(12, 5))
plt.title("Probability Distribution of Release Year")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.show()


# ---------- Cumulative probability distribution (CDF)

cdf = probability.cumsum()

plt.plot(cdf.index, cdf.values, marker="o")
plt.title("Cumulative Probability of Release Year")
plt.xlabel("Year")
plt.ylabel("Cumulative Probability")
plt.show()

print("P(game released in or before 2005) =", cdf[2005])


# ---------- Population

population = df["Global_Sales"].values
population_mean = np.mean(population)
population_std = np.std(population)

print("Population Mean:", population_mean)
print("Population Std:", population_std)

plt.figure(figsize=(8, 5))
plt.hist(population, bins=50, range=(0, 5))
plt.title("Population Distribution (Right-Skewed)")
plt.xlabel("Global Sales")
plt.ylabel("Frequency")
plt.show()


# ---------- Repeated sampling, n = 30

np.random.seed(50)
sample_size = 30
number_of_sample = 1000
sample_means = []

for i in range(number_of_sample):
    sample = np.random.choice(population, size=sample_size)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)


# ---------- Distribution of sample means

plt.figure(figsize=(8, 5))
plt.hist(sample_means, bins=30, edgecolor="black")
plt.axvline(population_mean, linestyle="--", color="red", label="Population Mean")
plt.title("Distribution of Sample Means (n=30)")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# ---------- CLT results and standard error

observed_se = np.std(sample_means, ddof=1)
theoretical_se = population_std / np.sqrt(sample_size)

print("CLT Results")
print("Population Mean:", population_mean)
print("Mean of Sample Means:", np.mean(sample_means))
print("Observed Standard Error:", observed_se)
print("Theoretical Standard Error:", theoretical_se)


# ---------- Effect of sample size

sizes = [5, 30, 100]
plt.figure(figsize=(10, 5))

for n in sizes:
    means = [np.mean(np.random.choice(population, size=n)) for i in range(1000)]
    plt.hist(means, bins=30, alpha=0.5, label=f"n={n}")

plt.title("Sample Means for Different Sample Sizes")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# ---------- Point estimation

sample = df["Global_Sales"].sample(n=50, random_state=42)

estimated_mean = sample.mean()
estimated_variance = sample.var()
estimated_std = sample.std()

print("Point Estimation")
print("Estimated Mean:", estimated_mean)
print("Estimated Variance:", estimated_variance)
print("Estimated Standard Deviation:", estimated_std)

error = abs(estimated_mean - population_mean)
print("Estimation Error:", error)


# ---------- Skewness: population vs sample means

print("Population Skewness:", pd.Series(population).skew())
print("Sample Means Skewness (n=30):", pd.Series(sample_means).skew())


# ---------- Bootstrap

#Estimate the mean from ONE sample of 50 games by resampling it 5000 times
sample = df["Global_Sales"].sample(n=50, random_state=1)
bootstrap_means = []

for i in range(5000):
    boot = sample.sample(n=50, replace=True)
    bootstrap_means.append(boot.mean())

plt.hist(bootstrap_means, bins=30, edgecolor="black")
plt.title("Bootstrap Distribution of Mean Global Sales")
plt.xlabel("Bootstrap Mean")
plt.ylabel("Frequency")
plt.show()

print("Population Mean:", population_mean)
print("Bootstrap Estimate:", np.mean(bootstrap_means))


# ---------- Sample mean vs sample median as estimators

means = []
medians = []

for i in range(1000):
    s = df["Global_Sales"].sample(n=30, replace=True)
    means.append(s.mean())
    medians.append(s.median())

print("Population Mean:", df["Global_Sales"].mean(), "| Average Sample Mean:", np.mean(means))
print("Population Median:", df["Global_Sales"].median(), "| Average Sample Median:", np.mean(medians))
print("Variance of Mean Estimator:", np.var(means, ddof=1))
print("Variance of Median Estimator:", np.var(medians, ddof=1))


# ---------- Confidence Interval

data = df["Global_Sales"]
n = len(data)
sample_mean = data.mean()
sample_std = data.std()

confidence = 0.95
standard_error = sample_std / np.sqrt(n)
t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
margin_error = t_critical * standard_error

print("Lower Limit:", sample_mean - margin_error)
print("Upper Limit:", sample_mean + margin_error)

#Confidence Interval using Z value
z_critical = stats.norm.ppf((1 + confidence) / 2)
margin_z = z_critical * standard_error

print("Z critical =", z_critical)
print("Lower Limit:", sample_mean - margin_z)
print("Upper Limit:", sample_mean + margin_z)

#Confidence Interval plot
plt.figure(figsize=(8, 3))
plt.errorbar(sample_mean, 1, xerr=margin_error, fmt="o", capsize=8)
plt.axvline(sample_mean, color="red", linestyle="--", label="Mean")
plt.yticks([])
plt.title("95% Confidence Interval of Mean Global Sales")
plt.xlabel("Global Sales (millions)")
plt.legend()
plt.show()


# ---------- Hypothesis Tests (t-tests)


# ---------- One-sample t-test

t, p = stats.ttest_1samp(df["Global_Sales"], 0.5)
print("Sample mean =", df["Global_Sales"].mean())
print("t-statistic =", t)
print("p-value =", p)
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")


# ---------- Two-sample t-test

action = df[df["Genre"] == "Action"]["Global_Sales"]
sports = df[df["Genre"] == "Sports"]["Global_Sales"]

t, p = stats.ttest_ind(action, sports)
print("Action mean =", action.mean())
print("Sports mean =", sports.mean())
print("t-statistic =", t)
print("p-value =", p)
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")

#95% Confidence Interval for difference in means (Action - Sports)
diff = action.mean() - sports.mean()
se = np.sqrt(action.var() / len(action) + sports.var() / len(sports))
dof = len(action) + len(sports) - 2

ci = stats.t.interval(0.95, dof, loc=diff, scale=se)
print("Difference in means =", diff)
print("95% CI =", ci)


# ---------- Paired t-test

t, p = stats.ttest_rel(df["NA_Sales"], df["EU_Sales"])
print("NA mean =", df["NA_Sales"].mean())
print("EU mean =", df["EU_Sales"].mean())
print("t-statistic =", t)
print("p-value =", p)
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")


# ---------- Z-test

#H0: mean Global Sales = 0.5 million
mu = 0.5
z = (df["Global_Sales"].mean() - mu) / (df["Global_Sales"].std() / np.sqrt(len(df)))
p_z = 2 * (1 - stats.norm.cdf(abs(z)))

print("Z-statistic =", z)
print("p-value =", p_z)
print("Decision:", "Reject H0" if p_z < 0.05 else "Fail to reject H0")

#Normal curve with critical values (alpha = 0.05)
x = np.linspace(-4, 4, 1000)

plt.figure(figsize=(8, 5))
plt.plot(x, stats.norm.pdf(x))
plt.axvline(-1.96, color="red", linestyle="--", label="Critical value = -1.96 / +1.96")
plt.axvline(1.96, color="red", linestyle="--")
plt.axvline(z, color="green", label="Our Z = " + str(round(z, 2)))
plt.title("Z-test: Critical Region")
plt.legend()
plt.show()

#Hypothesized mean vs sample mean
plt.bar(["Hypothesis mean", "Sample mean"], [mu, df["Global_Sales"].mean()])
plt.title("Hypothesized Mean vs Sample Mean")
plt.ylabel("Global Sales (millions)")
plt.show()


# ---------- ANOVA

groups = [grp["Global_Sales"].values for name, grp in df.groupby("Genre")]
F, p = stats.f_oneway(*groups)

print("F-statistic =", F)
print("p-value =", p)
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")

#Average sales by genre
df.groupby("Genre")["Global_Sales"].mean().sort_values().plot(kind="bar")
plt.title("Average Global Sales by Genre")
plt.ylabel("Average Global Sales (millions)")
plt.show()


# ---------- Two-way ANOVA

#Effect of Genre, Platform and their interaction on Global Sales
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

top5_platforms = df["Platform"].value_counts().head(5).index
top5_genres = df["Genre"].value_counts().head(5).index
data2 = df[df["Platform"].isin(top5_platforms) & df["Genre"].isin(top5_genres)]

model = ols("Global_Sales ~ C(Genre) + C(Platform) + C(Genre):C(Platform)", data=data2).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

print("Genre significant?", anova_table["PR(>F)"]["C(Genre)"] < 0.05)
print("Platform significant?", anova_table["PR(>F)"]["C(Platform)"] < 0.05)
print("Interaction significant?", anova_table["PR(>F)"]["C(Genre):C(Platform)"] < 0.05)


# ---------- Chi-Square Test

top_platforms = df["Platform"].value_counts().head(5).index
top_genres = df["Genre"].value_counts().head(5).index
small = df[df["Platform"].isin(top_platforms) & df["Genre"].isin(top_genres)]

observed = pd.crosstab(small["Genre"], small["Platform"])
print(observed)

chi2, p, dof, expected = stats.chi2_contingency(observed)
print("Chi-Square:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p)
print("Decision:", "Reject H0" if p < 0.05 else "Fail to reject H0")

#Expected frequencies
print(pd.DataFrame(expected, index=observed.index, columns=observed.columns).round(1))

#Degree of freedom by formula
print("DOF = (rows - 1) x (columns - 1) =", (observed.shape[0] - 1) * (observed.shape[1] - 1))


# ---------- Chi-Square goodness of fit

#H0: games are equally spread across the top 5 genres
from scipy.stats import chisquare

genre_counts = df["Genre"].value_counts().head(5)
print(genre_counts)

chi2_gof, p_gof = chisquare(genre_counts)
print("Chi-Square:", chi2_gof)
print("p-value:", p_gof)
print("Decision:", "Reject H0" if p_gof < 0.05 else "Fail to reject H0")


# ---------- Correlation

pairs = [("NA_Sales", "Global_Sales"), ("EU_Sales", "Global_Sales"), ("JP_Sales", "Global_Sales")]

for a, b in pairs:
    x, y = df[a], df[b]
    r, p1 = stats.pearsonr(x, y)
    rho, p2 = stats.spearmanr(x, y)
    print(a, "vs", b)
    print("Covariance =", x.cov(y))
    print("Pearson r =", r, ", p =", p1)
    print("Spearman =", rho, ", p =", p2)
    print("Decision:", "Reject H0 (Significant)" if p1 < 0.05 else "Fail to reject H0")
    print()


# ---------- Correlation heatmap

sns.heatmap(df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

#Direction and strength
r = df["NA_Sales"].corr(df["Global_Sales"])
print("Correlation (NA vs Global) =", r)
print("Direction:", "Positive" if r > 0 else "Negative")
print("Strength:", "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.3 else "Weak")

#Covariance matrix
print(df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].cov())


# ---------- PR7 - PART A: REGRESSION (Predict Global Sales from NA Sales)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X = df[["NA_Sales"]]      #Independent variable
Y = df["Global_Sales"]    #Dependent variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Training Sample:", len(X_train))
print("Testing Sample:", len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train all 5 models and compare
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []
for name, m in models.items():
    m.fit(X_train, Y_train)
    pred = m.predict(X_test)
    mae = mean_absolute_error(Y_test, pred)
    mse = mean_squared_error(Y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, pred)
    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
results_df = results_df.sort_values(by="R2", ascending=False)
print(results_df)

#Linear regression equation
lin = models["Linear Regression"]
print("Intercept:", lin.intercept_)
print("Slope:", lin.coef_[0])
print("Equation: Global Sales =", round(lin.intercept_, 2), "+", round(lin.coef_[0], 2), "* NA Sales (scaled)")

#Actual vs predicted (first 5 test games)
prediction_df = pd.DataFrame({"Actual": Y_test.values[:5]})
for name, m in models.items():
    prediction_df[name] = m.predict(X_test[:5]).round(2)
print(prediction_df)

#Best model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print("Best Model:", best_model_name)

#Save and load the model
import joblib
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------- PR7 - PART B: CLASSIFICATION (Will the game be a HIT?)

#Target: Hit = 1 if Global Sales >= 1 million, else 0
df["Hit"] = (df["Global_Sales"] >= 1).astype(int)
print(df["Hit"].value_counts())

df["Hit"].value_counts().plot(kind="bar")
plt.title("Class Distribution (0 = Not Hit, 1 = Hit)")
plt.xlabel("Hit")
plt.ylabel("Number of Games")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Xc = df[["NA_Sales"]]     #Input feature
yc = df["Hit"]            #Target (0 or 1)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
print("Training Sample:", len(Xc_train))
print("Testing Sample:", len(Xc_test))

scaler_c = StandardScaler()
Xc_train = scaler_c.fit_transform(Xc_train)
Xc_test = scaler_c.transform(Xc_test)

#Train 4 classification models and compare
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

c_results = []
for name, c in classifiers.items():
    c.fit(Xc_train, yc_train)
    pred = c.predict(Xc_test)
    c_results.append([name,
                      accuracy_score(yc_test, pred),
                      precision_score(yc_test, pred),
                      recall_score(yc_test, pred),
                      f1_score(yc_test, pred)])

c_results_df = pd.DataFrame(c_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
c_results_df = c_results_df.sort_values(by="F1-Score", ascending=False)
print(c_results_df)

#Best classification model (chosen by F1-Score)
best_clf_name = c_results_df.iloc[0]["Model"]
best_clf = classifiers[best_clf_name]
print("Best Classification Model:", best_clf_name)

cm = confusion_matrix(yc_test, best_clf.predict(Xc_test))
print(cm)

ConfusionMatrixDisplay(cm, display_labels=["Not Hit", "Hit"]).plot()
plt.title("Confusion Matrix - " + best_clf_name)
plt.show()

#Save and load the classification model
joblib.dump(best_clf, "best_classifier.pkl")
joblib.dump(scaler_c, "scaler_classifier.pkl")

clf = joblib.load("best_classifier.pkl")
scaler_c = joblib.load("scaler_classifier.pkl")

def predict_game(na_sales):
    x = pd.DataFrame({"NA_Sales": [na_sales]})
    sales = model.predict(scaler.transform(x))[0]
    hit = clf.predict(scaler_c.transform(x))[0]
    result = "Predicted Global Sales: " + str(round(sales, 2)) + " million copies\n"
    result = result + "Prediction: " + ("HIT game" if hit == 1 else "NOT a hit")
    return result

print(predict_game(1.0))     #test: a game that sold 1 million copies in North America
print(predict_game(0.2))


# ---------- PR7 - GUI: Streamlit app

# The Streamlit GUI code is in app.py. To open the app, run:  streamlit run app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

def load_shark_data():
    return pd.read_csv("shark.csv")

shark = load_shark_data()
shark.info()

'''
categorical_features = [
    "Season Number",
    "Startup Name",
    "Episode Number",
    "Pitch Number",
    "Season Start",
    "Season End",
    "Original Air Date",
    "Industry",
    "Business Description",
    "Company Website",
    "Pitchers Gender",
    "Pitchers City",
    "Pitchers State",
    "Pitchers Average Age",
    "Entrepreneur Names",
    "Multiple Entrepreneurs",
    "Got Deal",
    "Royalty Deal",
    "Deal has conditions",
    "Guest Name",
    "Barbara Corcoran Present",
    "Mark Cuban Present",
    "Lori Greiner Present",
    "Robert Herjavec Present",
    "Daymond John Present",
    "Kevin O Leary Present",
    "Guest Present"
]

numerical_features = [
    "US Viewership",
    "Original Ask Amount",
    "Original Offered Equity",
    "Valuation Requested",
    "Total Deal Amount",
    "Total Deal Equity",
    "Deal Valuation",
    "Number of sharks in deal",
    "Investment Amount Per Shark",
    "Equity Per Shark",
    "Advisory Shares Equity",
    "Loan",
    "Barbara Corcoran Investment Amount",
    "Barbara Corcoran Investment Equity",
    "Mark Cuban Investment Amount",
    "Mark Cuban Investment Equity",
    "Lori Greiner Investment Amount",
    "Lori Greiner Investment Equity",
    "Robert Herjavec Investment Amount",
    "Robert Herjavec Investment Equity",
    "Daymond John Investment Amount",
    "Daymond John Investment Equity",
    "Kevin O Leary Investment Amount",
    "Kevin O Leary Investment Equity",
    "Guest Investment Amount",
    "Guest Investment Equity"
]
'''

#Requirement 1
print(shark["Industry"].value_counts())

print(shark.describe())

plt.rcParams["figure.constrained_layout.use"] = True

ax = shark.hist(figsize=(18, 14), bins=30, layout=(6, 9))
for a in ax.ravel():
    a.set_title(a.get_title(), fontsize=7)
plt.suptitle("Histograms of Numeric Features", y=1.02)
plt.show()

#Requirement 2
plt.figure(figsize=(6, 5))
plt.scatter(
    shark["Original Offered Equity"],
    shark["Deal Valuation"],
    alpha=0.5
)
plt.yscale("log")
plt.xlabel("Original Offered Equity (%)")
plt.ylabel("Deal Valuation (log scale)")
plt.title("Equity Offered vs Deal Valuation")
plt.show()


#Requirement 3
corr_matrix = shark.corr(numeric_only = True)
print(corr_matrix["Got Deal"].sort_values(ascending=False))


#Requirement 4
leakage_columns = [
    "Total Deal Amount",
    "Total Deal Equity",
    "Deal Valuation",
    "Number of sharks in deal",
    "Investment Amount Per Shark",
    "Equity Per Shark",
    "Advisory Shares Equity",
    "Loan",
    "Barbara Corcoran Investment Amount",
    "Barbara Corcoran Investment Equity",
    "Mark Cuban Investment Amount",
    "Mark Cuban Investment Equity",
    "Lori Greiner Investment Amount",
    "Lori Greiner Investment Equity",
    "Robert Herjavec Investment Amount",
    "Robert Herjavec Investment Equity",
    "Daymond John Investment Amount",
    "Daymond John Investment Equity",
    "Kevin O Leary Investment Amount",
    "Kevin O Leary Investment Equity",
    "Guest Investment Amount",
    "Guest Investment Equity"
]

shark_prep = shark.drop(columns=leakage_columns, errors="ignore")

target = shark["Got Deal"]

categorical_features = [
    "Industry",
    "Pitchers Gender",
    "Pitchers State",
    "Multiple Entrepreneurs",
    "Barbara Corcoran Present",
    "Mark Cuban Present",
    "Lori Greiner Present",
    "Robert Herjavec Present",
    "Daymond John Present",
    "Kevin O Leary Present",
    "Guest Present"
]

numerical_features = [
    "Season Number",
    "Episode Number",
    "Pitch Number",
    "US Viewership",
    "Original Ask Amount",
    "Original Offered Equity",
    "Valuation Requested"
]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])

shark_prepared = preprocessor.fit_transform(shark_prep)

#Requirement 5
log_reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

log_reg_f1_scores = cross_val_score(
    log_reg_pipeline, shark_prep, target, cv=5, scoring="f1"
)

rf_f1_scores = cross_val_score(
    rf_pipeline, shark_prep, target, cv=5, scoring="f1"
)

print("LogReg (balanced) CV F1:", log_reg_f1_scores.mean())
print("RandomForest CV F1:", rf_f1_scores.mean())


#target.value_counts(normalize=True)
#print(target.value_counts(normalize=True))

#Requirement 6
log_reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

'''
param_grid_1 = {
    "model__solver": ["liblinear", "lbfgs"],
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l2"]
}

grid_1 = GridSearchCV(
    log_reg_pipeline,
    param_grid_1,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_1.fit(shark_prep, target)
print("Search 1 best params:", grid_1.best_params_)
print("Search 1 best F1:", grid_1.best_score_)
'''

'''
param_grid_2 = {
    "model__solver": ["liblinear"],
    "model__penalty": ["l2"],
    "model__C": [0.0001, 0.0005, 0.001, 0.005, 0.01]
}

grid_2 = GridSearchCV(
    log_reg_pipeline,
    param_grid_2,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_2.fit(shark_prep, target)

print("Search 2 best params:", grid_2.best_params_)
print("Search 2 best F1:", grid_2.best_score_)
'''

'''
param_grid_3 = {
    "model__solver": ["liblinear"],
    "model__penalty": ["l1", "l2"],
    "model__C": [0.00005, 0.0001, 0.0002]
}

grid_3 = GridSearchCV(
    log_reg_pipeline,
    param_grid_3,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_3.fit(shark_prep, target)

print("Search 3 best params:", grid_3.best_params_)
print("Search 3 best F1:", grid_3.best_score_)
'''

#Final model after hyperparameter tuning
final_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        C=0.00005
    ))
])

#Requirement 7
shark_train, shark_test, target_train, target_test = train_test_split(
    shark_prep,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

final_model.fit(shark_train, target_train)

target_pred = final_model.predict(shark_test)

test_f1 = f1_score(target_test, target_pred)
print("Test F1-score:", test_f1)

print("\nClassification Report:")
print(classification_report(target_test, target_pred))

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

def load_student_data():
    return pd.read_csv("student_data.csv")

students = load_student_data()

def do_the_cut(data):
    data['study_cat'] = pd.cut(
        data["study_hours_per_day"],
        bins=[0., 1.5, 3., 5., 7., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    data['study_cat'] = data['study_cat'].fillna(3)

do_the_cut(students)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(students, students["study_cat"]):
    strat_train_set = students.loc[train_index]
    strat_test_set = students.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("study_cat", axis=1, inplace=True)

data = strat_train_set.copy()

numeric_cols = [
    "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
    "attendance_percentage", "sleep_hours", "exercise_frequency",
    "mental_health_rating", "exam_score"
]
scatter_matrix(data[numeric_cols], figsize=(12, 8))
plt.show()

data["screen_time_total"] = data["social_media_hours"] + data["netflix_hours"]
data["sleep_study_ratio"] = data["sleep_hours"] / (data["study_hours_per_day"] + 0.1)
data["attendance_study_interaction"] = data["attendance_percentage"] * data["study_hours_per_day"]

corr_matrix = data.corr(numeric_only=True)
print(corr_matrix["exam_score"].sort_values(ascending=False))


corr_matrix = students.corr(numeric_only=True)
print(corr_matrix["exam_score"].sort_values(ascending=False))

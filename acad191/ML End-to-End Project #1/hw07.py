import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#The imports below are new for 4/7 lecture
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
#imnports for 4/9 lecture
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

def load_student_data():
    return pd.read_csv("student_data.csv")

student = load_student_data()

student["academic_effort"] = student["attendance_percentage"] * student["study_hours_per_day"]

student = student.drop(["age", "attendance_percentage"], axis=1)

num_cols = student.select_dtypes(include=['int64', 'float64']).columns
cat_cols = student.select_dtypes(include=['object']).columns
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

X_prep = preprocessor.fit_transform(student)

encoded_cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)

all_feature_names = np.concatenate([num_cols, encoded_cat_names])

X_prep_student = pd.DataFrame(X_prep, columns=all_feature_names)

X_prep_student["screen_time_total"] = student["social_media_hours"] + student["netflix_hours"]
X_prep_student["sleep_study_ratio"] = student["sleep_hours"] / student["study_hours_per_day"]

X_prep_student["exam_score"] = student["exam_score"]

corr_matrix = X_prep_student.corr(numeric_only=True)
print(corr_matrix["exam_score"].sort_values(ascending=False))

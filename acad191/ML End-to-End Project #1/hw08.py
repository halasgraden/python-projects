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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def load_student_data():
    return pd.read_csv("student_data.csv")

def do_the_cut(student):
    student['exam_cat'] = pd.cut(student["exam_score"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])

student = load_student_data()
do_the_cut(student)

student["academic_effort"] = student["attendance_percentage"] * student["study_hours_per_day"]

student = student.drop(["age", "attendance_percentage"], axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(student, student["exam_cat"]):
    strat_train_set = student.loc[train_index]
    strat_test_set = student.loc[test_index]

student = strat_train_set.copy()

student = strat_train_set.drop("exam_score", axis=1)
student_labels = strat_train_set["exam_score"].copy()


num_cols = student.select_dtypes(include=['int64', 'float64']).columns
cat_cols = student.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
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

X_prep_student["exam_score"] = student_labels.values

corr_matrix = X_prep_student.corr(numeric_only=True)
print(corr_matrix["exam_score"].sort_values(ascending=False))

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_prep, student_labels)

some_data = student.iloc[:5]
some_labels = student_labels.iloc[:5]
some_data_prepared = preprocessor.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

student_predictions = lin_reg.predict(X_prep)
lin_mse = mean_squared_error(student_labels, student_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


lin_scores = cross_val_score(lin_reg, X_prep, student_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("LE_Scores:", lin_rmse_scores)
print("LE_Mean:", lin_rmse_scores.mean())
print("LE_Standard deviation:", lin_rmse_scores.std())

#Decision Tree Regressor
'''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_prep, student_labels)
student_predictions = tree_reg.predict(X_prep)
tree_mse = mean_squared_error(student_labels, student_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


scores = cross_val_score(tree_reg, X_prep, student_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#display the results
print("DT_Scores:", scores)
print("DT_Mean:", scores.mean())
print("DT_Standard deviation:", scores.std())
'''

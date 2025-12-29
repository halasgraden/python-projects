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
import customtransformer as cf


#imports for v6
from sklearn.model_selection import GridSearchCV

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

student = student.drop("student_id", axis=1)
strat_train_set = strat_train_set.drop("student_id", axis=1)
strat_test_set = strat_test_set.drop("student_id", axis=1)

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

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()

'''
scores = cross_val_score(tree_reg, X_prep, student_labels,
                          scoring="neg_mean_squared_error", cv=5)
baseline_rmse = np.sqrt(-scores.mean())
print("Baseline RMSE:", baseline_rmse)
'''


param_grid = [
    {"n_estimators": [110, 130, 150], 
     "max_features": [9, 10, 11]}
]

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

grid_search.fit(X_prep, student_labels)
print(grid_search.best_params_)

'''
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    rmse = np.sqrt(-mean_score)
    print(rmse, params)
'''

final_model = grid_search.best_estimator_


X_test = strat_test_set.drop("exam_score", axis=1)
y_test = strat_test_set["exam_score"]

X_test_prepared = preprocessor.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE:", final_rmse)

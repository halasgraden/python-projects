import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy
import sklearn

STUDENT_PATH = os.path.join("datasets", "student")

def load_student_data():
    return pd.read_csv("student_data.csv")


student = load_student_data()
student.info()  #The info method is useful to get a quick description of the data

print(student["gender"].value_counts()) #Shows what categories exist and how many districts belong to each category
print(student["part_time_job"].value_counts())
print(student["diet_quality"].value_counts())
print(student["parental_education_level"].value_counts())
print(student["internet_quality"].value_counts())

print(student.describe()) #This method shows a summary of the numerical attributes

student.hist(bins=50, figsize=(14,9)) #shows the number of instances (vertical axis) that have a given value range
plt.show()   #Plots a histogram for each numerical attribute

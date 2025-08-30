import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

students = ['ahmed', 'mohamed', 'sara', 'ail', 'lina']
grades_math = [80, 70, 40, 90, 20]
grades_science = [30, 60, 70, 100, 50]
grades_english = [50, 60, 90, 10, 40]
behavior = ['good', 'bad', 'bad', 'good', 'good']
absences = [2, 2, 3, 5, 3]
issues = [0, 1, 1, 0, 2]

df = pd.DataFrame({
    'students': students,
    'Math': grades_math,
    'Science': grades_science,
    'English': grades_english,
    'Behavior': behavior,
    'Absences': absences,
    'Issues': issues
})

df.to_csv('students.csv', index=False)


le = LabelEncoder()
df['Behavior_Numeric'] = le.fit_transform(df['Behavior'])


X = df[['Math', 'Science', 'English', 'Absences', 'Issues']]
y = df['Behavior_Numeric']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

new_student = pd.DataFrame({
    'Math': [85],
    'Science': [85],
    'English': [91],
    'Absences': [0],
    'Issues': [10]
})

new_student_predict = model.predict(new_student)
convert = le.inverse_transform(new_student_predict)

print(f'Behavior new student is : {convert[0]}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#EDA

file_path = 'fetal_health.csv'
data = pd.read_csv(file_path)
data.info()

colors = ['red', 'green', 'blue']
sns.countplot(data=data, x='fetal_health', hue='fetal_health', palette=colors, legend='auto')

#Modeling

X = data.drop('fetal_health', axis=1)
y = data['fetal_health']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1score:.2f}")

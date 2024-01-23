import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = 'fetal_health.csv'
data = pd.read_csv(file_path)

X = data.drop('fetal_health', axis=1)
y = data['fetal_health']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
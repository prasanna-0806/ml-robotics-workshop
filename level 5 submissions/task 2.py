import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data = pd.read_csv("robot_sensor_data.csv")
if data.isnull().sum().any():
    data = data.fillna(data.mean())  
X = data.drop(columns=['label'])  
y = data['label']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)  # 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv("robot_sensor_classification_results.csv", index=False)
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", values_format='d')
plt.title("confusion matrix")
plt.show()

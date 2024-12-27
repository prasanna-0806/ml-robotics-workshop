import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
regression_data = pd.read_csv("robot_movements.csv")
X_reg = regression_data[['speed', 'acceleration', 'rotation']]
y_reg = regression_data['future_movement']
scaler = StandardScaler()
X_reg = scaler.fit_transform(X_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"--- Regression Model ---")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.6)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red', linestyle='--')
plt.xlabel('Actual Future Movement')
plt.ylabel('Predicted Future Movement')
plt.title('Actual vs Predicted (Regression Model)')
plt.show()
classification_data = pd.read_csv("robot_sensor_data.csv")
for col in classification_data.columns:
    if classification_data[col].dtype in ['float64', 'int64']:
        classification_data[col].fillna(classification_data[col].mean(), inplace=True)
label_encoder = LabelEncoder()
classification_data['label'] = label_encoder.fit_transform(classification_data['label'])
X_class = classification_data[['temperature', 'vibration', 'proximity']]
y_class = classification_data['label']
X_class = scaler.fit_transform(X_class)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
class_model = RandomForestClassifier(random_state=42)
class_model.fit(X_train_class, y_train_class)
y_pred_class = class_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')
print(f"--- Classification Model ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nxlassification report:\n", classification_report(y_test_class, y_pred_class))
ConfusionMatrixDisplay.from_estimator(class_model, X_test_class, y_test_class, cmap="Blues", values_format='d')
plt.title("confusion matrix (classification model)")
plt.show()

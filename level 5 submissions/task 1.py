import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv("robot_movements.csv")
if data.isnull().sum().any():
    data = data.dropna()  
X = data[['speed', 'acceleration', 'rotation']]
y = data['future_movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
print(f"root mean squared error (RMSE): {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv("robot_movement_predictions.csv", index=False)
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title("actual vs predicted movements")
plt.xlabel("actual movements")
plt.ylabel("predicted movements")
plt.grid()
plt.show()

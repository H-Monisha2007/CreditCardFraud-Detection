import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('advertising.csv')

#Info
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

#Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

#Visualize relationships with matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(df['TV'], df['Sales'], color='blue')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.title('TV vs Sales')

plt.subplot(1, 3, 2)
plt.scatter(df['Radio'], df['Sales'], color='green')
plt.xlabel('Radio Advertising')
plt.ylabel('Sales')
plt.title('Radio vs Sales')

plt.subplot(1, 3, 3)
plt.scatter(df['Newspaper'], df['Sales'], color='red')
plt.xlabel('Newspaper Advertising')
plt.ylabel('Sales')
plt.title('Newspaper vs Sales')

plt.tight_layout()
plt.show()

#Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr())

#Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

#Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

#Visualize Actual vs Predicted Sales
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

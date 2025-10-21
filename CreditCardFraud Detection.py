import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

print("🔹 Loading dataset...")
data = pd.read_csv("creditcard.csv") #dataset

print("\nDataset shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

print("\nClass distribution:\n", data['Class'].value_counts())

# Plot distribution
plt.figure(figsize=(5, 4))
classes = data['Class'].value_counts()
plt.bar(classes.index, classes.values, color=['green', 'red'])
plt.title('Distribution of Genuine (0) vs Fraudulent (1) Transactions')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Genuine', 'Fraud'])
plt.show()
# FEATURE SELECTION AND NORMALIZATION

X = data.drop('Class', axis=1)
y = data['Class']

# Normalize numerical columns
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# HANDLE CLASS IMBALANCE (Undersampling)
# Separate majority and minority classes
fraud = data[data['Class'] == 1]
genuine = data[data['Class'] == 0]

# Randomly undersample the majority class
genuine_sample = genuine.sample(n=len(fraud) * 5, random_state=42) # 5x fraud count
balanced_data = pd.concat([fraud, genuine_sample]).sample(frac=1, random_state=42)

print("\nAfter undersampling:")
print(balanced_data['Class'].value_counts())

# Split features and labels again after balancing
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Normalize again (on balanced data)
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])
#SPLIT INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
#TRAIN MODELS
print("\nTraining Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#EVALUATE MODELS
print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(classification_report(y_test, y_pred_log))

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, y_pred_rf))


#CONFUSION MATRIX (RANDOM FOREST)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Genuine', 'Fraud'])
plt.yticks(tick_marks, ['Genuine', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add counts in boxes
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()
#ROC CURVE COMPARISON
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6, 5))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


#CONCLUSION
print("""
✅ SUMMARY:
- Dataset was normalized and undersampled to handle imbalance.
- Logistic Regression and Random Forest models trained successfully.
- Random Forest generally performs better in detecting fraud.
- ROC Curve shows overall strong performance.
""")

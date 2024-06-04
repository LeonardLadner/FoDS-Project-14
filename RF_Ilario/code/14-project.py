import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data
data = pd.read_csv("../data/heart_2020_cleaned.csv")

# Encoding categorical variables (if not already preprocessed)
data_encoded = pd.get_dummies(data, drop_first=True)

# Separate the target variable and independent variables
X = data_encoded.drop('HeartDisease_Yes', axis=1)  
y = data_encoded['HeartDisease_Yes']

# Splitting the dataset into training and testing sets and creating balanced subsets of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

positive_patients = list(np.random.choice(X_train[y_train == 1].index, 1000))
negative_patients = list(np.random.choice(X_train[y_train == 0].index, 1000))
X_train_sub = X_train.loc[positive_patients+negative_patients]
y_train_sub = y_train.loc[positive_patients+negative_patients]

positive_patients_test = list(X_test[y_test == 1].index)
negative_patients_test = list(np.random.choice(X_test[y_test == 0].index, len(positive_patients_test)))
X_test_sub = X_test.loc[positive_patients_test+negative_patients_test]
y_test_sub = y_test.loc[positive_patients_test+negative_patients_test]

# Standardizing numeric features (Random Forest does not require feature scaling but it won't hurt)
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()
scaler = StandardScaler()
X_train_sub[numeric_features] = scaler.fit_transform(X_train_sub[numeric_features])
X_test_sub[numeric_features] = scaler.transform(X_test_sub[numeric_features])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100],  # Reduced range
    'max_features': [None, 'sqrt'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

# Use the threading backend
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,scoring='accuracy')

with parallel_backend('threading'):
    grid_search.fit(X_train_sub, y_train_sub)

best_rf = grid_search.best_estimator_

# Making predictions
y_pred = best_rf.predict(X_test_sub)
y_pred_prob = best_rf.predict_proba(X_test_sub)[:, 1]  # Probabilities for the positive class

# Evaluating the model
accuracy = accuracy_score(y_test_sub, y_pred)
conf_matrix = confusion_matrix(y_test_sub, y_pred)
cm_df = pd.DataFrame(conf_matrix)
cm_df.to_csv("../output/Confusion Matrix.csv")
report = classification_report(y_test_sub, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('../output/Classification Report.csv', index=True)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_sub, y_pred_prob)
roc_auc = roc_auc_score(y_test_sub, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlim([0.0 , 1.0])
plt.ylim([0.0 , 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("../output/Roc_Curve.png")

# Feature Importance
feature_importance = best_rf.feature_importances_
indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(12, 8))
sns.barplot(x=X_train_sub.columns[indices], y=feature_importance[indices])
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.tick_params(axis='x', rotation=90)
plt.savefig("../output/Feauture_Importance.png")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_sub, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("../output/Confusion_Matrix.png")
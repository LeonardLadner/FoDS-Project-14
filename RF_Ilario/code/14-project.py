import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numeric features (Random Forest does not require feature scaling but it won't hurt)
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Hyperparameter tuning using RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100],  # Reduced range
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

# Use the threading backend
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                   n_iter=50, cv=2, n_jobs=-1, verbose=2, random_state=42, 
                                   scoring='accuracy')

with parallel_backend('threading'):
    random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

# Making predictions
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
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
sns.barplot(x=feature_importance[indices], y=X_train.columns[indices])
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.savefig("../output/Feauture_Importance.png")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("../output/Confusion_Matrix.png")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

data = pd.read_csv("../data/heart_2020_cleaned.csv")

# Preprocessing the data
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('HeartDisease')  # Exclude the target variable from categorical columns
numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()

# Encode target variable
data['HeartDisease'] = data['HeartDisease'].map({'Yes': 1, 'No': 0})

data_encoded = pd.get_dummies(data, drop_first=False)


X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

pos_patients = list(np.random.choice(X_train[y_train == 1].index, 1000))
neg_patients = list(np.random.choice(X_train[y_train == 0].index, 1000))
X_train_sub = X_train.loc[pos_patients+neg_patients]
y_train_sub = y_train.loc[pos_patients+neg_patients]

pos_patients_test = list(X_test[y_test == 1].index)
neg_patients_test = list(np.random.choice(X_test[y_test == 0].index, len(pos_patients_test)))
X_test_sub = X_test.loc[pos_patients_test+neg_patients_test]
y_test_sub = y_test.loc[pos_patients_test+neg_patients_test]

scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()
X_train_sub[numeric_features] = scaler.fit_transform(X_train_sub[numeric_features])
X_test_sub[numeric_features] = scaler.transform(X_test_sub[numeric_features])


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}


grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train_sub, y_train_sub)


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


y_pred = grid_search.predict(X_test_sub)
y_pred_prob = grid_search.predict_proba(X_test_sub)[:, 1]


accuracy = accuracy_score(y_test_sub, y_pred)
conf_matrix = confusion_matrix(y_test_sub, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv('../output/Confusion Matrix LR.csv')
report = classification_report(y_test_sub, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('../output/Classification Report LR.csv', index=True)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Feature coefficients
coefficients = pd.DataFrame(grid_search.best_estimator_.coef_.flatten(), index=X_train_sub.columns,
                            columns=['Coefficients'])
coefficients = coefficients.sort_values(by='Coefficients', ascending=False)
coefficients.to_csv('../output/Feature Coefficients LR.csv')
plt.figure(figsize=(15, 12))
sns.barplot(x=coefficients.index, y=coefficients['Coefficients'], color='b')
plt.title('Feature Coefficients')
plt.xlabel('Feature')
plt.ylabel("Coefficient Value")
plt.xticks(rotation=90, ha='center')
plt.subplots_adjust(bottom=0.3)
plt.savefig('../output/Feature Coefficients LR.png')

# Permutation Importance
perm_importance = permutation_importance(grid_search.best_estimator_, X_test_sub, y_test_sub, n_repeats=2)
importance_df = pd.DataFrame({'Permutation Importance': perm_importance.importances_mean}, index=X.columns)
importance_df = importance_df.sort_values(by='Permutation Importance', ascending=False)
importance_df.to_csv('../output/Permutation Importance LR.csv')
plt.figure(figsize=(15,12))
sns.barplot(x=importance_df.index, y='Permutation Importance', data=importance_df, color='b')
plt.xlabel('Feature')
plt.ylabel('Permutation Importance')
plt.title('Permutation Importance')
plt.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.savefig('../output/Permutation Importance LR.png')

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("../output/Confusion Matrix LR.png")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_sub, y_pred_prob)
roc_auc = roc_auc_score(y_test_sub, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='Blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("../output/ROC Curve LR.png")



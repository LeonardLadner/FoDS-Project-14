import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
data_encoded = pd.get_dummies(data, drop_first=True)


X = data_encoded.drop('HeartDisease_Yes', axis=1)
y = data_encoded['HeartDisease_Yes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}


grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train, y_train)


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


y_pred = grid_search.predict(X_test)
y_pred_prob = grid_search.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)




if 'l1' in grid_search.best_params_['penalty']:  
    coefficients = pd.DataFrame(grid_search.best_estimator_.coef_.flatten(), index=X_train.columns, columns=['Coefficients'])
    coefficients = coefficients.sort_values(by='Coefficients', ascending=False)
    plt.figure(figsize=(20, 10))
    sns.barplot(x=coefficients['Coefficients'], y=coefficients.index)
    plt.title('Feature Importance')
    plt.savefig("../output/FeatureImportance.png")


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("../output/ConfusionMatrix.png")



fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='Blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("../output/ROCCurve.png")



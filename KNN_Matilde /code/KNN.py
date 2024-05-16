import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import math
from sklearn.model_selection import GridSearchCV
# Load the dataset
data = pd.read_csv('../data/heart_2020_cleaned.csv')

# Preprocessing the data
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('HeartDisease')  # Exclude the target variable from categorical columns
numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=505))
])

#aggiunto
param_grid = {
    'classifier__n_neighbors': [5, 10, 50, 100, 200, 300, 400, 500],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')


# Encode the target variable
data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)

# Splitting the dataset
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#aggiunta
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

#reinitialize the pipeline with the best parameters.
best_pipeline = grid_search.best_estimator_

# Train the model
best_pipeline.fit(X_train, y_train)

# Predictions
y_pred_best = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]  # probabilities for the positive class

#classification report
print(classification_report(y_test, y_pred_best))


# Optionally, you can also visualize the performance using ROC curve and Confusion Matrix
y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)


plt.figure(figsize=(10, 6))
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_best)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Best Model)')
plt.legend(loc="lower right")
plt.savefig("../output/ROC Curve Best Model.pdf", backend="pdf")
plt.close()

# Confusion Matrix
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix (Best Model)')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("../output/Confusion Matrix Best Model.pdf", backend="pdf")
plt.close()


# Select new data from the same dataset
new_data_same_dataset = data.sample(n=10, random_state=42)  # Selecting 10 random samples for demonstration, replace with your selection method

# Preprocess the new data
X_new_same_dataset = new_data_same_dataset.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is your target variable
y_new_pred_same_dataset = pipeline.predict(X_new_same_dataset)
y_new_prob_same_dataset = pipeline.predict_proba(X_new_same_dataset)[:, 1]  # probabilities for the positive class

# Optionally, you can append the predictions to the new data
new_data_same_dataset['Predicted_HeartDisease'] = y_new_pred_same_dataset
new_data_same_dataset['Probability_HeartDisease'] = y_new_prob_same_dataset

# Print or display the new data with predictions
print("New data with predictions:")
print(new_data_same_dataset)


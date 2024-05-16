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

# Encode the target variable
data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)

# Splitting the dataset
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming you have multiple training datasets
# Calculate the square root of the number of samples in each dataset
num_samples_train = len(X_train)
sqrt_num_samples_train = np.sqrt(num_samples_train)
# Calculate the mean of the square root values
square_root_mean = np.mean(sqrt_num_samples_train)
print("Square root of the number of samples in the training dataset:", square_root_mean)
# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # probabilities for the positive class

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("../output/ROC Curve.pdf",backend="pdf")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("../output/Confusion Matrix.pdf",backend="pdf")
plt.close()

# Classification Report
print(classification_report(y_test, y_pred))

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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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



# Separate the target variable and independent variables
X = data_encoded.drop('HeartDisease_Yes', axis=1)  # Adjust based on your actual target variable name
y = data_encoded['HeartDisease_Yes']

#aggiunta
#X['new_feature'] = np.log(X['some_feature'] + 1)
#selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
#X_selected = selector.transform(X)  # Questo X_selected sarà utilizzato per il training e il testing

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Standardizing numeric features
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()  # Assicurati che questa linea rifletta le trasformazioni
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])



# Initialize and train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)



# Making predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)



print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)



# Calculate scores
y_scores = model.predict_proba(X_test)[:, 1]
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate AUC
roc_auc = roc_auc_score(y_test, y_scores)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='Blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("../output/ROCCurve.png")




# Calculate precision-recall pairs
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("../output/Precision.png", dpi= 100)




# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("../output/Confusion.png")




# Extracting model coefficients
coefficients = pd.DataFrame(model.coef_.flatten(), index=X_train.columns, columns=['Coefficients'])
coefficients = coefficients.sort_values(by='Coefficients', ascending=False)
coefficients.index = coefficients.index.str.replace('_', ' ').str.title()
plt.figure(figsize=(20, 10))
sns.barplot(x=coefficients['Coefficients'], y=coefficients.index)
plt.title('Feature Importance')
plt.savefig("../output/FeatureImportance.png")


#class probability histogramm
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_prob, bins=30, kde=True, color='blue')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability of HeartDisease_Yes')
plt.ylabel('Frequency')
plt.savefig("../output/ClassProbability.png")

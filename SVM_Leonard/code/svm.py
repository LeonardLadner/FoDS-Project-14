import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, confusion_matrix, auc, classification_report
from sklearn import svm
from sklearn.inspection import permutation_importance
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# Classification report, confusion matrix, part of ROC curve
def performance_evaluation(y_eval, X_eval, cla):
    y_pred = cla.best_estimator_.predict(X_eval)
    y_prob = cla.best_estimator_.predict_proba(X_eval)[:, 1]

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    roc_auc = auc(fpr, tpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)

    class_report = classification_report(y_eval, y_pred, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    report_df.to_csv('../tables/Classification Report SVM.csv', index=True)
    cm = confusion_matrix(y_eval, y_pred)
    conf_matrix = pd.DataFrame(cm)
    conf_matrix.to_csv('../tables/Confusion Matrix SVM.csv')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("../plots/Confusion Matrix SVM.png")
    plt.close()


# Import data, assign dtype (can also be left out -> changes later in code)
data = pd.read_csv('../data/heart_2020_cleaned.csv',
                   dtype={
                       'BMI': float,
                       'Smoking': 'category',
                       'AlcoholDrinking': 'category',
                       'Stroke': 'category',
                       'PhysicalHealth': float,
                       'MentalHealth': float,
                       'DiffWalking': 'category',
                       'Sex': 'category',
                       'AgeCategory': 'category',
                       'Race': 'category',
                       'Diabetic': 'category',
                       'PhysicalActivity': 'category',
                       'GenHealth': 'category',
                       'SleepTime': float,
                       'Asthma': 'category',
                       'KidneyDisease': 'category',
                       'SkinCancer': 'category'
                   })

# Check for missing data
print(data.isna().sum())

# Map the label as 1/0 instead of yes/no
data['HeartDisease'] = data['HeartDisease'].map({'Yes': 1, 'No': 0})

# Check the dtypes
data_types = data.dtypes
print(data_types)

# Split features into numerical and categorical (label isn't a float, could be done differently/more optimally, here the
# dtype assignment from the start becomes relevant)
num_cols = data.select_dtypes(include=[float]).columns.tolist()
print(num_cols)
cat_cols = data.select_dtypes(include=['category']).columns.tolist()
print(cat_cols)

# Apply one hot encoding to the categorical features
encoded_data = pd.get_dummies(data, columns=cat_cols, dtype=int, drop_first=False)
print(encoded_data)

# Split the data into features and
X = encoded_data.drop('HeartDisease', axis=1)
y = encoded_data['HeartDisease']

# Check the prevalence of heart disease in data
print(y.mean())
print(y.sum())

# Define potential hyperparameters, not that many to avoid high computational cost
parameters = {'C': [1, 10, 100], 'gamma': [0.01, 0.1]}

# Create lists and interval used to calculate ROC curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Split data into train and test set, stratify y due to the low prevalence
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a subset of the data to reduce the massive computational costs from the large dataset
# Simultaneously balance the subset
pos_patients = list(np.random.choice(X_train[y_train == 1].index, 1000))
neg_patients = list(np.random.choice(X_train[y_train == 0].index, 1000))
X_train_sub = X_train.loc[pos_patients + neg_patients]
y_train_sub = y_train.loc[pos_patients + neg_patients]

pos_patients_test = list(X_test[y_test == 1].index)
neg_patients_test = list(np.random.choice(X_test[y_test == 0].index, len(pos_patients_test)))
X_test_sub = X_test.loc[pos_patients_test + neg_patients_test]
y_test_sub = y_test.loc[pos_patients_test + neg_patients_test]

# Standardization and data copies to avoid inplace operations
sc = StandardScaler()
X_train_scaled = X_train_sub.copy()
X_test_scaled = X_test_sub.copy()

X_train_scaled[num_cols] = sc.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols] = sc.transform(X_test_scaled[num_cols])

# Use SVM on the data. Cross-validation and hyperparameter tuning
SVM = svm.SVC(probability=True, kernel='linear')
clf = GridSearchCV(SVM, parameters, cv=5, scoring='accuracy')
clf.fit(X_train_scaled, y_train_sub)
print(clf.best_estimator_.get_params())

# Returns classification report and confusion matrix
performance_evaluation(y_test_sub, X_test_scaled, clf)

# Determine the permutation importance of each feature (importance of feature)
perm_importance = permutation_importance(clf.best_estimator_, X_test_scaled, y_test_sub, n_repeats=2)
importance_df = pd.DataFrame({'Permutation Importance': perm_importance.importances_mean}, index=X.columns)
importance_df = importance_df.sort_values(by='Permutation Importance', ascending=False)
importance_df.to_csv('../tables/Permutation Importance SVM.csv')
plt.figure(figsize=(15, 12))
sns.barplot(x=importance_df.index, y='Permutation Importance', data=importance_df, color='b')
plt.xlabel('Feature')
plt.ylabel('Permutation Importance')
plt.title('Permutation Importance')
plt.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.savefig('../plots/Permutation Importance SVM.png')
print(importance_df)

# Determine the coefficients of each feature (direction of feature)
coefficients = clf.best_estimator_.coef_[0]
coefficients_df = pd.DataFrame({'Coefficient': coefficients}, index=X.columns)
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
coefficients_df.to_csv('../tables/Feature Coefficients SVM.csv')
plt.figure(figsize=(15, 12))
sns.barplot(x=coefficients_df.index, y=coefficients_df['Coefficient'], color='b')
plt.title("Feature Coefficients")
plt.xlabel('Feature')
plt.ylabel("Coefficient Value")
plt.xticks(rotation=90, ha='center')
plt.subplots_adjust(bottom=0.3)
plt.savefig('../plots/Feature Coefficients SVM.png')
print(coefficients_df)

# Plot the ROC curve
plt.figure(figsize=(12, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'ROC curve (area = {mean_auc:.2f})')
std_tpr = np.std(tprs, axis=0)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('../plots/ROC Curve SVM.png')

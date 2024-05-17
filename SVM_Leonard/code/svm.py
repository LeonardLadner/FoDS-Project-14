import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, confusion_matrix, auc, accuracy_score, precision_score, recall_score, f1_score,
                             classification_report)
from sklearn import svm
from sklearn.inspection import permutation_importance
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def performance_evaluation(y_eval, X_eval, cla, name='Name'):
    y_pred = cla.best_estimator_.predict(X_eval)
    y_prob = cla.best_estimator_.predict_proba(X_eval)[:, 1]

    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    roc_auc = auc(fpr, tpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    axes[1].plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC {name} (AUC = {roc_auc:.2f})')
    print(classification_report(y_eval, y_pred))
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("../plots/Confusion Matrix.png")
    plt.close()


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

data['AgeCategory'] = data['AgeCategory'].cat.reorder_categories(new_categories=['18-24', '25-29', '30-34', '35-39',
                                                                                 '40-44', '45-49', '50-54', '55-59',
                                                                                 '60-64', '65-69', '70-74', '75-79',
                                                                                 '80 or older'], ordered=True)
data['Diabetic'] = data['Diabetic'].cat.reorder_categories(new_categories=['No', 'No, borderline diabetes',
                                                                           'Yes (during pregnancy)', 'Yes'],
                                                           ordered=True)
data['GenHealth'] = data['GenHealth'].cat.reorder_categories(new_categories=['Excellent', 'Very good', 'Good', 'Fair',
                                                                             'Poor'], ordered=True)

print(data.isna().sum())

data['HeartDisease'] = data['HeartDisease'].map({'Yes': 1, 'No': 0})

data_types = data.dtypes
print(data_types)

num_cols = data.select_dtypes(include=[float]).columns.tolist()
print(num_cols)
cat_cols = data.select_dtypes(include=['category']).columns.tolist()
print(cat_cols)

encoded_data = pd.get_dummies(data, columns=cat_cols, dtype=int, drop_first=False)
print(encoded_data)

X = encoded_data.drop('HeartDisease', axis=1)
y = encoded_data['HeartDisease']

print(y.mean())
print(y.sum())

parameters = {'C': [1, 10, 100], 'gamma': [0.01, 0.1]}

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

data_performance = pd.DataFrame(columns=['tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pos_patients = list(np.random.choice(X_train[y_train == 1].index, 600))
neg_patients = list(np.random.choice(X_train[y_train == 0].index, 600))
X_train_sub = X_train.loc[pos_patients+neg_patients]
y_train_sub = y_train.loc[pos_patients+neg_patients]

pos_patients_test = list(X_test[y_test == 1].index)
neg_patients_test = list(np.random.choice(X_test[y_test == 0].index, len(pos_patients_test)))
X_test_sub = X_test.loc[pos_patients_test+neg_patients_test]
y_test_sub = y_test.loc[pos_patients_test+neg_patients_test]

sc = StandardScaler()
X_train_scaled = X_train_sub.copy()
X_test_scaled = X_test_sub.copy()

X_train_scaled[num_cols] = sc.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols] = sc.transform(X_test_scaled[num_cols])

SVM = svm.SVC(probability=True, kernel='linear')
clf = GridSearchCV(SVM, parameters, cv=5)
clf.fit(X_train_scaled, y_train_sub)
print(clf.best_estimator_.get_params())

performance_evaluation(y_test_sub, X_test_scaled, clf, name='Test')

result = permutation_importance(clf.best_estimator_, X_test_scaled, y_test_sub, n_repeats=2)

importance_df = pd.DataFrame({'Permutation Importance': result.importances_mean}, index=X.columns)

print(data_performance)

coefficients = clf.best_estimator_.coef_[0]
coefficients_df = pd.DataFrame({'Coefficient': coefficients}, index=X.columns)
print(coefficients_df)

combined_df = pd.merge(coefficients_df, importance_df, right_index=True, left_index=True)
combined_df = combined_df.sort_values(by='Coefficient', ascending=False)
combined_df.to_csv('../tables/Coefficients and Permutation Importance')
print(combined_df)

normalized_importance = (combined_df['Permutation Importance'] - combined_df['Permutation Importance'].min()) / \
                        (combined_df['Permutation Importance'].max() - combined_df['Permutation Importance'].min())

bars = axes[0].bar(combined_df.index, combined_df['Coefficient'], color='blue')
for bar, alpha in zip(bars, normalized_importance):
    bar.set_alpha(alpha)

axes[0].set_xlabel('Coefficient Value')
axes[0].set_title('Feature Coefficients stratified by Permutation Importance')
axes[0].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom=0.7, wspace=0.2)

axes[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
axes[1].plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
axes[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')
axes[1].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve", xlabel='False Positive Rate',
            ylabel='True Positive Rate')
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.4)
plt.savefig('../plots/coefficients and roc_curve.png')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, mean_squared_error, confusion_matrix,
    classification_report, roc_curve, auc, ConfusionMatrixDisplay
)

# ================================
# Load and Explore Data
# ================================
data = pd.read_csv("Heart_Disease.csv")

print(data.head())
print(data.columns)
print(data.describe())
print(data.info())
print("Missing values before cleaning:\n", data.isnull().sum())

# ================================
# Preprocessing
# ================================
data['smoking_status'] = data['smoking_status'].replace('Unknown', np.nan)
data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

print("Missing values after cleaning:\n", data.isnull().sum())

# Encoding categorical variables
data['Gender'] = data['Gender'].replace({"Female": 0, "Male": 1})
data['Heart Disease'] = data['Heart Disease'].replace({"Yes": 1, "No": 0})
data['smoking_status'] = data['smoking_status'].replace({
    "never smoked": 0, "formerly smoked": 1, "smokes": 2
})

# Drop unnecessary columns
data.drop(['id','FBS over 120', 'work_type','Thallium'], axis=1, inplace=True)
data.drop(['Max HR','Exercise angina','Slope of ST','EKG results'], axis=1, inplace=True)

print("Data after preprocessing:\n", data.head(10))

# ================================
# Split Data
# ================================
x = data.drop(['Heart Disease'], axis=1)
y = data['Heart Disease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Feature Selection
selector = SelectKBest(chi2, k=7)
x_new = selector.fit_transform(x, y)
selected_features_indices = selector.get_support(indices=True)
selected_features_names = x.columns[selected_features_indices]
print("Selected features:", selected_features_names)

# ================================
# Models
# ================================
models = {}

# Logistic Regression
log_model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
log_model.fit(x_train, y_train)
models["Logistic Regression"] = log_model

# SVM
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(x_train, y_train)
models["SVM"] = svm_model

# Decision Tree
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt_model.fit(x_train, y_train)
models["Decision Tree"] = dt_model

# ================================
# Evaluation
# ================================
for name, model in models.items():
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Mean Square Error: {mse:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ================================
# Visualizations
# ================================

# 1. Distribution Plots
plt.figure(figsize=(8,6))
sns.histplot(data=data, x="Age", hue="Heart Disease", kde=True, bins=30, palette="Set2")
plt.title("Age Distribution by Heart Disease")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Heart Disease", y="Cholesterol", data=data, palette="Set1")
plt.title("Cholesterol Levels by Heart Disease")
plt.xticks([0,1], ["No", "Yes"])
plt.show()

# 2. Countplots
f = sns.countplot(x='Heart Disease', data=data)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(['No Heart Disease', 'Heart Disease'])
plt.xlabel("")
plt.show()

f = sns.countplot(x='Heart Disease', data=data, hue='Gender')
plt.legend(['Female', 'Male'])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(['No Heart Disease', 'Heart Disease'])
plt.xlabel("")
plt.show()

# 3. Heatmap
plt.figure(figsize=(10,8))
heat_map = sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.show()

# 4. Model Accuracy Comparison
accuracies = {name: accuracy_score(y_test, model.predict(x_test)) for name, model in models.items()}
plt.figure(figsize=(7,5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.show()

# 5. Confusion Matrix Heatmaps
for name, model in models.items():
    plt.figure(figsize=(5,4))
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# 6. ROC Curves
plt.figure(figsize=(7,6))
for name, model in models.items():
    y_prob = model.predict_proba(x_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Models")
plt.legend()
plt.show()

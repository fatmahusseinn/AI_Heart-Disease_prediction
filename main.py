import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2


#read data
data=pd.read_csv("Heart_Disease.csv")
#data info
data.head()

data.columns

data.describe()

data.info()

print(data.isnull().sum())

#preprocessing data handling missing values
data['smoking_status'] = data['smoking_status'].replace('Unknown', np.nan)
data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)

#checking handling is done
print(data.isnull().sum())

#Encoding data
data['Gender']=data['Gender'].replace("Female",0)
data['Gender']=data['Gender'].replace("Male",1)
data['Heart Disease']=data['Heart Disease'].replace("Yes",1)
data['Heart Disease']=data['Heart Disease'].replace("No",0)
data['smoking_status']=data['smoking_status'].replace("never smoked",0)
data['smoking_status']=data['smoking_status'].replace("formerly smoked",1)
data['smoking_status']=data['smoking_status'].replace("smokes",2)
data.drop(['id','FBS over 120', 'work_type','Thallium'], axis=1, inplace=True)
data.drop(['Max HR','Exercise angina','Slope of ST','EKG results'],axis=1,inplace=True)

#show data
data.head(10)


#split data
x=data.drop(['Heart Disease'],axis=1)
y=data['Heart Disease']

x_train,x_test,y_train,y_test=train_test_split(x , y , test_size=0.2 , random_state=1)

#scaling data
scaler= StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
scaler.fit(x_test)
x_test=scaler.transform(x_test)

#feature selection
selector = SelectKBest(chi2, k=7)
x_new = selector.fit_transform(x, y)
selected_features_indices = selector.get_support(indices=True)
selected_features_names = x.columns[selected_features_indices]
print("Selected features:", selected_features_names)


#LogisticRegression model
log_model=LogisticRegression(C=1.0,penalty='l2',solver='liblinear')
log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)

print("Accuracy of the logistic Regression %.3f" % metrics.accuracy_score(y_test,y_pred))

mse=mean_squared_error(y_test,y_pred)
print("Mean Square Error of the Logistic Regression %.3f" % mse)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

print("Classification Report: ")
print(classification_report(y_test,y_pred))


#svm model
svm_model=SVC(kernel='linear',C=1.0)
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)
print("Accuracy of the SVM %.3f" % metrics.accuracy_score(y_test,y_pred))

mse=mean_squared_error(y_test,y_pred)
print("Mean Square Error of the SVM %.3f" % mse)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

print("Classification Report: ")
print(classification_report(y_test,y_pred))


#DecisionTree model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt_model.fit(x_train,y_train)
y_pred = dt_model.predict(x_test)
print("Accuracy of the Decision Tree %.3f" % metrics.accuracy_score(y_test,y_pred))

mse=mean_squared_error(y_test,y_pred)
print("Mean Square Error of the Decision Tree %.3f" % mse)

print("Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

print("Classification Report: ")
print(classification_report(y_test,y_pred))

#visualization
f = sns.countplot(x='Heart Disease', data=data)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("")

f = sns.countplot(x='Heart Disease', data=data, hue='Gender')
plt.legend(['Female', 'Male'])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("")

heat_map = sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);







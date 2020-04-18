
import numpy as numpy
import pandas as pdd
import seaborn as sns
import matplotlib.pyplot as plt
data=pdd.read_csv("train.csv",header=0,sep=',',quotechar='"')
#a=data.info()
#b=data.describe()
#c=data.pivot_table(index='Sex',values='Survived')
#d=data.pivot_table(index='Pclass',values='Survived')
#e=d.plot.bar()
#f=plt.show()
#g=sns.FacetGrid(data,col='Survived')

#l=g.map(plt.hist,'Age',bins=20)
#grid=sns.FacetGrid(data,row='Pclass',col='Sex',height=2.2,aspect=1.6)
#grid.map(plt.hist,'Age',alpha=.5,bins=20)

#print(plt.show())
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pdd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']

data_new = process_age(data,cut_points,label_names)

age_cat_pivot = data_new.pivot_table(index="Age_categories",values="Survived")
age_cat_pivot.plot.bar()
#plt.show()
def create_dummies(df,column_name):
    dummies = pdd.get_dummies(df[column_name],prefix=column_name)
    df = pdd.concat([df,dummies],axis=1)
    return df

data_new = create_dummies(data_new,"Pclass")
#print(data_new.head())
data_new = create_dummies(data_new,"Sex")
data_new = create_dummies(data_new,"Age_categories")
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

X_data = data_new[columns]
Y_data = data_new['Survived']

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X_data,Y_data, test_size=0.2,random_state=0)
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
acc_log = accuracy_score(test_y, predictions)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_y)
predictions = knn.predict(test_X)
acc_knn = accuracy_score(test_y, predictions)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(test_X)
acc_decision_tree = accuracy_score(test_y, predictions)

svc = SVC(gamma='scale')
svc.fit(train_X, train_y)
predictions = svc.predict(test_X)
acc_svc = accuracy_score(test_y, predictions)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_y)
predictions = random_forest.predict(test_X)
acc_random_forest = accuracy_score(test_y, predictions)
models = pdd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
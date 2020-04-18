from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scoring = 'accuracy'
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier() ))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
seed = 7
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f {%f}" % (name, cv_results.mean(), cv_results.std())
    print(msg)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))    
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print(accuracy_score(y_test, predictions))
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions = dtc.predict(X_test)
print(accuracy_score(y_test, predictions))  

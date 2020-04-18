#Import functions
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
#Load data
boston = datasets.load_boston()
print(boston)

#Calculate some statistics
print("Statistics for Boston housing dataset:\n")
print( "Minimum price: ${:,.2f}".format(np.amin(boston.target)*1000))
print("Maximum price: ${:,.2f}".format(np.amax(boston.target)*1000))
print("Mean price: ${:,.2f}".format(np.mean(boston.target)*1000))
print("Median price ${:,.2f}".format(np.median(boston.target)*1000))
print("Standard deviation of prices: ${:,.2f}".format(np.std(boston.target)*1000))
#Shuffle and split the data into train (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.20, random_state=42)
#Define R2 score
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

#Optimized decision tree algorithm (grid search on 'max_depth' parameter)
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def fit_model(X, y):
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    #Decision tree regressor object
    regressor = DecisionTreeRegressor()

    #Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = [{'max_depth': range(1,11)}]

    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    #Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)

    #Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    #Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
#Mean squared error and 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)

#R2-score
print("R2 score: %.4f" % r2_score(y_test, reg.predict(X_test)))
#Regression: Gradient Boosting model with least squares loss
from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
print("R2 score: %.4f" % r2_score(y_test, clf.predict(X_test)))
#Plot training deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#Feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
from numpy import average  # noqa
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, make_scorer, f1_score, precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score  # noqa
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('./flood_tool/resources/postcodes_sampled.csv')
data = data.dropna()

# With unbalanced data
y_res = data.riskLabel
data = data[['easting', 'northing']]

scorer = make_scorer(recall_score, average='macro')
# #applied smote fct to balance the data
# #data = smote.smote_func('./flood_tool/resources/postcodes_sampled.csv')

# split the data
x_train, x_test, y_train, y_test = train_test_split(data,
                                                    y_res,
                                                    test_size=0.3,
                                                    random_state=42)
# Scale the data for Knn
sc = RobustScaler()
sc.fit(x_train)
X_train = sc.transform(x_train)
X_test = sc.transform(x_test)

# Knn
# Hyperparameter Grid
grid = {'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]}
# Instanciate Grid Search
search = GridSearchCV(KNeighborsClassifier(), grid,
                      scoring=scorer,
                      cv=5,
                      n_jobs=-2  # paralellize computation
                      )

search.fit(X_train, y_train)

# Save the best parameter
search.best_params_['n_neighbors']
search.best_score_
prediction = search.predict(X_test)
accuracy = sum(prediction == y_test)/(len(X_test))
accuracy
pd.DataFrame(confusion_matrix(prediction, y_test),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# RandomForest
# Hyperparameter Grid
grid2 = {'n_estimators': [50, 100, 150],
         'max_depth': [2, 10, 50]}
# Instanciate Grid Search
search2 = GridSearchCV(RandomForestClassifier(), grid2,
                       scoring=scorer,
                       cv=5,
                       n_jobs=-2  # paralellize computation
                       )
search2.fit(x_train, y_train)

# Save the best parameter
search2.best_params_['n_estimators']
search2.best_params_['max_depth']
search2.best_score_
prediction2 = search2.predict(x_test)
accuracy2 = sum(prediction2 == y_test)/(len(x_test))
accuracy2
pd.DataFrame(confusion_matrix(prediction2, y_test),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# NN
# Hyperparameter Grid
grid3 = {'hidden_layer_sizes': [(20, 30, 20), (10, 50, 10), (10, 10, 10)],
         'activation': ['relu', 'logistic'],
         'solver': ['lgbs', 'sgd', 'adam']
         }
# Instanciate Grid Search
search3 = GridSearchCV(MLPClassifier(), grid3,
                       scoring=scorer,
                       cv=5,
                       n_jobs=-2  # paralellize computation
                       )

search3.fit(x_train, y_train)

# Save the best parameter
search3.best_params_['activation']
search3.best_params_['hidden_layer_sizes']
search3.best_params_['solver']
search3.best_score_
prediction3 = search3.predict(x_test)
accuracy3 = sum(prediction3 == y_test)/(len(x_test))
accuracy3
pd.DataFrame(confusion_matrix(prediction3, y_test),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

####################################################################

data1 = pd.read_csv('./flood_tool/classification/balanced_data.csv')
data1 = data1.dropna()

# With balanced data
y_res1 = data1.riskLabel
data1 = data1[['easting', 'northing']]

scorer = make_scorer(recall_score, average='macro')
# #applied smote fct to balance the data
# #data = smote.smote_func('./flood_tool/resources/postcodes_sampled.csv')

# split the data
x_train1, x_test1, y_train1, y_test1 = train_test_split(data1,
                                                        y_res1,
                                                        test_size=0.3,
                                                        random_state=42)

# Scale the data for Knn
sc1 = RobustScaler()
sc1.fit(x_train1)
X_train1 = sc.transform(x_train1)
X_test1 = sc.transform(x_test1)

# Knn
# Hyperparameter Grid
grid4 = {'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]}
# Instanciate Grid Search
search4 = GridSearchCV(KNeighborsClassifier(), grid4,
                       scoring=scorer,
                       cv=5,
                       n_jobs=-2  # paralellize computation
                       )

search4.fit(X_train1, y_train1)

# Save the best parameter
search4.best_params_['n_neighbors']
search4.best_score_
prediction4 = search4.predict(X_test1)
accuracy4 = sum(prediction4 == y_test1)/(len(X_test1))
accuracy4
pd.DataFrame(confusion_matrix(prediction4, y_test1),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# RandomForest
# Hyperparameter Grid
grid5 = {'n_estimators': [50, 100],
         'max_depth': [2, 10]}
# Instanciate Grid Search
search5 = GridSearchCV(RandomForestClassifier(), grid5,
                       scoring=scorer,
                       cv=5,
                       n_jobs=-2  # paralellize computation
                       )
search5.fit(x_train1, y_train1)

# Save the best parameter
search5.best_params_['n_estimators']
search5.best_params_['max_depth']
search5.best_score_
prediction5 = search2.predict(x_test1)
accuracy5 = sum(prediction5 == y_test1)/(len(x_test1))
accuracy5
pd.DataFrame(confusion_matrix(prediction5, y_test1),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# NN
# Hyperparameter Grid
grid6 = {'hidden_layer_sizes': [(20, 30, 20), (10, 50, 10), (10, 10, 10)],
         'activation': ['relu', 'logistic'],
         'solver': ['lgbs', 'sgd', 'adam']
         }
# Instanciate Grid Search
search6 = GridSearchCV(MLPClassifier(), grid6,
                       scoring=scorer,
                       cv=5,
                       n_jobs=-2  # paralellize computation
                       )

search6.fit(x_train1, y_train1)

# Save the best parameter
search6.best_params_['activation']
search6.best_params_['hidden_layer_sizes']
search6.best_params_['solver']

prediction6 = search6.predict(x_test1)
accuracy6 = sum(prediction6 == y_test1)/(len(x_test1))
accuracy6
pd.DataFrame(confusion_matrix(prediction6, y_test1),
             columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

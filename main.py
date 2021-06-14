from metrics import Metrics, Error

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

try:
    data = load_breast_cancer() 
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    yhat_prob = clf.predict_proba(X_test)
    metr = Metrics(y_test, yhat, yhat_prob[:, 0])
    print(f'Accuracy: {metr.accuracy_score()}')
    print(f'Precision score: {metr.precision_score()}')
    print(f'Recall score: {metr.recall_score()}')
    print(f'F1 score: {metr.fbeta_score()}')
    print(f'Log loss: {metr.log_loss()}')
    print(f'Roc Auc: {metr.roc_auc_score()}')

    X_test_reg, y_test_reg = make_regression()
    reg = LinearRegression()
    reg.fit(X_test_reg, y_test_reg)
    yhat_reg = reg.predict(X_test_reg)
    metr_reg = Metrics(y_test_reg, yhat_reg, task='Regression')
    print(f'MAE: {metr_reg.mean_absolute_error()}')
    print(f'MSE: {metr_reg.mean_squared_error()}')
    print(f'RMSE: {metr_reg.mean_squared_error(squared=False)}')
    print(f'R2 score: {metr_reg.r2_score()}')
    
except Error as err:
    print(f'Error: {err}')

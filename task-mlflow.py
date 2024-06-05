import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
experiment_name = "iris_classifn"

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name)

def objective(trail):
    n_estimators = trail.suggest_int('n_estimator',50,200)
    max_depth = trail.suggest_int('max_depth',2,10)
    min_sample_split = trail.suggest_int('min_sample_split',2,10)

    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_sample_split=min_sample_split)

    clf.fit(X_train,y_train)

    pred = clf.predict(X_test)

    accuracy_score = accuracy_score(y_test,pred)

    with mlflow.start_run():
        mlflow.log_param("model","Random Forest")
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("min_sample_split",min_sample_split)
        mlflow.log_param("accuracy_score",accuracy_score)

        mlflow.log_metric("precision",)

    return accuracy_score


        
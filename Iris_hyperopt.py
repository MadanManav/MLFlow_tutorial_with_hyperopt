from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime


def loading_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=10)
    split_data = (Xtrain, Xtest, ytrain, ytest)
    return split_data


def objective(params, data):
    X_train, X_test, y_train, y_test = data
    with mlflow.start_run(run_name="HPO with hyperopt"+ str(datetime.now())):
        clf = MLPClassifier(random_state=42,
                            max_iter=1000,
                            activation=params['activation'],
                            hidden_layer_sizes=params['hidden_layer_sizes'],
                            solver=params['solver']
                            )

        for k, v in params.items():  # logging each configuration
            mlflow.log_param(k, v)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_f1_score', test_f1_score)
        mlflow.sklearn.log_model(clf, "model")  # command to serialize the model (as 2 flavours)
        print('test_accuracy', test_accuracy)
    return {'loss': -test_accuracy, 'status': STATUS_OK}


def train():
    search_space = {
        'activation': hp.choice('activation', ['logistic', 'tanh', 'relu']),
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50, 50, 50), (50, 100, 50), (50, 1000, 100, 50)]),
        'solver': hp.choice('solver', ['sgd', 'adam'])
    }
    algo = tpe.suggest
    trials = Trials()
    data_set = loading_dataset()
    fmin_objective = partial(objective, data=data_set)
    argmin = fmin(
        fn=fmin_objective,
        space=search_space,
        algo=algo,
        max_evals=100,
        verbose=True,
        trials=trials
    )
    columns = list(search_space.keys())
    print('columns \n', columns)
    results = pd.DataFrame(columns=['iteration'] + columns + ['loss'])
    print('results \n', results)
    for idx, trial in enumerate(trials.trials):
        row = [idx]
        translated_eval = space_eval(search_space, {k: v[0] for k, v in trial['misc']['vals'].items()})
        for k in columns:
            row.append(translated_eval[k])
        row.append(trial['result']['loss'])
        results.loc[idx] = row
    results['loss'] = results['loss'].abs()
    results.plot(x="iteration", y="loss")
    plt.show()

def evaluate():
    #  this will be chnaged
    logged_model = 'file:///C:/Users/name/PycharmProjects/MLFlow_trial/mlruns/1/2902630408434b0d86415eec8e047e5b/artifacts/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    dummy_value = [6.1, 1.2, 3.4, 4.3]
    predictions = loaded_model.predict([dummy_value])
    print(predictions)

if __name__ == "__main__":
    '''
    With mlproject, there is currently a bug (https://github.com/mlflow/mlflow/issues/2735) due to which experiment can't be set from inside the python script
    '''
    # experiment_id = mlflow.create_experiment("MlpClassifier-Iris")  # create an experiment and it returns an ID
    # mlflow.set_experiment("MlpClassifier-Iris")  # set the created experiment as an active experiment
    mlflow.set_tracking_uri("sqlite:///mlruns.db")  # setting the backend store to a sqlite database for registry
    mlflow.sklearn.autolog()
    np.random.seed(12)
    #  train(experiment_id)
    train ()    
    #  evaluate()  # write the path in the function

# MLFlow_tutorial_with_hyperopt
This repository contains example of using MLFlow with sklearn. This also showcases how to use hyperopt library. One dissadvantage is that every run is saved indivdually not under a single main experiment.

## MLflow tracking:
Mlflow have "experiments" as outer group in which runs are saved. This is useful when we have to compare runs which are intened for the same task.
You can pass experiment name for each individual run. If no experiment is defined then all the runs are saved under Default experiment.   
In order to use mlflow registry this example uses runs backend (storage location) as an sqlite database whereas the artifacts are saved locally. 
This means that the meta files for the runs are read from database (mlruns.db) and artifacts are read from ./mlruns folder

- Step1) To restart this example, delete the ./mlruns folder and the database mlruns.db
and then run python Iris_hyperopt.py (assuming that you are in a conda enviornment which was created by running 'conda env create -f conda.yaml')

- Step2) Run the mlflow server/ UI to see the runs: 
'mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns'
There will be two sections i.e. experiments and models. In the experiment section under your defined experiment you will find all the iterations of the hyperopt.

## MLflow Models:
It is used to store the trained model. It consitutes of a file describing the environment (in which the model instance was created in), and a descriptor file that lists several “flavors” the model can be used in.
When you log your model (mlflow.sklearn.log_model) a MLmodel file is created where different flavors are specified in which models are saved. You also define dependency for that model (conda or docker). 
MLflow Models can be used to deploy the model to a simple HTTP server.

## MLflow projects:
Each mlflow project is a directory of files, or a Git repository, containing the code and data.
You have to create the mlproject file (YAML formatted)
In the file you specify (1) the name of the project, (2) which Conda/Docker/System enviornment the dependencies should be loaded from and finally (3) the entry points (Commands that can be run within the project, and information about their parameters). 
In the entry point there has to be a "main" keyword where the project starts from, you can specify multiple scirpts that can be executed sequentially to create a workflow (see https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/MLproject).

###### To use MLflow projet: 
- Step 1) Go to the directory where you have cloned this repo as you will find the conda.yaml and mlproject file.
- Step 2) Delete the ./mlruns folder and the database mlruns.db
- Step 3) Create a new conda enviornment (conda create -n new_env python=3.7)
- Step 4) Activate the enviornment (conda activate new_env)
- Step 5) install pip (conda install pip)
- Step 6) install mlflow (pip install mlflow==1.14.1)
- Step 7) Run from the directory where project is by using command (mflow run .) or if you have defined input paramters then Run (mlflow run . -P name_of_param_1=20 -P name_of_param_2=50)

## MLflow Registry:
Enables versioning of models, makes it possible to tag models such as "Staging" or in "Production". Model registry requires setting up a database.
You can register a model either from UI or from python script by passing a name for the argument 'registered_model_name' in mlflow.sklearn.log_model


# links

- https://www.mlflow.org/docs/latest/index.html
- http://hyperopt.github.io/hyperopt/

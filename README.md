# Mushroom MLOps

An end-to-end MLOps pipeline for mushroom classification.

The project uses the Kaggle Binary Prediction of Poisonous Mushrooms dataset. ```https://www.kaggle.com/competitions/playground-series-s4e8/data?select=train.csv```

The project implements the complete machine learning lifecycle:

1. Data preprocessing
2. Model training and evaluation
3. Model deployment through a REST API
4. Web application for predictions
5. Automated orchestration with Apache Airflow
6. Experiment tracking with MLflow
7. Docker-based deployment

The complete pipeline is automatically executed every 5 minutes.

---

## Project Architecture

```text
                         Apache Airflow
                              |
                    every 5 minutes
                              |
                              v
                    +-------------------+
                    |  Data Processing  |
                    | preprocess.py     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Model Training    |
                    | CatBoost          |
                    | Evaluation        |
                    | MLflow            |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Docker Deployment |
                    +---------+---------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        +----------------+        +----------------+
        |    FastAPI     |        |   Streamlit    |
        |      API       | <----  |      App       |
        |   port 8000    |        |   port 8501    |
        +----------------+        +----------------+
```


---

# 1. Data Engineering

The data engineering stage processes the raw mushroom dataset and creates the processed training and test datasets.

The preprocessing pipeline is located at:

```text
code/datasets/preprocess.py
```
The preprocessing pipeline removes duplicate rows, handles missing values, removes numerical outliers using the IQR method, and performs a stratified 80/20 train-test split.

The resulting datasets are stored in:

```text
data/processed/
├── train.csv
└── test.csv
```

The datasets are versioned with DVC.
The data is checked using:

```text
code/datasets/check_data.py
```

---

# 2. Model Engineering

The model training pipeline is located at:

```text
code/models/train.py
```

The project uses a **CatBoostClassifier**, which is suitable for the categorical features present in the mushroom dataset.

The training pipeline:

1. Loads the processed training and test datasets.
2. Creates a training sample of 100,000 rows.
3. Detects categorical features.
4. Trains a CatBoost classification model.
5. Evaluates the model on the test dataset.
6. Calculates classification metrics.
7. Logs parameters and metrics to MLflow.
8. Saves the trained model.

The main evaluation metrics are:

* MCC
* Accuracy
* Precision
* Recall
* F1-score

The trained model is stored at:

```text
models/mushroom_catboost.cbm
```

MLflow tracks model parameters, evaluation metrics, training time, model artifact.


---

# 3. Deployment

The deployment consists of two separate Docker containers:

```text
FastAPI API
    |
    | port 8000
    |
    +---- Mushroom model

Streamlit application
    |
    | port 8501
    |
    +---- FastAPI API
```

The deployment configuration is located at:

```text
code/deployment/docker-compose.yml
```

## API

The REST API is implemented with FastAPI:

```text
code/deployment/api/main.py
```

The API container uses:

```text
code/deployment/api/Dockerfile
```

The API is available at:

```text
http://localhost:8000
```

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Web Application

The user interface is implemented with Streamlit:

```text
code/deployment/app/streamlit_app.py
```

The application is containerized separately from the API.

The Streamlit application is available at:

```text
http://localhost:8501
```

The application provides input fields, a prediction action, and displays the model prediction.

---

# 4. Airflow Automation

Apache Airflow is used to orchestrate the complete MLOps pipeline.

The DAG is located at:

```text
services/airflow/dags/mushroom_pipeline.py
```

The pipeline consists of three tasks:

```text
preprocess_data
       |
       v
train_and_evaluate
       |
       v
deploy
```

The DAG is scheduled using:

```text
*/5 * * * *
```

Therefore, the complete pipeline is automatically executed every 5 minutes.

The deployment task uses Docker Compose to build and start the API and Streamlit containers.

The Airflow environment is defined in:

```text
airflow-docker-compose.yml
```

Airflow is available at:

```text
http://localhost:8080
```

---

# 5. Running the Project

## Start Airflow

From the project root:

```bash
docker compose -f airflow-docker-compose.yml up -d --build
```

Airflow UI:

```text
http://localhost:8080
```

The DAG:

```text
mushroom_mlops_pipeline
```

automatically executes every 5 minutes.

---

## Start the Deployment Manually

The deployment can also be started manually:

```bash
docker compose -f code/deployment/docker-compose.yml up -d --build
```

This starts:

```text
mushroom-api
mushroom-app
```

The services are then available at:

```text
API:      http://localhost:8000
Swagger:  http://localhost:8000/docs
Streamlit:http://localhost:8501
```

---

# 6. Docker Containers

The project uses separate containers for the machine learning services:

```text
mushroom-api
mushroom-app
mushroom-airflow
```

Check running containers with:

```bash
docker ps
```

Stop the deployment containers with:

```bash
docker stop mushroom-api mushroom-app
```

Stop Airflow with:

```bash
docker stop mushroom-airflow
```

---

# 7. Complete Automated Pipeline

The final automated workflow is:

```text
Raw Data
   |
   v
Data Preprocessing
   |
   v
Processed Train/Test Data
   |
   v
CatBoost Training
   |
   v
Model Evaluation
   |
   v
MLflow Logging
   |
   v
Saved CatBoost Model
   |
   v
Docker Build
   |
   +-------------------+
   |                   |
   v                   v
FastAPI             Streamlit
API                 Web App
```

Airflow executes this workflow automatically every 5 minutes.

---

# 8. Technologies

* Python
* Pandas
* Scikit-learn
* CatBoost
* MLflow
* FastAPI
* Streamlit
* Docker
* Docker Compose
* Apache Airflow


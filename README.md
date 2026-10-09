
# ML Experiment Tracking with MLflow

## Project Overview

This project demonstrates machine learning experiment tracking
using MLflow on the California Housing dataset.

The objective is to train a regression model using three different
hyperparameter configurations and identify the best-performing
model based on validation RMSE.

## Technologies Used

- Python
- Scikit-learn
- MLflow
- Pandas
- NumPy

## Dataset

The California Housing dataset contains 20,640 samples.

- Training data: 80%
- Validation data: 20%
- Random seed: 42

## Machine Learning Model

HistGradientBoostingRegressor

## Experiment Configurations

| Run | Max Depth | Learning Rate |
|-----|-----------|---------------|
| Run 1 | 3 | 0.1 |
| Run 2 | 5 | 0.05 |
| Run 3 | 7 | 0.01 |

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/mlflow-task.git
cd mlflow-task
```

2. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies.

```bash
python -m pip install -r requirements.txt
```

4. Start the MLflow tracking server.

```bash
mlflow server --host 127.0.0.1 --port 5000
```

5. Open a second terminal and run:

```bash
source .venv/bin/activate
python train.py
```

6. View the experiments in your browser.

http://127.0.0.1:5000

## Experiment Comparison

![Run Comparison](screenshots/05-run-comparison.png)

| Run | Max Depth | Learning Rate | RMSE | MAE | R2 |
|-----|-----------|---------------|------|-----|----|
| Run 1 | 3 | 0.1 | 0.5406377209961797 | 0.37018343505047957 | 0.7769481511365148 |
| Run 2 | 5 | 0.05 | 0.52218231619472 | 0.35488950625051596 | 0.7919165879704807 |
| Run 3 | 7 | 0.01 | 0.7213234100524357 | 0.5509052755164383 | 0.6029426915342583 |

## Best Model

Selected run: 2

Max Depth: 5

Learning Rate: 0.05

Validation RMSE: 0.52218231619472

## Why This Model Was Selected

The selected model achieved the lowest RMSE on the validation
dataset among the three experiments.

RMSE was the primary selection metric because the assignment
requires minimizing the magnitude of prediction errors,
with larger errors receiving a greater penalty.

## Conclusion

This project demonstrates the complete machine learning
experiment workflow:

Train -> Track -> Compare -> Select

MLflow makes it possible to record model parameters,
validation metrics, and trained models, allowing different
configurations to be compared systematically.
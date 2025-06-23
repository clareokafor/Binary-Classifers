# Supervised Learning Methods for Binary Classification

## Overview

This project evaluates and compares three supervised learning algorithms on a binary classification problem. The goal is to classify a dataset with two classes (class 0 and class 1) using:

- **K Nearest Neighbour (KNN)**
- **Decision Tree Classifier**
- **Logistic Regression**

## Project Description

The project's primary objective is to understand both the benefits and limitations of each classification method. Each classifier underwent hyperparameter tuning—using GridSearchCV (which incorporates K-Fold cross-validation by default)—to identify the best-performing parameters. The evaluation was based on the following metrics:
  
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**

Among the three methods, KNN was found to perform best overall on the dataset, though each method shows its own strengths and weaknesses.

## Dependencies

The project is implemented in Python. Key libraries include:

- **pandas** – Data manipulation and analysis
- **numpy** – Mathematical operations and array manipulation
- **matplotlib** and **seaborn** – Data visualization
- **scikit-learn** – Machine learning library for model building, hyperparameter tuning, and evaluation

## Hyperparameter Tuning Details

- **K Nearest Neighbour (KNN):**
  - Tuned hyperparameters include the number of neighbours (range: 5 to 10), weight options (`uniform` and `distance`), and distance metrics (`euclidean` and `manhattan`).

- **Decision Tree:**
  - Hyperparameters tuned are maximum depth (range: 3 to 10) and minimum samples split (range: 5 to 50).

- **Logistic Regression:**
  - Tuned using the regularization parameter (C; range: 0.001 to 10), solvers (`lbfgs` and `liblinear`), and setting max iterations (default: 1000).

## Evaluation & Results

The evaluation metrics recorded for each classifier are:

- **KNN (best model parameters for k = 7, 8, or 9)**:
  - Precision: 0.9784
  - Recall: 0.9729
  - F1 Score: 0.9756
  - Accuracy: 0.9786

- **Decision Tree (using maximum depths m = 3 and m = 5)**:
  - The model with max depth 5 achieved slightly better performance and fewer misclassifications than the m = 3 version.

- **Logistic Regression (using regularization parameters c = 0.1 and c = 0.001)**:
  - Both configurations yielded identical evaluation metrics.

Overall, KNN was the best performer on this dataset, even though each model carries its own applicability and limitations.

## Conclusions

This project not only demonstrates the implementation of several classification algorithms but also highlights the importance of model selection and hyperparameter tuning. The experience gained in performing cross-validation, tailoring hyperparameters, and evaluating model performance provides a solid foundation for tackling more complex or multi-class classification problems in the future.

## License

This project is licensed under the [MIT License](https://github.com/clareokafor/Binary-Classifers?tab=MIT-1-ov-file).

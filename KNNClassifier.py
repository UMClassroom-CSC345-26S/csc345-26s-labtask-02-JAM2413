#!/usr/bin/env python3.11

#----K-Nearest Neighbours (KNN) Classification for Car Style Prediction
#----Task:
#----1) Remove non-ordinal (non-numeric) features
#----2) Normalize ordinal (numeric) features
#----3) Randomly split into 80% training and 20% testing
#----4) Train KNN by iterating K to find the highest accuracy
#----5) Save:
#----   Training.csv  (normalized ordinal features + Style)
#----   Testing.csv   (normalized ordinal features + Style + Prediction + Confidence)
#----   Accuracy.csv  (columns: K, Accuracy)
#--------------------------------------------------------------------------------------------------

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#--------------------------------------------------------------------------------------------------
def get_data(filename, target_column):
    #----Read the dataset as a pandas dataframe
    dataset = pd.read_csv(filename)

    #----Separate target (label) from features
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns are: {list(dataset.columns)}")

    y = dataset[target_column]
    X = dataset.drop(columns=[target_column])

    #----Remove non-ordinal features: keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric (ordinal) feature columns found after removing non-ordinal columns.")

    X_ord = X[numeric_cols].copy()

    return X_ord, y, numeric_cols

#--------------------------------------------------------------------------------------------------
def split_and_normalize(X_ord, y, test_size=0.2, random_state=42):

    #----Random split into training and testing sets (stratify when possible)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_ord, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    #----Handle missing values:
    #----Fill with TRAIN medians only
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    #----Normalize ordinal features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------
def iterate_k_and_train(X_train_scaled, y_train, X_test_scaled, y_test, k_max=75):

    #----Try K = 1..k_max and record accuracy for each K
    k_max = min(int(k_max), len(X_train_scaled))
    accuracy_list = []

    best_k = None
    best_accuracy = -1.0
    best_model = None

    for k in range(1, k_max + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, pred))

        accuracy_list.append([k, acc])

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            best_model = model

    return best_model, best_k, best_accuracy, accuracy_list

#--------------------------------------------------------------------------------------------------
def save_outputs(numeric_cols, X_train_scaled, y_train, X_test_scaled, y_test,
                 best_model, accuracy_list,
                 training_file="Training.csv", testing_file="Testing.csv", accuracy_file="Accuracy.csv"):

    #----Accuracy.csv
    accuracy_df = pd.DataFrame(accuracy_list, columns=["K", "Accuracy"])
    accuracy_df.to_csv(accuracy_file, index=False)

    #----Training.csv
    training_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
    training_df["Style"] = y_train.to_numpy()
    training_df.to_csv(training_file, index=False)

    #----Testing.csv
    testing_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)
    testing_df["Style"] = y_test.to_numpy()

    prediction = best_model.predict(X_test_scaled)
    proba = best_model.predict_proba(X_test_scaled)
    confidence = proba.max(axis=1)

    testing_df["Prediction"] = prediction
    testing_df["Confidence"] = confidence
    testing_df.to_csv(testing_file, index=False)

#--------------------------------------------------------------------------------------------------
def main():

    #----Paramaeters
    input_csv = "AllCars.csv"
    target_column = "Style"
    test_size = 0.2
    random_state = 42
    k_max = 75

    X_ord, y, numeric_cols = get_data(input_csv, target_column)

    X_train_scaled, X_test_scaled, y_train, y_test = split_and_normalize(
        X_ord, y, test_size=test_size, random_state=random_state
    )

    best_model, best_k, best_accuracy, accuracy_list = iterate_k_and_train(
        X_train_scaled, y_train, X_test_scaled, y_test, k_max=k_max
    )

    save_outputs(
        numeric_cols,
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        best_model,
        accuracy_list,
        training_file="Training.csv",
        testing_file="Testing.csv",
        accuracy_file="Accuracy.csv"
    )

    print(f"Best K = {best_k}")
    print(f"Highest Accuracy = {best_accuracy:.4f}")
    input("Press Enter to end")

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------

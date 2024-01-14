"""
Module contains functions for predicting wine quality through data processing and model training.

Functions:
    - train_model: Trains RandomForestRegressor model on provided data.
    - low_correlation_columns: Removes columns with low correlation with the target column.
    - high_correlation_columns: Removes columns with high correlation with each other.
    - check_correlation: Calculates correlation between columns and the target column.
    - train_all_models: Trains models using different variants of the dataset.

Author: Oksana Matiazh
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("winequality-red.csv", sep=";")
target_column = "quality"


def check_correlation(df, target_column):
    columns_to_correlate = df.columns.difference([target_column])
    correlations = df[columns_to_correlate].corrwith(df[target_column])
    return correlations


def low_correlation_columns(df, target_column, threshold=0.17):
    correlations = check_correlation(df, target_column)
    selected_columns = [
        col
        for col in correlations.index
        if col != target_column and abs(correlations[col]) >= threshold
    ]
    df_selected = df[selected_columns].copy()
    df_selected[target_column] = df[target_column].values
    return df_selected


def high_correlation_columns(df, target_column, threshold=0.4):
    correlation_matrix = df.drop(columns=[target_column]).corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    correlated_columns = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    while correlated_columns and df.shape[1] > 2:
        column_to_remove = np.random.choice(df.columns[df.columns != target_column])
        df = df.drop(columns=column_to_remove, axis=1)
        correlation_matrix = df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        correlated_columns = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
    return df


def train_model(df, target_column):
    # Then the function should split the data into X, y.
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Then split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Then scale the data appropriately (without leakage).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Then train a RandomForestRegressor model on the scaled data.
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Then predict the test data on the trained model.
    y_pred = model.predict(X_test_scaled)

    # Compare predicted data with actual and display the mean_absolute_error metric for them.
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nMean Absolute Error: {mae}\n")


def train_all_models(df, target_column="quality"):
    print("\nFull Dataset:")
    train_model(df, target_column)

    print("\nDataset for PCA- 1 feature:")
    pca_1 = PCA(n_components=1)
    df_pca_1 = pd.DataFrame(
        pca_1.fit_transform(df.drop(target_column, axis=1)), columns=["PCA1"]
    )
    df_pca_1[target_column] = df[target_column]
    train_model(df_pca_1, target_column)

    print("\nDataset for PCA- 2 features:")
    pca_2 = PCA(n_components=2)
    df_pca_2 = pd.DataFrame(
        pca_2.fit_transform(df.drop(target_column, axis=1)), columns=["PCA1", "PCA2"]
    )
    df_pca_2[target_column] = df[target_column]
    train_model(df_pca_2, target_column)

    print("\nDataset for PCA- 4 features:")
    pca_4 = PCA(n_components=4)
    df_pca_4 = pd.DataFrame(
        pca_4.fit_transform(df.drop(target_column, axis=1)),
        columns=["PCA1", "PCA2", "PCA3", "PCA4"],
    )
    df_pca_4[target_column] = df[target_column]
    train_model(df_pca_4, target_column)

    print("\nDataset after removing columns with low correlation:")
    low_corr_df = low_correlation_columns(df, target_column)
    train_model(low_corr_df, target_column)

    print("\nDataset after removing randomly high correlated columns:")
    high_corr_df = high_correlation_columns(df, target_column)
    train_model(high_corr_df, target_column)


train_all_models(df)

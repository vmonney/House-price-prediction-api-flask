"""Utility functions for the API."""


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for the model."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        self.numerical_features = ["HouseAge", "DistanceToStation", "NumberOfPubs"]
        self.categorical_features = ["PostCode"]
        self.date_feature = "TransactionDate"

    def fit(self, x: pd.DataFrame) -> "Preprocessor":
        """Fit the preprocessor to the data.

        Args:
        ----
            x (pd.DataFrame): The input data.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
        -------
            Preprocessor: The fitted preprocessor.
        """
        # Define and fit the pipeline for numerical features
        self.num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())],
        )
        self.num_pipeline.fit(x[self.numerical_features])

        # Fit OneHotEncoder for categorical features
        self.onehot = OneHotEncoder(handle_unknown="ignore")
        self.onehot.fit(x[self.categorical_features])

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        num_features = self.num_pipeline.transform(x[self.numerical_features])
        num_df = pd.DataFrame(
            num_features,
            columns=self.numerical_features,
            index=x.index,
        )

        # Transform categorical features
        onehot_features = self.onehot.transform(x[self.categorical_features])
        onehot_df = pd.DataFrame(
            onehot_features.toarray(),
            columns=self.onehot.get_feature_names_out(self.categorical_features),
            index=x.index,
        )

        # Extract year and month from TransactionDate
        transformed_df = x.copy()
        transformed_df["Year"] = pd.to_datetime(
            transformed_df[self.date_feature],
            format="%Y.%m",
        ).dt.year
        transformed_df["Month"] = pd.to_datetime(
            transformed_df[self.date_feature],
            format="%Y.%m",
        ).dt.month
        transformed_df = transformed_df.drop(self.date_feature, axis=1)

        # Combine all features
        return pd.concat(
            [
                transformed_df.drop(
                    self.numerical_features + self.categorical_features,
                    axis=1,
                ),
                num_df,
                onehot_df,
            ],
            axis=1,
        )

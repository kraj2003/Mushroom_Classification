from src.mushroom.exceptions.exception import ClassificationException
from src.mushroom.logging import logging
from sklearn.model_selection import train_test_split
from src.mushroom.entity.config_entity import DataTransformationConfig
import pandas as pd
import numpy as np
from src.mushroom.utils.common import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import sys
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def read_data(self, file_path) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise ClassificationException(e, sys)

    def get_transformer_object(self):
        """
        Initializes a transformer pipeline for categorical features only.
        """
        logging.info("Entered get_transformer_object method of DataTransformation class")
        try:
            # Read the dataset
            data = pd.read_csv(self.config.data_file)

            # Identify all categorical features except the target column 'class'
            cat_feature = data.select_dtypes(include='object').drop(columns=['class'], errors='ignore').columns.tolist()
            logging.info(f"Categorical features identified: {cat_feature}")

            # Create a pipeline for categorical data
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False))
                ]
            )

            # Build the column transformer with only categorical pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_pipeline", cat_pipeline, cat_feature)
                ]
            )

            logging.info("Preprocessor for categorical features created successfully.")
            return preprocessor

        except Exception as e:
            raise ClassificationException(e, sys)


    def train_test_splitting(self):
        try:
            data = pd.read_csv(self.config.data_file)
            train, test = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Train and test data split successfully.")

            preprocessor_obj = self.get_transformer_object()

            target_column = data['class']
            dependent_features = data.drop(columns=['class'], axis=1)

            input_feature_train_df = train.drop(columns=['class'], axis=1)
            target_feature_train_df = train[['class']]

            input_feature_test_df = test.drop(columns=['class'], axis=1)
            target_feature_test_df = test[['class']]   

            logging.info(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")

            # Fit and transform training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Reshape the target data to ensure correct concatenation
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            logging.info(f"Transformed input train shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed input test shape: {input_feature_test_arr.shape}")

            # Concatenate the features and target
            if input_feature_train_arr.shape[0] == target_feature_train_arr.shape[0]:
                train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
                test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            else:
                logging.error("Dimension mismatch between features and target. Check the sizes.")
                raise ValueError("Feature and target dimensions don't match.")
            
            np.save(os.path.join(self.config.root_dir, "train.npy"), train_arr)
            np.save(os.path.join(self.config.root_dir, "test.npy"), test_arr)
            # Save the preprocessor correctly
            save_object("final_model/preprocessor.pkl", preprocessor_obj)

            return (
                train_arr,
                test_arr,
                self.config.preprocessor
            )
        except Exception as e:
            raise ClassificationException(e, sys)


# At the bottom of data_transformation.py

import numpy as np
import pandas as pd
from src.mushroom.utils.common import load_object
from sklearn.exceptions import NotFittedError
import logging

def validate_transformation(train_arr, test_arr, data_path, preprocessor_path):
    """
    Validates the data transformation pipeline and logs each step for traceability.
    """

    logging.info("🔍 Starting data transformation validation...")

    try:
        # Load original data
        df = pd.read_csv(data_path)
        logging.info(f"📂 Original data loaded from: {data_path}, shape: {df.shape}")

        # Load preprocessor object
        preprocessor = load_object(preprocessor_path)
        logging.info(f"📦 Preprocessor loaded from: {preprocessor_path}")

        # Identify categorical features (excluding target column)
        cat_features = df.select_dtypes(include='object').drop(columns=['class'], errors='ignore').columns.tolist()

        # Check if preprocessor is fitted
        try:
            preprocessor.transform(df[cat_features].iloc[:5])
            logging.info("✅ Preprocessor is fitted and successfully transforms sample categorical data.")
        except NotFittedError:
            raise ValueError("🚨 Preprocessor is not fitted.")

        # Check shapes after splitting
        expected_train_rows = int(df.shape[0] * 0.8)
        expected_test_rows = df.shape[0] - expected_train_rows

        assert train_arr.shape[0] == expected_train_rows, \
            f"Expected {expected_train_rows} rows in training set, got {train_arr.shape[0]}"
        assert test_arr.shape[0] == expected_test_rows, \
            f"Expected {expected_test_rows} rows in test set, got {test_arr.shape[0]}"

        logging.info(f"✅ Train/Test split sizes: {train_arr.shape[0]} train, {test_arr.shape[0]} test.")

        # Check feature count
        num_train_features = train_arr.shape[1] - 1  # exclude target
        num_test_features = test_arr.shape[1] - 1
        assert num_train_features == num_test_features, \
            f"Feature mismatch: {num_train_features} train vs {num_test_features} test"

        logging.info(f"✅ Feature shape check passed: {num_train_features} features.")

        # Check target distribution
        train_target_counts = pd.Series(train_arr[:, -1]).value_counts()
        test_target_counts = pd.Series(test_arr[:, -1]).value_counts()

        logging.info(f"🎯 Train class distribution:\n{train_target_counts}")
        logging.info(f"🎯 Test class distribution:\n{test_target_counts}")

        if not np.isclose(train_target_counts.sum(), test_target_counts.sum()):
            logging.warning("⚠️ Class distribution may be imbalanced between train and test.")

        # Check for missing or NaN values in numeric data
        if not np.issubdtype(train_arr.dtype, np.number) or not np.issubdtype(test_arr.dtype, np.number):
            logging.warning("⚠️ Non-numeric dtypes detected. Skipping missing value check.")
        else:
            assert np.all(np.isfinite(train_arr)), "❌ NaN/inf in train array"
            assert np.all(np.isfinite(test_arr)), "❌ NaN/inf in test array"
            logging.info("✅ No missing/invalid numeric values in transformed arrays.")

        # Range check (optional)
        if np.issubdtype(train_arr.dtype, np.number):
            for i in range(train_arr.shape[1] - 1):
                train_min, train_max = train_arr[:, i].min(), train_arr[:, i].max()
                test_min, test_max = test_arr[:, i].min(), test_arr[:, i].max()

                logging.debug(f"Feature[{i}] train: ({train_min:.3f}, {train_max:.3f}), test: ({test_min:.3f}, {test_max:.3f})")

                if not (-10 <= train_min <= 10 and -10 <= train_max <= 10):
                    logging.warning(f"🔍 Unexpected range in train feature[{i}]: ({train_min}, {train_max})")
                if not (-10 <= test_min <= 10 and -10 <= test_max <= 10):
                    logging.warning(f"🔍 Unexpected range in test feature[{i}]: ({test_min}, {test_max})")

        logging.info("✅ All transformation validation checks passed.")
        logging.info(f"📏 Final transformed shapes: train {train_arr.shape}, test {test_arr.shape}")

    except Exception as e:
        logging.error(f"❌ Transformation validation failed: {e}")
        raise ClassificationException(e, sys)

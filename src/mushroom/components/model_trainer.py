from src.mushroom.exceptions.exception import ClassificationException
from src.mushroom.logging import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.mushroom.utils.common import *
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from src.mushroom.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config=config

    
    def load_data(self,train_path : str, test_path: str):
        train_arr = np.load(train_path, allow_pickle=True)
        test_arr = np.load(test_path, allow_pickle=True)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        return X_train, X_test, y_train, y_test

    def train_and_save(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data(self.config.train_arr,self.config.test_arr)

            clf = RandomForestClassifier(
                n_estimators=int(self.config.n_estimators),
                criterion=self.config.criterion,
                random_state=42
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            model_path = self.config.root_dir , self.config.model_name
            os.makedirs(self.config.root_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(self.config.root_dir, self.config.model_name))

            save_json(Path(self.config.root_dir) / "metrics.json", {
                "accuracy": acc,
                "report": report
            })

            print(f"✅ Model saved at: {model_path}")
            print(f"📊 Accuracy: {acc:.4f}")
        except Exception as e:
            raise ClassificationException(e,sys)

from src.mushroom.exceptions.exception import ClassificationException
from src.mushroom.logging import logging
from src.mushroom.utils.common import *
import os
import sys
from src.mushroom.config.configuration import ConfigurationManager
from src.mushroom.components.model_trainer import ModelTrainer


STAGE_NAME="Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_training=ModelTrainer(config=model_trainer_config)
        model_training.train_and_save()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.initiate_model_training()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise ClassificationException(e,sys)
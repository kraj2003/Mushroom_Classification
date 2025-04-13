from src.mushroom.config.configuration import ConfigurationManager
from src.mushroom.components.data_validation import Data_validation
from src.mushroom.logging import logging
import sys
from src.mushroom.exceptions.exception import ClassificationException

STAGE_NAME="Data Ingestion Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config=ConfigurationManager()
        data_validation_config=config.get_data_validaion_config()
        data_validation=Data_validation(config=data_validation_config)
        data_validation.validate_data()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise ClassificationException(e,sys)
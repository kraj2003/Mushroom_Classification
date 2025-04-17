from src.mushroom.config.configuration import ConfigurationManager
from src.mushroom.components.data_transformation import DataTransformation
from src.mushroom.logging import logging
from src.mushroom.components.data_transformation import validate_transformation
from src.mushroom.exceptions.exception import ClassificationException
import sys

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_Transformation=DataTransformation(config=data_transformation_config)
        train_arr, test_arr, preprocessor_path=data_Transformation.train_test_splitting()
        validate_transformation(
            train_arr=train_arr,
            test_arr=test_arr,
            data_path=data_transformation_config.data_file,
            preprocessor_path=preprocessor_path
        )


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise ClassificationException(e,sys)
from src.mushroom.logging import logging
from src.mushroom.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.mushroom.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.mushroom.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.mushroom.pipeline.model_trainer_pipeline import ModelTrainingPipeline
from src.mushroom.exceptions.exception import ClassificationException
import sys



STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.initiate_data_validation()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise ClassificationException(e,sys)

STAGE_NAME = "Data Transformation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataTransformationTrainingPipeline()
   data_validation.initiate_data_transformation()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise ClassificationException(e,sys)

STAGE_NAME = "Model Training stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.initiate_model_training()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise ClassificationException(e,sys)
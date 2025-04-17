from src.mushroom.utils.common import *
from src.mushroom.constants import SCHEMA_FILE_PATH , CONFIG_FILE_PATH , PARAMS_FILE_PATH
from src.mushroom.entity.config_entity import DataIngestionConfig, DataValidationConfig , DataTransformationConfig
from src.mushroom.utils.common import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        # params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml_file(config_filepath)
        # self.params = read_yaml(params_filepath)
        self.schema = read_yaml_file(schema_filepath)

        create_directories([self.config["artifacts_root"]])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"]
        )

        return data_ingestion_config
    
    def get_data_validaion_config(self)->DataValidationConfig:
        config=self.config["data_validation"]
        schema=self.schema['COLUMNS']

        create_directories([config['root_dir']])
        data_validation_config=DataValidationConfig(
            root_dir=config['root_dir'],
            data_file=config['data_file'],
            status_file=config['status_file'],
            all_schema=schema

        )
        return data_validation_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config=self.config["data_transformation"]

        create_directories([config['root_dir']])
        data_transformation_config=DataTransformationConfig(
            root_dir=config['root_dir'],
            data_file=config['data_file'],
            final_model=config['final_model'],
            preprocessor=config['preprocessor']
        )
        return data_transformation_config

    @property
    def data_file(self):
        """Returns the data file path."""
        config = self._read_config_file()
        return config.get('data_file')  # Ensure data_file is defined in the YAML config
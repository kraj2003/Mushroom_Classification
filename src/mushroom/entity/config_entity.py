from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir:Path
    data_file:Path
    status_file:str
    all_schema:dict

@dataclass
class DataTransformationConfig:
    root_dir:Path
    data_file:Path
    final_model: Path
    preprocessor: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_arr: Path
    test_arr :Path
    model_name : str
    alpha:float
    l1_ratio:float
    target_column:str
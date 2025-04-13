from src.mushroom.exceptions.exception import ClassificationException
from src.mushroom.logging import logging
from src.mushroom.config.configuration import DataValidationConfig
import pandas as pd
import os
import sys

class Data_validation:
    def __init__(self,config=DataValidationConfig):
        self.config=config

    def validate_data(self)->bool:
        try:
            validation_status=0
            data=pd.read_csv(self.config.data_file)
            all_cols=list(data.columns)
            
            all_schema=self.config.all_schema.keys()
            datatype=self.config.all_schema.values()
            for col in all_cols:
                if col not in all_schema :
                    validation_status=False
                    with open(self.config.status_file,'w') as f:
                        f.write(f"Validation_status: {validation_status}")
                else:
                    validation_status=True
                    with open(self.config.status_file,'w') as f:
                        f.write(f"Validation Status : {validation_status}")

            return validation_status
        except Exception as e:
            raise ClassificationException(e,sys)
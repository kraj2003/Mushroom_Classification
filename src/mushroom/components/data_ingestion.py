from src.mushroom.logging import logger
from src.mushroom.entity.config_entity import DataIngestionConfig
import pandas as pd
import os
import zipfile
import gzip
import urllib.request as request
import requests
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
        
    def download_file(self):
        """
        Downloads a ZIP file from the source URL to the local data file path.
        """
        if not os.path.exists(self.config.local_data_file):
            response = requests.get(self.config.source_URL)
            if response.status_code == 200:
                with open(self.config.local_data_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"ZIP file downloaded to {self.config.local_data_file}")
            else:
                logger.error(f"Failed to download ZIP file. Status code: {response.status_code}")
        else:
            logger.info("ZIP file already exists. Skipping download.")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


# def extract_zip_file(self):
#     """
#     Extracts the zip file into the data directory
#     """
#     unzip_path = self.config.unzip_dir
#     os.makedirs(unzip_path, exist_ok=True)

#     try:
#         with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
#             zip_ref.extractall(unzip_path)
#             logger.info(f"Extracted zip file to {unzip_path}")
#     except zipfile.BadZipFile:
#         logger.error("Extraction failed: Not a valid zip file.")
#     except FileNotFoundError:
#         logger.error(f"File not found: {self.config.local_data_file}")

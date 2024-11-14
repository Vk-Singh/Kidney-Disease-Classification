from cnnClassifier.components import *
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.components.data_transformation import *
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from cnnClassifier.constants import *
from cnnClassifier.config_manager.configuration import ConfigurationManager

#print(globals())
config = ConfigurationManager(CONFIG_FILE_PATH, PARAMS_FILE_PATH)




class DataIngestionPipeline:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def run(self):
        data = DataIngestion(self.config)
        data.download_file()
        data.extract_zip_file()


class DataTransformationPipeline:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self._split_data()

    def _split_data(self):
        self._data = DataSplit(self.config)

    def get_train_dataset(self):
        return self._data.train_dataset
    
    def get_valid_dataset(self):
        return self._data.valid_dataset
    
    def get_test_dataset(self):
        return self._data.test_dataset
     
    

#ingestion_pipe = DataIngestionPipeline(config.get_data_ingestion_config())
#ingestion_pipe.run()


data_transform_pipe = DataTransformationPipeline(config.get_data_transformation_config())
data_transform_pipe.get_train_dataset()




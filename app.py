from cnnClassifier.components import *
from cnnClassifier.components.data_ingestion import *
from cnnClassifier.components.data_transformation import *
from cnnClassifier.components.model import *
from cnnClassifier.constants import *
from cnnClassifier.config_manager.configuration import ConfigurationManager

#print(globals())
config = ConfigurationManager(CONFIG_FILE_PATH, PARAMS_FILE_PATH)

print(config.get_data_ingestion_config())
print(config.get_data_transformation_config())
print(config.get_training_config())

print(config.params)

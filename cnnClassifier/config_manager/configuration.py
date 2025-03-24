from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                DataTransformationConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        """
        Initialize ConfigurationManager with configuration and parameters file paths.

        Parameters
        ----------
        config_filepath : str or Path, optional
            Path to the configuration YAML file. Defaults to CONFIG_FILE_PATH.
        params_filepath : str or Path, optional
            Path to the parameters YAML file. Defaults to PARAMS_FILE_PATH.

        Attributes
        ----------
        config : dict
            Loaded configuration data from the YAML file.
        params : dict
            Loaded parameters data from the YAML file.

        Raises
        ------
        Any exception raised by read_yaml or create_directories.
        """

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root]) 


    def get_data_ingestion_config(self) ->DataIngestionConfig:
        """
        Initialize and return DataIngestionConfig object from the configuration file.

        The method creates the root directory specified in the configuration file if it does not exist.

        Returns
        -------
        DataIngestionConfig
            DataIngestionConfig object with the configuration details.

        Raises
        ------
        Any exception raised by Path or create_directories.
        """
        config =self.config.data_ingestion

        create_directories([
            Path(config.root_dir)
        ])

        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_URL = config.source_URL,
            local_data_file = Path(config.local_data_file),
            unzip_dir = Path(config.unzip_dir)
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Initialize and return DataTransformationConfig object from the configuration file.

        The method creates the root directory specified in the configuration file if it does not exist.

        Returns
        -------
        DataTransformationConfig
            DataTransformationConfig object with the configuration details.

        Raises
        ------
        Any exception raised by Path or create_directories.
        """
        config =self.config.data_transformation
        params = self.params
        create_directories([
            Path(config.root_dir)
        ])
        data_Transformation_config = DataTransformationConfig(
            root_dir = Path(config.root_dir),
            seed = self.config.seed,
            images_folder_name = config.images_folder_name,
            file_name = config.file_name,
            test_data_size = config.test_data_size,
            train_valid_split = config.train_valid_split,
            image_size = params.image_size
        )
        return data_Transformation_config


    def get_training_config(self) -> TrainingConfig:
        """
        Initialize and return TrainingConfig object from the configuration file.

        The method creates the root directory and trained model path specified in the configuration file if they do not exist.

        Returns
        -------
        TrainingConfig
            TrainingConfig object with the configuration details.

        Raises
        ------
        Any exception raised by Path or create_directories.
        """
        training = self.config.training
        params = self.params
        #training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir),
            Path(training.trained_model_path)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            model_name = params.model_name,
            train_batch_size =  params.train_batch_size,
            valid_batch_size =  params.valid_batch_size,
            epochs=  params.epochs,
            img_size = params.image_size,
            T_max = params.T_max,
            checkpoint_path = params.checkpoint_path,
            scheduler = params.scheduler,
            learning_rate = params.learning_rate,
            min_lr = params.min_lr,
            n_accumulate = params.n_accumulate,
            weight_decay = params.weight_decay,
            classes = params.classes,
        )

        return training_config
    


    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Initialize and return EvaluationConfig object from the configuration file.

        Returns
        -------
        EvaluationConfig
            EvaluationConfig object with the configuration details.

        Raises
        ------
        Any exception raised by EvaluationConfig initialization.
        """

        eval_config = EvaluationConfig(
            path_of_model="models/training/model.h5",
            training_data="models/data_ingestion/kidney-ct-scan-image",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config


if __name__ == "__main__":
    pass

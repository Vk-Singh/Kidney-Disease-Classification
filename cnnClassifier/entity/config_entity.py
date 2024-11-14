from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    images_folder_name: str
    file_name: str
    test_data_size: float
    train_valid_split: float
    image_size: int
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_name: str
    trained_model_path: Path
    epochs: int
    img_size: list
    T_max: int
    checkpoint_path: Path
    scheduler: str
    learning_rate: float
    min_lr: float
    n_accumulate: int
    weight_decay: float
    classes: int
    train_batch_size: int
    valid_batch_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
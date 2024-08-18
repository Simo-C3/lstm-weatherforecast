from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    batch_size: int = 16
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 128


@dataclass
class TrainDataloaderConfig:
    file_path: str = ""
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 4
    validation_split: float = 0.2


@dataclass
class TestDataloaderConfig:
    file_path: str = ""
    batch_size: int = 16
    shuffle: bool = False
    num_workers: int = 4
    validation_split: float = 0.0


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class CriterionConfig:
    pass


@dataclass
class LoggerConfig:
    name: str = "default"
    level: str = "INFO"
    config_file: str = None


@dataclass
class TrainerConfig:
    epochs: int = 100
    early_stop: int = 10
    save_dir: str = ""
    device: str = "cuda"


@dataclass
class TestConfig:
    save_dir: str = ""
    device: str = "cuda"


@dataclass
class Config:
    # 共通設定
    data_dir: str = ""
    output_dir: str = ""
    seed: int = 0
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    target: str = "train"

    # 各パラメータの設定
    model: ModelConfig = ModelConfig()
    train_dataloader: TrainDataloaderConfig = TrainDataloaderConfig()
    test_dataloader: TestDataloaderConfig = TestDataloaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: CriterionConfig = CriterionConfig()
    logger: LoggerConfig = LoggerConfig()
    trainer: TrainerConfig = TrainerConfig()
    test: TestConfig = TestConfig()

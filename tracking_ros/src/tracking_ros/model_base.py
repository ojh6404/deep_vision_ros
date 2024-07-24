from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceConfigBase(ABC):
    model_name: str
    device: str = "cuda:0"

    @abstractmethod
    def get_predictor(self):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls):
        pass


class InferenceModelBase(ABC):
    def __init__(self, config: InferenceConfigBase):
        self.model_config = config
        self.predictor = config.get_predictor()

    @abstractmethod
    def predict(self, image):
        pass

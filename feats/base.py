from abc import ABC, abstractmethod

class BaseFeaturizer(ABC):
    def __init__(self, config):
        super(BaseFeaturizer, self).__init__()
        self.config = config

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

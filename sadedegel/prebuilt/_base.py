from typing import List, Union
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from ..bblock import Doc


class BaseClassifier(ABC):

    tags = ["classification"]

    def __init__(self, base_model=None, embedding_type=None):
        self.base_model = base_model
        self.embedding_type = embedding_type

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _get_embeddings(self, document):
        pass

    @abstractmethod
    def _predict(self):
        pass

    def predict(self, document: Union[Doc, str]):
        if isinstance(document, Doc):
            pass
        elif isinstance(document, str):
            document = Doc(document)

        prediction = self._predict(
            self._get_embeddings(document)
        )

        return prediction

    def __contains__(self, tag: str) -> bool:
        """Check whether instance tags contain a given tag for filtering summarizers.
        """

        return tag in self.tags

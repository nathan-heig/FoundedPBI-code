from abc import ABC
from sklearn.utils import all_estimators
import importlib

class SklearnClassifier(ABC):
    """
    Abstract base class for classifiers based on Scikit-learn
    """

    def _get_sklearn_classifier(self, model_name: str) -> type:
        estimators = all_estimators(type_filter='classifier')
        for name, class_ in estimators:
            if name == model_name:
                module_name = str(class_).split("'")[1].split(".")[1]
                class_name = class_.__name__
                return getattr(importlib.import_module(f"sklearn.{module_name}"), class_name)
            
        raise ValueError(f"Incorrect model name ({model_name}) for SklearnClassifier.")



    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, sklearn_model_name: str, sklearn_model_params: dict) -> None:
        self.sklearn_model_name = sklearn_model_name
        self.sklearn_model_params = sklearn_model_params

        self.sklearn_model = self._get_sklearn_classifier(sklearn_model_name)(**sklearn_model_params)
        
    def fit(self, *args, **kwargs):
        return self.sklearn_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.sklearn_model.predict(*args, **kwargs)
    
    def reset_model(self, *args, **kwargs):
        self.sklearn_model = self._get_sklearn_classifier(self.sklearn_model_name)(**self.sklearn_model_params)

    def name(self) -> str:
        return f"SklearnClassifier-{self.sklearn_model_name}"
    
    def __repr__(self):
        return f"SklearnClassifier(sklearn_model={self.sklearn_model_name}({self.sklearn_model_params}))"
    
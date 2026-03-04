import importlib
import inspect
import os

from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)

# Dynamically import all the strategies in this folder, so they are available when importing the package
# It is usually not a good practice to modify the globals() or __all__, but in this case I use it so that the main file does not have to be changed each time a new strategy is added
__all__ = []

dirname = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(dirname):
    if not filename.endswith(".py") or filename in {
        "__init__.py",
        "abstract_merger_strategy.py",
    }:
        continue

    module_name = filename[:-3]
    module = importlib.import_module(f".{module_name}", package=__package__)

    # Inspect each module for subclasses of BaseStrategy
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, AbstractMergerStrategy)
            and obj is not AbstractMergerStrategy
        ):
            # Inject class directly into this package’s namespace
            globals()[name] = obj
            __all__.append(name)  # type: ignore

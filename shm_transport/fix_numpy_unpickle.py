
# fix issue: Cannot unpickle file created by NumPy 2.x in NumPy 1.x
# https://github.com/numpy/numpy/issues/28340


def fix():
    import sys
    import pkgutil
    import numpy as np

    if np.__version__[:2] == "1.":

        modules = [name for _, name, ispkg in pkgutil.iter_modules(np.core.__path__)]
        for name in modules:
            try:
                sys.modules[f"numpy._core.{name}"] = getattr(np.core, name)
                # NOTE: this may cause unexpected/inconsistent behaviors !!!!!!!!
            except AttributeError:
                pass


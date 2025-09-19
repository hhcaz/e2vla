import sys
import Pyro4.util
sys.excepthook = Pyro4.util.excepthook

from . import msgpack_numpy  # activates msgpack_numpy registration
from .log import setup_level as setup_log_level
from .shm_service import (
    oneway, expose, dive_into, 
    get_shm_proxy, run_simple_server, 
    AllocateStrategy
)

from . import fix_numpy_unpickle

fix_numpy_unpickle.fix()


__all__ = [
    "oneway", "expose", "dive_into", 
    "get_shm_proxy", "run_simple_server", 
    "AllocateStrategy",
    "setup_log_level"
]

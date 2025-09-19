import traceback
from dataclasses import dataclass
from typing import Set, List, Callable
from multiprocessing.shared_memory import SharedMemory
from .log import logger


@dataclass
class Protocol(object):
    check_if: Callable
    nbytes_fn: Callable
    reduce_fn: Callable
    rebuild_fn: Callable
    copy_fn: Callable
    shm_cpu: bool
    shm_cuda: bool


class Protocols(object):
    def __init__(self):
        self.check_if_set: Set[Callable] = set()
        self.proto_list: List[Protocol] = []
    
    def add(self, p: Protocol):
        if p.check_if in self.check_if_set:
            logger.warning("protocol {} already added".format(p))
        else:
            self.check_if_set.add(p.check_if)
            self.proto_list.append(p)


protocols = Protocols()


has_numpy = True
try:
    import numpy as np
except Exception as e:
    has_numpy = False
    logger.warning("Error when importing `numpy`")
    logger.info(traceback.format_exc())

has_torch = True
try:
    import torch
    from torch.multiprocessing import reductions
except Exception as e:
    has_torch = False
    logger.warning("Error when importing `torch`")
    logger.info(traceback.format_exc())


if has_numpy:
    def check_if_numpy(x):
        return isinstance(x, np.ndarray)

    def calc_numpy_nbytes(x: np.ndarray):
        return x.nbytes

    def reduce_numpy(shm: SharedMemory, x: np.ndarray, offset: int):
        """Record the shared memory name, dtype and shape of numpy array"""
        if not x.data.c_contiguous:
            x = np.ascontiguousarray(x)
        
        shm_np_view = np.ndarray(x.shape, x.dtype, buffer=shm.buf, offset=offset)
        np.copyto(shm_np_view, x)
        metadata = (x.shape, x.dtype)
        return metadata
    
    def rebuild_numpy(shm: SharedMemory, metadata, offset: int):
        """Rebuild numpy array from shared memory, shape amd dtype"""
        shape, dtype = metadata
        data = np.ndarray(shape, dtype, buffer=shm.buf, offset=offset)
        return data
    
    def copy_numpy(x: np.ndarray):
        return x.copy()
    
    protocols.add(Protocol(check_if_numpy, calc_numpy_nbytes, 
                           reduce_numpy, rebuild_numpy, copy_numpy, 
                           True, False))


if has_torch:
    def check_if_torch_cuda(x):
        return isinstance(x, torch.Tensor) and x.is_cuda

    def calc_torch_cuda_nbytes(x: torch.Tensor):
        return 0  # shared via GPU, therefore 0 for shared memory on cpu
    
    def reduce_torch_cuda(shm, x: torch.Tensor, offset):
        _, metadata = reductions.reduce_tensor(x.detach().contiguous())
        return metadata
    
    def rebuild_torch_cuda(shm, metadata, offset):
        tensor = reductions.rebuild_cuda_tensor(*metadata)
        return tensor
    
    def copy_torch_cuda(x: torch.Tensor):
        return x.detach().clone()
    
    protocols.add(Protocol(check_if_torch_cuda, calc_torch_cuda_nbytes, 
                           reduce_torch_cuda, rebuild_torch_cuda, copy_torch_cuda, 
                           False, True))
    

    def check_if_torch_cpu(x):
        return isinstance(x, torch.Tensor) and x.is_cpu
    
    def calc_torch_cpu_nbytes(x: torch.Tensor):
        return x.untyped_storage().nbytes()  # newer version of pytorch has x.nbytes
    
    def reduce_torch_cpu(shm: SharedMemory, x: torch.Tensor, offset: int):
        """Record the shared memory name, dtype and shape of numpy array"""
        return reduce_numpy(shm, np.asarray(x.detach().contiguous()), offset)
    
    def rebuild_torch_cpu(shm: SharedMemory, metadata, offset: int):
        """Rebuild torch tensor from memory address, size, dtype and shape"""
        return torch.from_numpy(rebuild_numpy(shm, metadata, offset))
    
    def copy_torch_cpu(x: torch.Tensor):
        return x.detach().clone()
    
    protocols.add(Protocol(check_if_torch_cpu, calc_torch_cpu_nbytes, 
                           reduce_torch_cpu, rebuild_torch_cpu, copy_torch_cpu, 
                           True, False))



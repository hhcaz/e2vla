import copy
from typing import Callable
from functools import partial
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from .proto import protocols


def flatten(x):
    """
    Arguments:
    - x: structures (dict, list, or tuple) containing non-structure
        elements, can be nested
    
    Returns:
    - structure: Same structure as x, with all the non-structure 
        elements set to be None
    - elements: elements searched from x, DFS order

    NOTE: use with `recover`
    """
    elements = []
    def _recursive_flat(x):
        if isinstance(x, dict):
            structure = {k:_recursive_flat(v) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            structure = type(x)(_recursive_flat(item) for item in x)
        elif hasattr(x, "_dive_into_") or hasattr(x.__class__, "_dive_into_"):
            structure = copy.copy(x)
            for k, v in x.__dict__.items():
                setattr(structure, k, _recursive_flat(v))
        else:
            structure = None
            elements.append(x)
        return structure
    
    structure = _recursive_flat(x)
    return structure, elements


def recover(structure, elements: list):
    """
    Arguments:
    - structure: nested structures of (dict, list, or tuple), its own
        containing elements will be ignored
    - elements: list or tuple of elements need to be put on structure

    Returns:
    - x: same structure with elements put with DFS order

    NOTE: use with `flatten`
    """
    index = 0
    def _recursive_recover(x):
        nonlocal index
        if isinstance(x, dict):
            structure = {k:_recursive_recover(v) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            structure = type(x)(_recursive_recover(item) for item in x)
        elif hasattr(x, "_dive_into_") or hasattr(x.__class__, "_dive_into_"):
            structure = copy.copy(x)
            for k, v in x.__dict__.items():
                setattr(structure, k, _recursive_recover(v))
        else:
            structure = elements[index]
            index += 1
        return structure
    
    return _recursive_recover(structure)


@dataclass
class _ReducedData(object):
    rebuild_fn: Callable
    metadata: tuple
    copy_fn: Callable
    shm_cpu: bool
    shm_cuda: bool


def calc_shm_nbytes(elements: list):
    """
    Arguments:
    - elements: list of array/tensor

    Returns:
    - nbytes: number of bytes for shared_memory if used
    """
    nbytes = 0
    for ele in elements:
        for proto in protocols.proto_list:
            if proto.check_if(ele):
                nbytes += proto.nbytes_fn(ele)
                break
    return nbytes


def reduce_elements(elements: list, shm=None, en_sh_cuda=True):
    """Reduce elements to metadatas and store data to shared memory.

    Arguments:
    - elements: list of array/tensor
    - shm: None or SharedMemory instance, if shm is None, then don't reduce cpu array
    - en_sh_cuda: bool, if True, transport shared cuda tensor
    """
    offset = 0
    for i, ele in enumerate(elements):
        for proto in protocols.proto_list:
            if not proto.check_if(ele):
                continue

            if (proto.shm_cpu and shm is not None) or (proto.shm_cuda and en_sh_cuda):
                metadata = proto.reduce_fn(shm, ele, offset)
                elements[i] = _ReducedData(partial(proto.rebuild_fn, offset=offset),
                                           metadata, proto.copy_fn, proto.shm_cpu, proto.shm_cuda)
                offset = offset + proto.nbytes_fn(ele)
                break


def rebuild_elements(elements: list, shm: SharedMemory, copy_cpu: bool, copy_cuda: bool):
    """Rebuild elements from shared memory"""
    for i, ele in enumerate(elements):
        if isinstance(ele, _ReducedData):
            elements[i] = ele.rebuild_fn(shm, ele.metadata)
            if (ele.shm_cpu and copy_cpu) or (ele.shm_cuda and copy_cuda):
                elements[i] = ele.copy_fn(elements[i])

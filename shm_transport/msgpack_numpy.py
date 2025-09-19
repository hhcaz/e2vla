import numpy as np
# from functools import partial
# from dataclasses import asdict
from Pyro4.util import MsgpackSerializer, _serializers, _serializers_by_id

# from .shm_service import _ReducedArguments, _ReducedReturns


# def as_dict(cls, obj):
#     items = asdict(obj)
#     items["__class__"] = cls.__module__ + "." + cls.__name__
#     return items


# def from_dict(cls, items: dict):
#     items.pop("__class__")
#     return cls(**items)


class MsgpackNumpySerializer(MsgpackSerializer):

    # def __init__(self):
    #     super().__init__()
    #     self.register_class_to_dict(_ReducedArguments, partial(as_dict, _ReducedArguments))
    #     self.register_class_to_dict(_ReducedReturns, partial(as_dict, _ReducedReturns))
    #     self.register_dict_to_class(
    #         _ReducedArguments.__module__ + "." + _ReducedArguments.__name__,
    #         partial(from_dict, _ReducedArguments)
    #     )
    #     self.register_dict_to_class(
    #         _ReducedReturns.__module__ + "." + _ReducedReturns.__name__,
    #         partial(from_dict, _ReducedReturns)
    #     )

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }
        return super().default(obj)
    
    def object_hook(self, obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        
        return super().object_hook(obj)


try:
    import msgpack
    from .log import logger
    
    _ser = MsgpackNumpySerializer()
    # this overrites the default msgpack serializer of Pyro4
    _serializers["msgpack"] = _ser
    _serializers_by_id[_ser.serializer_id] = _ser
    logger.debug("Replace msgpack with custom msgpack_numpy")
    del _ser
except ImportError:
    pass


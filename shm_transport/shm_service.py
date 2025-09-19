import os
import uuid
import time
import Pyro4
import inspect
import traceback
import Pyro4.naming
import multiprocessing
from dataclasses import dataclass
from typing import Callable, Union

from .log import logger
from . import nested_data
from .shm_mng import (
    get_server_shm_manager, 
    get_client_shm_managers, 
    _ShmManager,
    _EXIT_CALLED, 
    AllocateStrategy, 
    register_cleanup_this_process
)


def my_base64(x: int):
    table = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
    if x == 0:
        return "0"
    
    b64 = []
    while x > 0:
        x, idx = divmod(x, 64)
        b64.append(table[idx])
    return "".join(reversed(b64))


def my_base64_len11(x: int):
    return "{:0>11s}".format(my_base64(x))  # 64bit cpu takes 11-width addr for base64


@dataclass
class _ReducedArguments(object):
    structure: Union[tuple, list, dict]
    elements: list
    proxy_id_func_name: str
    arg_shm_enabled: bool


@dataclass
class _ReducedReturns(object):
    structure: Union[tuple, list, dict]
    elements: list
    ret_shm_name: str
    compute_time: float


def reduce_inputs_rebuild_returns(
    f: Callable, 
    proxy_shm_mng: _ShmManager, 
    proxy_id_func_name: str, 
    en_arg_shm: bool,
    en_arg_sh_cuda: bool,
    shm_nbytes_thresh: int,
    allocate_strategy: AllocateStrategy,
    copy_ret_shm: bool,
    copy_ret_sh_cuda: bool
):
    """
    Reduce the inputs, send to remote server, and rebuild its returns

    :Arguments:
    - f: wrapper of remote object's function, typically, a Pyro4's RemoteMethod instance
    - proxy_shm_mng: a shared memory manager owned by a proxy
    - proxy_id_func_name: format {id(proxy)}.{proxy's function name}
    - en_arg_shm: enable transport arguments via shared memory to remote server
    - en_arg_sh_cuda: enable transport arguments via GPU to remote server
    - shm_nbytes_thresh: threshold of data bytes to trigger shared memory data transporting
    - allocate_strategy: 
    - copy_ret_shm: whether to copy the data from shared memory owned by remote server
    - copy_ret_sh_cuda: whether to copy shared CUDA tensor
        - NOTE: If you want to preserve the data from shared memory owned by the remote server, 
        you'd better copy it since the remote server may be idle and the resource related to 
        the shared memory will be invalid.
    """

    def client_f_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        structure, elements = nested_data.flatten((args, kwargs))

        arg_shm = None
        if en_arg_shm:
            nbytes = nested_data.calc_shm_nbytes(elements)
            if nbytes > shm_nbytes_thresh:
                arg_shm = proxy_shm_mng.try_new_shm(nbytes, proxy_id_func_name, 
                                                    allocate_strategy)

        nested_data.reduce_elements(elements, arg_shm, en_arg_sh_cuda)
        arg_shm_enabled = arg_shm is not None
        request = _ReducedArguments(
            structure=structure,
            elements=elements,
            proxy_id_func_name=arg_shm.name if arg_shm_enabled else proxy_id_func_name,
            arg_shm_enabled=arg_shm_enabled
        )
        response = f(request)

        if isinstance(response, _ReducedReturns):
            if response.ret_shm_name is not None:
                ret_shm = proxy_shm_mng.try_open_shm(response.ret_shm_name)
            else:
                ret_shm = None
            nested_data.rebuild_elements(response.elements, ret_shm, copy_ret_shm, copy_ret_sh_cuda)
            ret = nested_data.recover(response.structure, response.elements)
            t1 = time.perf_counter()
            total_time = t1 - t0
            compute_time = response.compute_time
            logger.debug("Shm transport time: total = {:.3f}s, commu = {:.3f}s, compute = {:.3f}s"
                        .format(total_time, total_time - compute_time, compute_time))
        else:
            ret = response
        return ret
    return client_f_wrapper


def rebuild_inputs_reduce_returns(
    f: Callable, 
    copy_arg_shm: bool,
    copy_arg_sh_cuda: bool, 
    en_ret_shm: bool,
    en_ret_sh_cuda: bool,
    shm_nbytes_thresh: int,
    allocate_strategy: AllocateStrategy,
):
    """
    Rebuild the inputs send from proxy, do culculation, and reduce the returns

    :Arguments:
    - f: wrapper of a method
    - copy_arg_shm: whether to copy the data from shared memory owned by the requesting proxy
    - copy_arg_sh_cuda: whether to copy shared CUDA tensor
        - NOTE: If you want to preserve the data from shared memory owned by the proxy, 
        you'd better copy it since the proxy may be idle and the resource related to 
        the shared memory will be invalid.
    - en_ret_shm: bool, enable transport returns via shared memory to proxy
    - en_ret_sh_cuda: bool, enable transport returns via GPU to proxy
    - shm_nbytes_thresh: int, threshold of data bytes to trigger shared memory data transporting
    - allocate_strategy: 
    """
    def server_f_remote_call(self_or_cls, request: _ReducedArguments):
        if request.arg_shm_enabled:
            arg_shm = get_server_shm_manager().try_open_shm(request.proxy_id_func_name)
        else:
            arg_shm = None
        nested_data.rebuild_elements(request.elements, arg_shm, copy_arg_shm, copy_arg_sh_cuda)
        args, kwargs = nested_data.recover(request.structure, request.elements)

        try:
            t0 = time.perf_counter()
            if self_or_cls:
                ret = f(self_or_cls, *args, **kwargs)
            else:
                ret = f(*args, **kwargs)
            t1 = time.perf_counter()
            compute_time = t1 - t0
        except Exception as e:
            traceback.print_exc()
            raise e

        ret_structure, ret_elements = nested_data.flatten(ret)
        ret_shm = None
        ret_shm_name = ".".join(request.proxy_id_func_name.split(".")[:2] + ["ret"])
        if en_ret_shm:
            nbytes = nested_data.calc_shm_nbytes(ret_elements)
            if nbytes > shm_nbytes_thresh:
                ret_shm = get_server_shm_manager().try_new_shm(
                    nbytes, ret_shm_name, allocate_strategy)

        nested_data.reduce_elements(ret_elements, ret_shm, en_ret_sh_cuda)
        response = _ReducedReturns(
            structure=ret_structure,
            elements=ret_elements,
            ret_shm_name=ret_shm.name if ret_shm is not None else None,
            compute_time=compute_time
        )
        return response

    def server_f_wrapper(*args, **kwargs):
        """
        args can be two cases:
        - (reduced arguments,)
        - (inst or cls handle, reduced arguments)
        - (arguments from local call)
        """
        if len(args) == 1 and isinstance(args[0], _ReducedArguments):
            return server_f_remote_call(None, args[0])
        elif len(args) == 2 and isinstance(args[1], _ReducedArguments):
            return server_f_remote_call(*args)
        else:
            return f(*args, **kwargs)
    
    server_f_wrapper._pyroExposed = True
    server_f_wrapper.__name__ = f.__name__
    return server_f_wrapper


def rebuild_inputs_no_return(
    f: Callable,
    copy_arg_shm: bool,
    copy_arg_sh_cuda: bool, 
):
    def server_f_remote_call(self_or_cls, request: _ReducedArguments):
        if request.arg_shm_enabled:
            arg_shm = get_server_shm_manager().try_open_shm(request.proxy_id_func_name)
        else:
            arg_shm = None
        nested_data.rebuild_elements(request.elements, arg_shm, copy_arg_shm, copy_arg_sh_cuda)
        args, kwargs = nested_data.recover(request.structure, request.elements)

        try:
            if self_or_cls:
                ret = f(self_or_cls, *args, **kwargs)
            else:
                ret = f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e
        return ret  # ret is simply None
    
    def server_f_wrapper(*args, **kwargs):
        """
        args can be two cases:
        - (reduced arguments,)
        - (inst or cls handle, reduced arguments)
        - (arguments from local call)
        """
        if len(args) == 1 and isinstance(args[0], _ReducedArguments):
            return server_f_remote_call(None, args[0])
        elif len(args) == 2 and isinstance(args[1], _ReducedArguments):
            return server_f_remote_call(*args)
        else:
            return f(*args, **kwargs)
    
    server_f_wrapper._pyroOneway = True
    server_f_wrapper.__name__ = f.__name__
    return server_f_wrapper


def oneway(
    copy_arg_shm: bool,
    copy_arg_sh_cuda: bool
):
    """Rebuild inputs only, no returns. Non-blocking.

    :Arguments:
    - copy_arg_shm: whether to copy the data from shared memory owned by the requesting proxy
    - copy_arg_sh_cuda: whether to copy shared CUDA tensor
        - NOTE: If you want to preserve the data from shared memory owned by the proxy, 
        you'd better copy it since the proxy may be idle and the resource related to 
        the shared memory will be invalid.
    """
    def rebuild_inputs_no_return_wrapper(f: Callable):
        attrname = getattr(f, "__name__", None)
        if not attrname or isinstance(f, (classmethod, staticmethod)):
            # we could be dealing with a descriptor (classmethod/staticmethod), 
            # this means the order of the decorators is wrong
            if inspect.ismethoddescriptor(f):
                attrname = f.__get__(None, dict).__name__
                raise AttributeError("using @oneway on a classmethod/staticmethod must be done "
                                     "below @classmethod/@staticmethod. Method: " + attrname)
            else:
                raise AttributeError("@oneway cannot determine what this is: " + repr(f))
        return rebuild_inputs_no_return(
            f,
            copy_arg_shm=copy_arg_shm,
            copy_arg_sh_cuda=copy_arg_sh_cuda
        )
    return rebuild_inputs_no_return_wrapper


def expose(
    copy_arg_shm=True,
    copy_arg_sh_cuda=True, 
    en_ret_shm=True, 
    en_ret_sh_cuda=True,
    shm_nbytes_thresh=0, 
    allocate_strategy=AllocateStrategy.default()
):
    """ Make the decorated method exposed to proxy

    :Arguments:
    - f: wrapper of a method
    - copy_arg_shm: whether to copy the data from shared memory owned by the requesting proxy
    - copy_arg_sh_cuda: whether to copy shared CUDA tensor
        - NOTE: If you want to preserve the data from shared memory owned by the proxy, 
        you'd better copy it since the proxy may be idle and the resource related to 
        the shared memory will be invalid.
    - en_ret_shm: bool, enable transport returns via shared memory to proxy
    - en_ret_sh_cuda: bool, enable transport returns via GPU to proxy
    - shm_nbytes_thresh: int, threshold of data bytes to trigger shared memory data transporting
    - allocate_strategy: 
    """
    def rebuild_inputs_reduce_returns_wrapper(f: Callable):
        attrname = getattr(f, "__name__", None)
        if not attrname or isinstance(f, (classmethod, staticmethod)):
            # we could be dealing with a descriptor (classmethod/staticmethod), 
            # this means the order of the decorators is wrong
            if inspect.ismethoddescriptor(f):
                attrname = f.__get__(None, dict).__name__
                raise AttributeError("using @expose on a classmethod/staticmethod must be done "
                                     "below @classmethod/@staticmethod. Method: " + attrname)
            else:
                raise AttributeError("@expose cannot determine what this is: " + repr(f))
        
        return rebuild_inputs_reduce_returns(
            f, 
            copy_arg_shm=copy_arg_shm, 
            copy_arg_sh_cuda=copy_arg_sh_cuda, 
            en_ret_shm=en_ret_shm,
            en_ret_sh_cuda=en_ret_sh_cuda,
            shm_nbytes_thresh=shm_nbytes_thresh,
            allocate_strategy=allocate_strategy
        )
    return rebuild_inputs_reduce_returns_wrapper


def dive_into(inst_or_cls):
    """Add a flag to a custom instance or class, indicating that it's attributes 
    should be reduced/rebuild and data will be transported via shared memory.
    """
    inst_or_cls._dive_into_ = True
    return inst_or_cls


class ShmProxy(object):
    __myAttributes = frozenset(["_pyro_proxy", "_prev_id", 
                                "_shm_mng", "_shm_name_prefix",  
                                "_en_arg_shm", "_en_arg_sh_cuda",
                                "_shm_nbytes_thresh", "_allocate_strategy", 
                                "_copy_ret_shm", "_copy_ret_sh_cuda",
                                "_same_machine_as_remote", "_force_use_socket"
                                ])

    def __init__(
        self, 
        uri, 
        connected_socket=None, 
        en_arg_shm=True,
        en_arg_sh_cuda=True,
        shm_nbytes_thresh=0, 
        allocate_strategy=AllocateStrategy.default(),
        copy_ret_shm=True,
        copy_ret_sh_cuda=True, 
        force_use_socket=False,
        serializer="pickle"
    ):
        self._shm_mng = None
        self._shm_name_prefix = None
        self._en_arg_shm = en_arg_shm
        self._en_arg_sh_cuda = en_arg_sh_cuda
        self._shm_nbytes_thresh = shm_nbytes_thresh
        self._allocate_strategy = allocate_strategy
        self._copy_ret_shm = copy_ret_shm
        self._copy_ret_sh_cuda = copy_ret_sh_cuda
        self._same_machine_as_remote = None
        self._force_use_socket = force_use_socket

        # if multiprocessing.parent_process() is not None:
        #     start_method = multiprocessing.get_context().get_start_method()
        #     assert "spawn" == start_method, \
        #            "ShmProxy only supports mp with spawn, got {}".format(start_method)

        register_cleanup_this_process()
        self._prev_id = id(self)
        self._pyro_proxy = Pyro4.Proxy(uri, connected_socket)
        self._pyro_proxy._pyroSerializer = serializer
        self._pyro_proxy._pyroGetMetadata()
        
        remote_uuid = self._pyro_proxy.get_machine_id_via_uuid()
        if remote_uuid == str(uuid.getnode()):
            self._same_machine_as_remote = True
            if not self._force_use_socket:
                logger.info("Proxy at the same machine as remote, shm transport is enabled.")
        else:
            self._same_machine_as_remote = False
            logger.warning("Proxy not at the same machine as remote, shm transport is disabled.")
    
    def __lazy_init(self):
        self._shm_name_prefix = my_base64_len11(id(self))
        self._shm_mng = get_client_shm_managers().try_new_mng(
            self._pyro_proxy, self._shm_name_prefix
        )
    
    def __getattr__(self, name):
        if name in ShmProxy.__myAttributes:
            # allows it to be safely pickled
            raise AttributeError(name)

        if (self._shm_name_prefix is None) or (self._prev_id != id(self)):
            self.__lazy_init()
        self._prev_id = id(self)

        arg_shm_name = ".".join([self._shm_name_prefix, name, "arg"])
        f = getattr(self._pyro_proxy, name)
        
        if self._same_machine_as_remote and (not self._force_use_socket):
            assert not _EXIT_CALLED, "Exit called, abort proxy call."
            return reduce_inputs_rebuild_returns(
                f=f,
                proxy_shm_mng=self._shm_mng,
                proxy_id_func_name=arg_shm_name,
                en_arg_shm=self._en_arg_shm,
                en_arg_sh_cuda=self._en_arg_sh_cuda,
                shm_nbytes_thresh=self._shm_nbytes_thresh,
                allocate_strategy=self._allocate_strategy,
                copy_ret_shm=self._copy_ret_shm,
                copy_ret_sh_cuda=self._copy_ret_sh_cuda
            )
        else:
            return f

    def __del__(self):
        if self._shm_name_prefix is not None:
            get_client_shm_managers().del_proxy_mng(self._shm_name_prefix)


def get_shm_proxy(
    uri_name: str,
    ns_host=None,
    ns_port=None,
    en_arg_shm=True,
    en_arg_sh_cuda=True,
    shm_nbytes_thresh=8192,
    allocate_strategy=AllocateStrategy.default(),
    copy_ret_shm=True,
    copy_ret_sh_cuda=True, 
    force_use_socket=False,
    serializer="pickle"
):
    """Get the proxy which transport data between remote object via shared memory
    
    :Arguments:
    - uri_name: str, name of the desired service
    - ns_host: host of naming server
    - ns_port: port of naming server
    - en_arg_shm: whether enable shared memory to transport the arguments
    - en_arg_sh_cuda: whether enbale shared cuda tensor to transport the arguments
    - shm_nbytes_thresh: threshold of bytes to transport the arguments when enable shared memory
    - allocate_strategy
    - copy_ret_shm: whether to copy the data from shared memory
    - copy_ret_sh_cuda: whether to copy the data from shared cuda tensor
    - force_use_socket: disable all the shm transport for args and returns
    - serializer: default to pickle. if you faces np._core module not found error, you can change it to msgpack

    :Returns:
    - proxy: a proxy of remote object
    """
    ns: Pyro4.naming.NameServer = Pyro4.locateNS(ns_host, ns_port)
    uri = ns.lookup(uri_name)
    logger.info("Find uri = {}".format(uri))
    proxy = ShmProxy(
        uri=uri, 
        connected_socket=None, 
        en_arg_shm=en_arg_shm, 
        en_arg_sh_cuda=en_arg_sh_cuda, 
        shm_nbytes_thresh=shm_nbytes_thresh, 
        allocate_strategy=allocate_strategy,
        copy_ret_shm=copy_ret_shm,
        copy_ret_sh_cuda=copy_ret_sh_cuda,
        force_use_socket=force_use_socket,
        serializer=serializer
    )
    return proxy


def run_simple_server(
    obj,
    uri_name: str, 
    pickle_version=5,
    multiplex=True,
    ns_host=None,
    ns_port=None,
    daemon_host=None,
    daemon_port=0,
):
    """
    :Arguments:
    - obj: object to be served
    - uri_name: name of service
    - pickle_version: 5 for python>=3.8, or 4 for python<3.8
    - multiplex: set True to use multiplex, otherwise use thread
    - ns_host: host of naming server
    - ns_port: port of naming server
    - daemon_host: host of service
    - daemon_port: port of service
    """
    if multiprocessing.parent_process() is not None:
        start_method = multiprocessing.get_context().get_start_method()
        assert "spawn" == start_method, \
               "run_simple_server only supports mp with spawn, got {}".format(start_method)

    Pyro4.config.reset()
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle", "json", "marshal", "serpent", "msgpack"])
    Pyro4.config.SERVERTYPE = "multiplex" if multiplex else "thread"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = pickle_version

    def exposed_oneway(x):
        x = Pyro4.expose(x)
        x = Pyro4.oneway(x)
        return x

    @staticmethod
    @exposed_oneway
    def request_closing_proxy(name: str):
        if isinstance(name, str):
            logger.debug("Receives request to close all shm owned by proxy: {}".format(name))
            get_server_shm_manager().del_proxy_related_shm(name)
    
    @staticmethod
    @expose(en_ret_shm=False, en_ret_sh_cuda=False)
    def get_machine_id_via_uuid(): return str(uuid.getnode())

    # call this method when the proxy is to be deleted to 
    # clear the proxy related shared memory
    obj.__class__.request_closing_proxy = request_closing_proxy
    obj.__class__.get_machine_id_via_uuid = get_machine_id_via_uuid
    register_cleanup_this_process()

    with Pyro4.Daemon(
        host=daemon_host,
        port=daemon_port
    ) as daemon:
        ns: Pyro4.naming.NameServer = Pyro4.locateNS(
            host=ns_host,
            port=ns_port
        )
        uri = daemon.register(obj, force=True)
        ns.register(uri_name, uri)
        logger.info("Start service, uri = {}".format(uri))
        daemon.requestLoop()


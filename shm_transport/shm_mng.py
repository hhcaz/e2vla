import os
import Pyro4
import traceback
from typing import Dict, Set, Tuple
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from .log import logger
from . import safe_exit


# Ref:
# https://stackoverflow.com/questions/77285558/
# why-does-python-shared-memory-implicitly-unlinked-on-exit
def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        shm_cleanup_fn = resource_tracker._CLEANUP_FUNCS.pop("shared_memory")
    else:
        shm_cleanup_fn = None
    return shm_cleanup_fn


def add_shm_to_resource_tracker(shm_cleanup_fn):
    resource_tracker.register = resource_tracker._resource_tracker.register
    resource_tracker.unregister = resource_tracker._resource_tracker.unregister
    if shm_cleanup_fn is not None:
        resource_tracker._CLEANUP_FUNCS["shared_memory"] = shm_cleanup_fn


def no_track_open_shm(name: str):
    # disable auto unlink operation when the read process exit
    shm_cleanup_fn = remove_shm_from_resource_tracker()
    shm = SharedMemory(name, create=False)
    add_shm_to_resource_tracker(shm_cleanup_fn)
    return shm


# naming of a shared memory:
# {instance_id}.{function_name}.{arg or ret}.{version}


# def parse_shm_name(name: str):
#     prefix, func_name, is_arg, version = name.split(".")
#     is_arg = (is_arg == "arg")
#     version = int(version)
#     return prefix, func_name, is_arg, version


def split_version(name_wi_version: str):
    dot_idx = name_wi_version.rfind(".")
    name_wo_version = name_wi_version[:dot_idx]
    version = int(name_wi_version[dot_idx+1:])
    return name_wo_version, version


class AllocateStrategy(object):
    class Init(object):
        class Compact(object): pass
        class Block(object):
            def __init__(self, block_size: int = 512):
                self.block_size = block_size
    
    class Inc(object):
        class Compact(object): pass
        class Double(object): pass
        class Block(object):
            def __init__(self, block_size: int = 512):
                self.block_size = block_size
    
    def __init__(self, init, inc):
        self.init = init
        self.inc = inc
    
    @classmethod
    def default(cls):
        return cls(cls.Init.Compact(), cls.Inc.Compact())
    
    def override_shm(self, old_nbytes: int, new_nbytes: int, new_full_name: str):
        actual_nbytes = old_nbytes
        if isinstance(self.inc, self.Inc.Double):
            while actual_nbytes < new_nbytes:
                actual_nbytes *= 2
        elif isinstance(self.inc, self.Inc.Block):
            n_block, n_res = divmod(new_nbytes, self.inc.block_size)
            actual_nbytes = (n_block + int(n_res > 0)) * self.inc.block_size
        else:
            actual_nbytes = new_nbytes
        shm = SharedMemory(name=new_full_name, create=True, size=actual_nbytes)
        return shm
    
    def new_shm(self, nbytes: int, shm_full_name: str):
        """
        Arguments:
        - nbytes: int, desired nbytes (compact) of shared memory
        - shm_full_name: str or an instance of SharedMemory
        """
        if isinstance(self.init, self.Init.Block):
            n_block, n_res = divmod(nbytes, self.init.block_size)
            new_nbytes = (n_block + int(n_res > 0)) * self.init.block_size
        else:
            new_nbytes = nbytes
        shm = SharedMemory(name=shm_full_name, create=True, size=new_nbytes)
        return shm


class _ShmManager(object):
    def __init__(self):
        self.sname2version_own: Dict[str, int] = {}  # short name
        self.fname2shm_own: Dict[str, SharedMemory] = {}  # full name

        self.sname2version_other: Dict[str, int] = {}
        self.fname2shm_other: Dict[str, SharedMemory] = {}

    def try_new_shm(
        self, 
        nbytes: int, 
        name_wo_version: str, 
        strategy: AllocateStrategy = AllocateStrategy.default()
    ):
        """
        Open a shared memory for read and write, 
        if the memory already exists, then reuse it
        """
        if name_wo_version in self.sname2version_own:
            version = self.sname2version_own[name_wo_version]
            name_wi_version = ".".join([name_wo_version, str(version)])
            shm = self.fname2shm_own[name_wi_version]
            logger.debug("Find exsiting shm: {}, capacity size = {}, data size = {}"
                         .format(name_wi_version, shm.size, nbytes))
            
            if shm.size < nbytes:
                # delete the old version
                old_shm = self.fname2shm_own.pop(name_wi_version)
                old_shm.close()
                old_shm.unlink()
                # update version
                self.sname2version_own[name_wo_version] = version + 1
                name_wi_version = ".".join([name_wo_version, str(version + 1)])
                shm = strategy.override_shm(shm.size, nbytes, name_wi_version)
                self.fname2shm_own[shm.name] = shm

                logger.debug("Resize owned shm: {} -> {}".format(old_shm.name, shm.name))
                logger.debug("Origianl size = {}, required size = {}, actual new size = {}"
                             .format(old_shm.size, nbytes, shm.size))
                del old_shm
        else:
            name_wi_version = ".".join([name_wo_version, "0"])
            shm = strategy.new_shm(nbytes, name_wi_version)
            self.sname2version_own[name_wo_version] = 0
            self.fname2shm_own[shm.name] = shm
            logger.debug("New shm: {}, size = {}".format(shm.name, shm.size))
        return shm

    def try_open_shm(self, name_wi_version: str):
        """
        Open a shared memory for read, 
        if the memory already exists, then reuse it
        """
        name_wo_version, version = split_version(name_wi_version)
        logger.debug("Open other's shm: {}".format(name_wi_version))
        if name_wo_version in self.sname2version_other:
            old_version = self.sname2version_other[name_wo_version]
            if version != old_version:
                old_shm_fname = ".".join([name_wo_version, str(old_version)])
                old_shm = self.fname2shm_other.pop(old_shm_fname)
                old_shm.close()
                logger.debug("Find older version shm: {}, delete it".format(old_shm.name))
                del old_shm
                shm = no_track_open_shm(name_wi_version)
                self.fname2shm_other[shm.name] = shm
            else:
                shm = self.fname2shm_other[name_wi_version]
        else:
            shm = no_track_open_shm(name_wi_version)
            self.fname2shm_other[shm.name] = shm
        self.sname2version_other[name_wo_version] = version
        return shm
    
    def del_shm(self, name_wo_version: str):
        if name_wo_version in self.sname2version_other:
            version = self.sname2version_other.pop(name_wo_version)
            name_wi_version = ".".join([name_wo_version, str(version)])
            shm = self.fname2shm_other.pop(name_wi_version)
            shm.close()  # only close, don't unlink
            del shm
            logger.debug("Close other's shm: {}".format(name_wi_version))
        
        if name_wo_version in self.sname2version_own:
            version = self.sname2version_own.pop(name_wo_version)
            name_wi_version = ".".join([name_wo_version, str(version)])
            shm = self.fname2shm_own.pop(name_wi_version)
            shm.close(); shm.unlink()
            del shm
            logger.debug("Unlink owned shm: {}".format(name_wi_version))
    
    def clear_all(self):
        snames = list(self.sname2version_other.keys())
        for sname in snames:
            self.del_shm(sname)
        
        snames = list(self.sname2version_own.keys())
        for sname in snames:
            self.del_shm(sname)
    
    def __del__(self):
        self.clear_all()


class _ServerShmManager(_ShmManager):
    """Shared memory manager of the server side"""
    def __init__(self):
        super().__init__()
        self.prefix2names: Dict[str, Set[str]] = {}
    
    def _add_proxy_related_shm(self, proxy_name: str, shm_name: str):
        if proxy_name in self.prefix2names:
            self.prefix2names[proxy_name].add(shm_name)
        else:
            self.prefix2names[proxy_name] = set([shm_name])
    
    def _del_proxy_related_shm(self, proxy_name: str, shm_name: str):
        if proxy_name in self.prefix2names:
            self.prefix2names[proxy_name].discard(shm_name)
            if len(self.prefix2names[proxy_name]) == 0:
                self.prefix2names.pop(proxy_name)

    def try_new_shm(
        self, 
        nbytes: int, 
        name_wo_version: str, 
        strategy: AllocateStrategy = AllocateStrategy.default()
    ):
        """
        Open a shared memory for read and write, 
        if the memory already exists, then reuse it
        """
        shm = super().try_new_shm(nbytes, name_wo_version, strategy)
        # register lookup table for each requesting proxy
        proxy = shm.name.split(".")[0]
        self._add_proxy_related_shm(proxy, name_wo_version)
        return shm
    
    def try_open_shm(self, name_wi_version: str):
        shm = super().try_open_shm(name_wi_version)
        # register lookup table for each requesting proxy
        proxy = shm.name.split(".")[0]
        name_wo_version, version = split_version(name_wi_version)
        self._add_proxy_related_shm(proxy, name_wo_version)
        return shm

    def del_proxy_related_shm(self, proxy: str):
        if proxy in self.prefix2names:
            name_wo_versions = self.prefix2names.pop(proxy)
            for sname in name_wo_versions:
                self.del_shm(sname)


class _ClientShmManagers(object):
    """Shared memory managers of the client side"""
    def __init__(self):
        self.managers: Dict[str, Tuple[Pyro4.Proxy, _ShmManager]] = {}
    
    def try_new_mng(self, proxy: Pyro4.Proxy, name: str):
        if name not in self.managers:
            self.managers[name] = (proxy, _ShmManager())
        return self.managers[name][1]

    def _req_close_proxy(self, proxy: Pyro4.Proxy, name: str):
        try:
            logger.debug("Request closing proxy {} related shm on remote".format(name))
            proxy.request_closing_proxy(name)
        except Exception as e:
            logger.error("Error when trying to close proxy {} related shm"
                         .format(name))
            logger.error(traceback.format_exc())
    
    def del_proxy_mng(self, name: str):
        if name in self.managers:
            proxy, mng = self.managers.pop(name)
            self._req_close_proxy(proxy, name)
            mng.clear_all()
    
    def clear_all(self):
        while self.managers:
            name, (proxy, mng) = self.managers.popitem()
            self._req_close_proxy(proxy, name)
            mng.clear_all()



_pid_to_server_shm_manager = {}
_pid_to_client_shm_managers = {}


def get_server_shm_manager():
    pid = os.getpid()
    if pid in _pid_to_server_shm_manager:
        return _pid_to_server_shm_manager[pid]
    else:
        manager = _ServerShmManager()
        _pid_to_server_shm_manager[pid] = manager
        return manager


def get_client_shm_managers():
    pid = os.getpid()
    if pid in _pid_to_client_shm_managers:
        return _pid_to_client_shm_managers[pid]
    else:
        managers = _ClientShmManagers()
        _pid_to_client_shm_managers[pid] = managers
        return managers


def _cleanup_shm():
    logger.debug("Cleanup shm at exit called")
    global _EXIT_CALLED
    _EXIT_CALLED = True
    get_server_shm_manager().clear_all()
    get_client_shm_managers().clear_all()


_registered_process = set()
_EXIT_CALLED = False


def register_cleanup_this_process():
    pid = os.getpid()
    if pid not in _registered_process:
        logger.debug("Register cleanup function for pid = {}".format(pid))
        safe_exit.register(_cleanup_shm)



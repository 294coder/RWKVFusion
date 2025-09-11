import json
import logging
import os
import os.path as osp
import signal
import time
import warnings
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial, wraps
from importlib import reload
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import accelerate
import loguru
import numpy as np
import shortuuid
import torch
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.misc import NameSpace, default, get_process_index, is_main_process


def get_time(sec):
    h = int(sec // 3600)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    return h, m, s


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            start = self.start
        except AttributeError:
            start = self.start = time.time()

        time_elapsed = get_time(time.time() - start)

        record.relative = "{0}:{1:02d}:{2:02d}".format(*time_elapsed)

        # self.last = record.relativeCreated/1000.0
        return True


def save2json_file(d: dict, path: str, mode: str = "w", indent: int = 4):
    assert path.endswith(".json"), "@path should end with .json"
    with open(path, mode) as f:
        json.dump(d, f, indent=indent)
    print(f"save json file in {path}")


def loss_dict2str(
    loss_dict: "dict[int, float | torch.Tensor]", world_size: int = 1, round_fp: int = 6
) -> str:
    log_str = ""
    for k, v in loss_dict.items():
        log_str += f"{k}: {v / world_size:.{round_fp}f} "
    return log_str


## decrepted
class TrainStatusLogger(object):
    def __init__(
        self, id="None", path="./train_status/status.pt", resume=False, args=None
    ):
        """
        track training status as a context manager
        :param id: run id, which is defined in parser args
        :param path: pkl file's path
        :param resume: resume training. if you want to resume last training run, set id=None.
                       if you want to specify one run to resume, pass a specified id.
        """
        self.time_now = datetime.now()
        self.path = path
        self.id = id
        self.resume = resume
        self.status = {
            "id": id,
            "status": "running",
            "time_stamp": str(self.time_now.timestamp()),
            "time": self.time_now.strftime("%Y-%m-%d, %H:%M:%S"),
            "args": args,
        }
        self._base_path = os.path.dirname(self.path)
        if not os.path.exists(self.path):
            if not os.path.exists(self._base_path):
                os.mkdir(self._base_path)
            self.status_all = [self.status]
        else:
            self.status_all = self.load_train_status()
            self.check_unique_id()
            if resume:
                if id == "None":
                    self.status, _ = self.find_last_untrained_status()
                else:
                    self.status = self.find_run_by_id(id)
                    # print('warning: argument @id is not equal to the resume id which will be ignored')
            else:
                self.status_all.append(self.status)

        # ======= handle the KeyboardInterrupt signal =======
        def handler(*args):
            print("catch signal: KeyboardInterrupt")
            print("EXITTING...")
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handler)

    @staticmethod
    def _check_status_legal(status):
        assert status in ("running", "done", "break")

    def load_train_status(self):
        if os.path.getsize(self.path) > 0:
            # with open(self.path, "rb") as f:
            #     l = pickle.load(f)
            l = torch.load(self.path)
        else:
            raise EOFError("file is empty, you should delete it")
        print("load previous train status")
        return l

    def save_train_status(self):
        # with open(self.path, "wb") as f:
        #     pickle.dump(self.status_all, f)
        torch.save(self.status_all, self.path)
        print("save all train status")

    def update_train_status(self, status):
        self._check_status_legal(status)
        self.status["status"] = status

    def find_last_untrained_status(self):
        f_sort = lambda d: d["time_stamp"] if d["status"] == "break" else "0"
        last_status = sorted(self.status_all, key=f_sort)[-1]
        return last_status, last_status["id"]

    def find_run_by_id(self, id):
        s = self._find_id(id)
        if s["status"] != "break":
            return s
        raise AttributeError(
            f"no id: {id} in not an existing run or has already been done"
        )

    def _find_id(self, id):
        for s in self.status_all:
            if s["id"] == id:
                return s

    def print_status_by_id(self, id):
        s = self._find_id(id)
        for k, v in s:
            if isinstance(v, NameSpace):
                print(v)
            else:
                print(f"{k}: {v}")

    def check_unique_id(self):
        ids = []
        for d in self.status_all:
            ids.append(d["id"])
        assert len(ids) == len(np.unique(ids)), "exist id conflict"
        assert self.status["id"] not in ids or self.resume, (
            "id conflicts, check your run id "
            "or delete all tracker pkl file. "
            f"the pkl file can be found in {self.path}"
        )

    def __enter__(self):
        nbreak = 0
        ndone = 0
        for d in self.status_all:
            s = d["status"]
            if s == "done":
                ndone += 1
            elif s == "break":
                nbreak += 1

        print("=" * 20, "Log Train Process", "=" * 20, sep="")
        print(f"all runs: {ndone} run(s) done, {nbreak} run(s) break")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f'traceback: {exc_tb}')
        if exc_type is not None or exc_val is not None:
            print("=" * 20, "Find Error Happen", "=" * 20, sep="")
            print(f"catch error type: {exc_type}, error value: {exc_val}")
            self.update_train_status("break")
        else:
            print("=" * 20, "Training End", "=" * 20, sep="")
            self.update_train_status("done")

            # only save in main process
            if is_main_process():
                self.save_train_status()

    def __repr__(self):
        def dict_str(d):
            s = "id: {:<10}  status: {:<7}  time_stamp: {:<20}  time: {:<20}".format(
                d["id"], d["status"], d["time_stamp"], d["time"]
            )
            return s

        repr = ""
        for d in self.status_all:
            repr += dict_str(d) + "\n"
        return repr


def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


# * ========================================================================
# * logging utils
# * 1. logging version
# * 2. loguru version
# * ========================================================================


def easy_logger(
    func_name: str | None = None,
    level: str | None = "INFO",
    format_str: str | None = None,
    print_process_index: bool = False,
    only_main_process: bool = False,
    file_path: str | None = None,
    file_mode: str = "a+",
    *,
    never_save_in_file: bool = False,
):
    """
    Get a logger with rich handler and file handler.
    Markerup is used to colorize the log.

    Args:
        level (str | None): log level, default is 'INFO'
        format_str (str | None): log format string, default is None
        func_name (str | None): function name, default is None
        print_process_index (bool): whether to print process index, default is False
        never_save_in_file (bool): whether to save log in file, default is False

    Returns:
        ProtocalLogger: a custom logger object

    This function creates and configures a logger, supporting console and file output.
    It uses the Rich library for formatting and allows customizing the log format and function name.
    The returned ProtocalLogger object provides convenient logging methods.

    Env Args (set in os.environ or shell)
        LOGGER_LEVEL (str | None): log level, default is None
        LOG_FILE (str | None): log file path, default is None
    """

    reload(logging)
    if only_main_process:
        print_process_index = False
    _log_flag = (not only_main_process) or is_main_process()

    if format_str is not None and func_name is not None:
        format_str = "%(func_name)s: " + format_str
    elif format_str is None and func_name is not None:
        format_str = "%(func_name)s: %(message)s"
    else:
        format_str = "%(message)s"

    if print_process_index:
        format_str = "(rank-%(process_index)d) | " + format_str

    logger = logging.getLogger(func_name or "default_logger")
    level = level.upper()
    if os.getenv("LOGGER_LEVEL", None) is None:
        logger.setLevel(level)
    else:
        logger.setLevel(os.getenv("LOGGER_LEVEL"))

    file_name = file_path or os.getenv("LOG_FILE", None)
    # has file_handler or not
    # only main process can write log to file
    if file_name and is_main_process() and not never_save_in_file:
        file_console = Console(file=open(file_name, "a+"))
        file_handler = RichHandler(
            console=file_console, show_path=False, level=level, markup=True
        )
        file_handler.setFormatter(logging.Formatter(format_str, datefmt="%X"))
        logger.addHandler(file_handler)
    else:
        file_console = None
        file_handler = None

    rich_handler = RichHandler(show_path=False, level=level, markup=True)
    rich_handler.setFormatter(logging.Formatter(format_str, datefmt="%X"))
    logger.addHandler(rich_handler)

    # add func_name to prefix the log information
    class FuncNameFilter(logging.Filter):
        def filter(self, record):
            # make the `func_name` underlined
            record.func_name = f"[bold cyan underline]{func_name}[/bold cyan underline]"
            if print_process_index:
                record.process_index = get_process_index()
            return True

    class ProtocalLogger(object):
        # flags
        _file_handler_set: bool = False
        _not_save_in_file: bool = never_save_in_file

        # console and handler
        _console: Console = rich_handler.console
        _file_handler: logging.FileHandler = file_handler

        # warn once
        _warn_once_msg_set: set[str] = set()
        _warn_once_id_set: set[str] = set()

        def has_file_handler(self):
            # logger: logging.Logger
            return any(
                [
                    isinstance(hdl, (logging.FileHandler, RichHandler))
                    for hdl in logger.handlers
                ]
            )

        def is_file_handler_set(self):
            return self._file_handler_set

        def may_set_file_handler(self):
            if self.is_file_handler_set() or self._not_save_in_file:
                return
            else:
                file_name = os.environ.get("LOG_FILE", None)
                # has file_handler or not
                # only main process can write log to file
                if file_name and is_main_process():
                    file_console = Console(file=open(file_name, file_mode))
                    file_handler = RichHandler(
                        console=file_console, show_path=False, level=level, markup=True
                    )
                    file_handler.setFormatter(
                        logging.Formatter(format_str, datefmt="%X")
                    )
                    logger.addHandler(file_handler)
                    self._file_handler_set = True

        def print(
            self,
            *msg,
            sep=" ",
            level: "str | int" = "INFO",
            violate_only_main: bool = False,
        ):
            msg = sep.join(map(str, msg))
            if isinstance(level, str):
                level = eval(f"logging.{level.upper()}")
            self.may_set_file_handler()
            if _log_flag or violate_only_main:
                if violate_only_main and not print_process_index:
                    # print process index info
                    msg = f"(rank-{get_process_index()}) | {msg}"
                logger.log(level, msg, extra={"markup": True})

        def debug(self, *msg, sep=" ", level: "str | int" = "INFO"):
            self.print(*msg, sep, level="DEBUG")

        def info(self, *msg, sep=" ", level: "str | int" = "INFO"):
            self.print(*msg, sep, level="INFO")

        def warning(
            self,
            *msg,
            sep=" ",
            level: "str | int" = "INFO",
            violate_only_main: bool = False,
        ):
            self.print(*msg, sep, level="WARNING", violate_only_main=violate_only_main)

        def error(
            self,
            *msg,
            sep=" ",
            raise_error: bool = False,
            error_type=None,
            violate_only_main: bool = False,
        ):
            msg = sep.join(map(str, msg))
            self.print(*msg, sep, level="ERROR", violate_only_main=violate_only_main)
            if raise_error:
                if error_type is not None:
                    raise error_type(msg)

                raise RuntimeError(msg)

        def warn_once(
            self,
            *msg: str,
            sep: str = " ",
            level: str = "WARNING",
            warn_once_id: str | None = None,
            violate_only_main: bool = False,
        ):
            msg_str = None

            if warn_once_id is not None:
                _not_warn = warn_once_id not in self._warn_once_id_set
                if _not_warn:
                    self._warn_once_id_set.add(warn_once_id)
            else:
                msg_str = sep.join(
                    map(str, msg)
                )  # to avoid the msg is not hashable since it is a tuple
                _not_warn = msg_str not in self._warn_once_msg_set
                if _not_warn:
                    self._warn_once_msg_set.add(msg_str)

            if _not_warn:
                if msg_str is None:
                    msg_str = sep.join(map(str, msg))
                msg = f"[red][Warning Once][/red] {msg_str}"
                self.print(msg, sep, level=level, violate_only_main=violate_only_main)

    # add filter to add func_name to log
    if func_name is not None:
        logger.addFilter(FuncNameFilter())

    return ProtocalLogger()


from loguru import _logger, logger


class LoguruLogger:
    _logger = logger
    console = None
    handler = []

    _first_import = True
    _default_file_format = (
        "<green>[{time:MM-DD HH:mm:ss}]</green> "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> "
        "<level>[{level}]</level> - <level>{message}</level>"
    )
    _default_console_format = (
        "[{time:MM-DD HH:mm:ss}] "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> "
        "<level>[{level}]</level> - <level>{message}</level>"
    )

    @classmethod
    def logger(cls, sink=None, format=None, filter=None, **kwargs) -> "_logger.Logger":
        reload(loguru)

        if cls._first_import:
            cls._logger.remove()  # the first time import
            cls.console = Console(color_system=None)
            handler = cls._logger.add(
                default(sink, lambda *x: cls.console.print(*x, end="")),
                colorize=True,
                format=default(format, cls._default_console_format),
                **kwargs,
            )
            cls.handler.append(handler)

            cls._first_import = False

        else:
            if sink is not None:
                sink = default(sink, os.environ.get("LOG_FILE", None))
                handler = cls._logger.add(
                    sink,
                    format=default(format, cls._default_file_format),
                    filter=filter,
                    **kwargs,
                )
                if sink is not None:
                    cls._logger.info(f"loguru: sink is set to {sink}")

                cls.handler.append(handler)

        return cls._logger

    @classmethod
    def add(cls, *args, **kwargs):
        handler = cls._logger.add(*args, **kwargs)
        cls.handler.append(handler)

    @classmethod
    def remove_all(cls):
        for h in cls.handler:
            cls._logger.remove(h)
        cls.handler = []

    @classmethod
    def remove_id(cls, id):
        cls._logger.remove(id)

    @classmethod
    def bind(cls, *args, **kwargs):
        return cls._logger.bind(*args, **kwargs)


def catch_any_error(func: "Callable | None" = None):
    @contextmanager
    def error_catcher():
        try:
            logger = LoguruLogger.logger()
            yield logger
        except Exception as e:
            logger.error(f"catch error: {e}", raise_error=True)
            logger.exception(e)
        finally:
            LoguruLogger.remove_all()

    if func is None:
        return error_catcher()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with error_catcher() as logger:
            return func(*args, **kwargs)

    return wrapper


def get_logger(
    base_path: str | None = None,
    run_name: str = None,
    method_name: str = "default_run",
    dataset_name: str = "default_dataset",
    std_level=logging.INFO,
    file_level: tuple | int = (logging.DEBUG,),
    file_handler_names: tuple[str] | str = ("debug",),
    file_mode: str = "w",
    show_pid: bool = False,
    method_dataset_as_prepos=True,
    args: NameSpace | None = None,  # deprecated
    **discarded_kwargs,  # to avoid unused kwargs
) -> tuple[logging.Logger, list[logging.Handler], str]:
    """
    get logger to log
    :param base_path: such like './log/'
    :param name: logger name such as 'train_epoch_300'
    :param std_level: stream level
    :param file_level: file level
    :param file_handler_names:
    :param file_mode: 'a' append, 'w' write
    :param show_pid: show thread id
    :return: logger and List[handlers]
    """
    reload(logging)

    # decrepted `args`
    if args is not None:
        warnings.warn("args is deprecated to specify logger configurations")

    if base_path is not None:
        assert run_name is not None, "@param name should not be None"

    if not show_pid:
        format_str = "%(message)s"
    else:
        format_str = "pid: %(thread)d) %(message)s"
    rich_handler = RichHandler(show_path=False, markup=True)
    rich_handler.setFormatter(logging.Formatter(format_str, datefmt="%X"))

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    hdls = []

    if base_path is not None:
        assert len(file_handler_names) == len(
            file_level
        ), "`file_handler_names` and `file_level` should be list and equal length"
        for n, level in zip(file_handler_names, file_level):
            if method_dataset_as_prepos:
                method_name = args.full_arch if args is not None else method_name
                dataset_name = args.dataset if args is not None else dataset_name
                # /base_path/method_name/dataset_name/run_name
                file_log_dir = os.path.join(
                    base_path, method_name, dataset_name, run_name
                )
            else:
                # /base_path/run_name
                file_log_dir = os.path.join(base_path, run_name)
            if not os.path.exists(file_log_dir):
                os.makedirs(file_log_dir)
                print(f"logging: make log file [{os.path.abspath(file_log_dir)}]")

            # log file
            file_log_path = os.path.join(file_log_dir, n + ".log")
            file_console = Console(file=open(file_log_path, "a+"))
            file_handler = RichHandler(
                console=file_console, show_path=False, markup=True
            )
            file_handler.setLevel(level)

            # add file handler
            hdls.append(file_handler)
            logger.addHandler(file_handler)
    else:
        file_console = None
        file_handler = None
        file_log_dir = None
        file_log_path = None

    for handler in logger.handlers:
        handler.addFilter(TimeFilter())

    # log print function
    state = accelerate.state.PartialState()

    def log_print(*msg, sep=" ", level="INFO", dist=False, proc_id=None):
        if isinstance(level, str):
            level_int = eval(f"logging.{level}")

        if dist:
            with state.main_process_first():
                default_proc_id = default(proc_id, state.local_process_index)
                msgs = f"{default_proc_id=} | "
                msgs += sep.join(map(str, msg))
                logger.log(level=level_int, msg=msgs)
        elif state.is_main_process:
            msgs = sep.join(map(str, msg))
            logger.log(level=level_int, msg=msgs)

    # set some attributes
    logger.print = log_print
    logger._console = rich_handler.console
    logger._file_console = file_console
    logger._file_path = file_log_path

    return logger, hdls, file_log_dir


class NoneLogger:
    def __init__(self, args: NameSpace, name: str = "default", **discarded_kwargs):
        """
        args.logger_config.base_path
        args.logger_config.name
        args.dataset
        args.full_arch

        """

        class NoneWriter:
            def __init__(self) -> None:
                pass

            def close(self, *args, **kwargs):
                pass

        self.writer = NoneWriter()
        self.logger, *_ = get_logger(run_name=name)

        # add time and run_id
        if isinstance(args, DictConfig):
            args = NameSpace.dict_to_namespace(
                OmegaConf.to_container(args, resolve=True)
            )

        args.logger_config.name = (
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            + "_"
            + args.logger_config.name
        )
        run_id = args.default_none_getattr("run_id")
        if run_id is not None:
            args.logger_config.name += "_" + run_id + f"_{args.comment}"
        else:
            args.logger_config.name += f"_{args.comment}"
        name = args.logger_config.name

        self.log_file_dir = os.path.join(
            args.logger_config.base_path, args.full_arch, args.dataset, name
        )

    @property
    def console(self):
        return self.logger._console

    @property
    def file_console(self):
        return self.file_console._file_console

    def watch(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_images(self, *args, **kwargs):
        pass

    def log_curve(self, *args, **kwargs):
        pass

    def log_curves(self, *args, **kwargs):
        pass

    def log_network(self, *args, **kwargs):
        pass

    def print(self, *msg, level="INFO", dist=False, proc_id=None):
        if dist or is_main_process():
            if isinstance(level, str):
                level_int = eval(f"logging.{level}")
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)

    def info(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.INFO
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)

    def debug(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.DEBUG
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)

    def warning(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.WARNING
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)

    def error(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.ERROR
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)


# * ==========================================================
# * Tensorboard Logger


class TensorboardLogger:
    def __init__(
        self,
        logger_cfgs: dict = dict(
            method_name="default_run",
            dataset_name="default_dataset",
            comment="",
            run_id=None,
            base_path="log_file/",
        ),
        tsb_logdir=None,
        file_stream_log=True,
        method_dataset_as_prepos=False,
        tsb_comment=None,  # only to Tensorboard comment, not used
        *,
        args: "NameSpace | DictConfig | None" = None,
    ):
        """

        Args:
            logger_cfgs: logger config
            tsb_logdir: tensorboard log dir
            tsb_comment: tensorboard comment
            file_stream_log: file stream dir
            config_file_mv: where arch_config.yaml dir at
            args: config args from main.py
                attributes:
                    logger_config.name
                    comment
                    full_arch
                    dataset
        """
        self.grad_dict = {}
        self.hooks = {}
        self.watch_type = "None"
        self.freq = 10

        # use args or logger cfgs
        def get_run(args, logger_cfgs: dict):
            if isinstance(args, DictConfig):
                args = NameSpace.dict_to_namespace(
                    OmegaConf.to_container(args, resolve=True)
                )

            if args is not None:
                # add time and run_id
                args.logger_config.name = (
                    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    + "_"
                    + args.logger_config.name
                )
                if args.default_none_getattr("run_id") is not None:
                    args.logger_config.name += "_" + args.run_id + f"_{args.comment}"
                else:
                    assert (
                        args.default_none_getattr("comment") is not None
                    ), "`args.comment` should not be None"
                    args.logger_config.name += f"_{args.comment}"

                # take out the args
                run_name = args.logger_config.name
                method_name = args.full_arch
                dataset_name = args.dataset

            else:  # use logger_cfgs
                # add time and run_id
                run_name = (
                    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    + "_"
                    + logger_cfgs["method_name"]
                )
                if logger_cfgs["run_id"] is not None:
                    run_name += (
                        "_" + logger_cfgs["run_id"] + f"_{logger_cfgs['comment']}"
                    )
                else:
                    assert (
                        logger_cfgs["comment"] is not None
                    ), "`logger_cfgs['comment']` should not be None"
                    run_name += (
                        f"_{logger_cfgs['comment']}"
                        if logger_cfgs["comment"] != ""
                        else ""
                    )

            return run_name, method_name, dataset_name

        run_name, method_name, dataset_name = get_run(args, logger_cfgs)

        self.logger_name = run_name

        if file_stream_log:
            self.file_logger, self.file_hdls, self.log_file_dir = get_logger(
                # **args.logger_config.to_dict(),
                # args=args,
                base_path=logger_cfgs["base_path"],
                run_name=run_name,
                dataset_name=dataset_name,
                method_name=method_name,
                method_dataset_as_prepos=method_dataset_as_prepos,
            )
            config_cp_path = os.path.join(self.log_file_dir, "config.json")
            save2json_file(args.to_dict(), config_cp_path)
            self.print(
                f"\nmove config file to {os.path.abspath(self.log_file_dir)}",
                level="INFO",
            )
        else:
            # TODO: add non-file logger
            ...

        self.writer = SummaryWriter(default(tsb_logdir, self.log_file_dir), tsb_comment)

        # set file path to os.environ
        os.environ["LOG_FILE"] = self.logger_file_path
        self.file_logger.debug(f"set os environ `LOG_FILE`={self.logger_file_path}")

        # warn once set
        self._warn_once_id_set = {}

    @property
    def console(self) -> Console:
        return self.file_logger._console

    @property
    def logger_file_path(self) -> str:
        return self.file_logger._file_path

    @property
    def file_console(self):
        return self.file_console._file_console

    def check_tensor_float(self, x):
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)

        return x

    @is_main_process
    def watch(self, network: nn.Module, watch_type: str, freq: int):
        assert watch_type in (
            "all",
            "grad",
            "None",
        ), "@watch_type should only be all, grad or None"
        if watch_type == "None":
            return
        self.watch_type = watch_type
        self.freq = freq

        def _hook(grad, name):
            self.grad_dict[name] = grad

        for n, p in network.named_parameters():
            hook = partial(_hook, name=n)
            self.hooks[n] = hook
            p.register_hook(hook)

    @is_main_process
    def log_curve(self, x, name, epoch):
        if x is None:
            return

        self.writer.add_scalar(name, self.check_tensor_float(x), epoch)

    @is_main_process
    def log_curves(self, x_dict: Dict, epoch, *, prefix: str | None = None):
        """
        e.g.,

        for i in range(100):
            writer.add_scalars('run_14h', {'xsinx': i * np.sin(i / r),
                                           'xcosx': i * np.cos(i / r),
                                           'tanx': np.tan(i / r)}, i)
        """
        if x_dict is None:
            return

        for k, v in x_dict.items():
            if prefix is not None:
                k = f"{prefix}/{k}"
            self.writer.add_scalar(k, self.check_tensor_float(v), epoch)

    @is_main_process
    def log_image(self, image, name, epoch):
        if image is None:
            return

        if image.ndim == 3:
            assert image.shape[0] <= 3, (
                f"the number of image channel "
                f"should not greater than 3 but got shape {image.shape}"
            )
        self.writer.add_image(
            name, self.check_tensor_float(image), epoch, dataformats="CHW"
        )

    @is_main_process
    def log_images(
        self,
        batch_imgs: Sequence,
        nrow: int,
        names: Sequence,
        task: str,
        epoch: int,
        ds_name: str | None = None,
        **grid_kwargs,
    ):
        assert task in ["fusion", "sharpening"], "@task should be fusion or sharpening"

        if batch_imgs is None:
            return

        for batch_img, name in zip(batch_imgs, names):
            from .visualize import get_spectral_image_ready

            batch_img = get_spectral_image_ready(
                self.check_tensor_float(batch_img), name, task, ds_name
            )
            grid_img = make_grid(batch_img, nrow=nrow, **grid_kwargs)
            self.log_image(grid_img, name, epoch)

    @is_main_process
    def log_network(self, network: nn.Module, ep: int):
        if self.watch_type != "None":
            if ep % self.freq == 0:
                for (_, g), (n, p) in zip(
                    self.grad_dict.items(), network.named_parameters()
                ):
                    if self.watch_type == "all":
                        self.writer.add_histogram(n + "_data", p.flatten().float(), ep)
                    self.writer.add_histogram(n + "_grad", g.flatten().float(), ep)

    def print(self, *msg, level="INFO", dist=False, proc_id=None):
        """
        Unified print method to handle all log levels.
        """
        if dist or is_main_process():
            if isinstance(level, str):
                level_int = eval(f"logging.{level.upper()}")
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            msgs += " ".join(map(str, msg))
            self.file_logger.log(level=level_int, msg=msgs)

    def info(self, *msg, dist=False, proc_id=None):
        self.print(*msg, level="INFO", dist=dist, proc_id=proc_id)

    def debug(self, *msg, dist=False, proc_id=None):
        self.print(*msg, level="DEBUG", dist=dist, proc_id=proc_id)

    def warning(self, *msg, dist=False, proc_id=None):
        self.print(*msg, level="WARNING", dist=dist, proc_id=proc_id)

    def error(
        self, *msg, dist=False, proc_id=None, raise_error: bool = False, error_type=None
    ):
        self.print(*msg, level="ERROR", dist=dist, proc_id=proc_id)
        if raise_error:
            if error_type is not None:
                raise error_type(" ".join(map(str, msg)))
            raise RuntimeError(" ".join(map(str, msg)))

    def warn_once(
        self,
        *msg: str,
        level: str = "WARNING",
        warn_once_id: str | None = None,
        dist: bool = False,
        proc_id: str | None = None,
    ):
        """
        avoid multi-GPU training print the same warning in each GPU
        """
        if warn_once_id is not None:
            _not_warn = warn_once_id not in self._warn_once_id_set
            if _not_warn:
                self._warn_once_id_set.add(warn_once_id)
        else:
            msg_str = " ".join(
                map(str, msg)
            )  # to avoid the msg is not hashable since it is a tuple
            _not_warn = msg_str not in self._warn_once_msg_set
            if _not_warn:
                self._warn_once_msg_set.add(msg_str)

        if _not_warn:
            self.print(*msg, level=level, dist=dist, proc_id=proc_id)


# * ====================================================================
# * experimental logger and weights specification


def experimental_logger(exp_name: str = "MainExp", config_args: NameSpace = None):
    state = PartialState()
    if state.is_main_process:
        # unify tensorboard logger and file logger
        # will save config file to log dir

        # need args.full_arch, args.dataset
        logger = TensorboardLogger(
            args=config_args,
            tsb_comment=config_args.comment,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
    else:
        logger = NoneLogger(cfg=config_args, name=exp_name)

    return logger


def specify_weights_path(
    args: NameSpace | DictConfig, logger, accelerator: "accelerate.Accelerator"
):
    """
    args.output_dir: accelerate.project_configuration.project_dir
    args.save_model_path: ema model path
    args.save_base_path: place holder
    """

    if accelerator.is_main_process:
        weight_path = osp.join(logger.log_file_dir, "weights")
        if accelerator.is_main_process:
            os.makedirs(weight_path, exist_ok=True)
        args.output_dir = weight_path
        args.save_model_path = osp.join(args.output_dir, "ema_model.pth")
        args.save_base_path = args.output_dir
    else:
        # for ddp broadcast
        args.output_dir = args.save_model_path = args.save_base_path = [None]

    if accelerator.use_distributed:
        (args.output_dir, args.save_model_path, args.save_base_path) = (
            accelerate.utils.broadcast_object_list(
                [args.output_dir, args.save_model_path, args.save_base_path]
            )
        )
    accelerator.project_configuration.project_dir = args.output_dir
    logger.info(f"model weights will be save at {args.output_dir}")

    return args


if __name__ == "__main__":

    def main():
        logger = easy_logger(func_name="main")
        logger.info("hello")
        logger.info("world")
        logger.info("hello world")
        raise ValueError("123")

        logger.info("hello [bold cyan underline]world[/bold cyan underline]")
        logger.warn_once("warn", "once", warn_once_id="1", violate_only_main=True)
        logger.warn_once("warn", "once", warn_once_id="2")
        logger.warn_once("warn", "once", warn_once_id="3")

    main()

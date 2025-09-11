from contextlib import nullcontext
from typing import Callable, List, Union

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .log_utils import easy_logger


class EasyProgress:
    tbar: "Progress | None" = None
    task_desp_ids: dict[str, int] = {}
    logger = easy_logger(func_name="EasyProgress")
    _is_debug: bool = False

    @classmethod
    def console(cls):
        assert cls.tbar is not None, "`tbar` has not initialized"
        return cls.tbar.console

    @classmethod
    def close_all_tasks(cls, remove_task: bool = True):
        if cls.tbar is not None:
            for task_id in cls.tbar.task_ids:
                cls.tbar.stop_task(task_id)
                # set the task_id all unvisible
                cls.tbar.update(task_id, visible=False)
                if remove_task:
                    cls.tbar.remove_task(task_id)

        if remove_task:
            cls.task_desp_ids.clear()

    @classmethod
    def close_task_id(cls, task_id: int, remove_task: bool = False):
        if cls.tbar is not None:
            cls.tbar.stop_task(task_id)
            cls.tbar.update(task_id, visible=False)
            if remove_task:
                cls.tbar.remove_task(task_id)
                for desp, id in cls.task_desp_ids.items():
                    if id == task_id:
                        cls.task_desp_ids.pop(desp)

    @classmethod
    def easy_progress(
        cls,
        task_desciptions: list[str],
        task_total: list[int],
        tbar_kwargs: dict = {},
        task_kwargs: list[dict[str, Union[str, int]]] = None,
        is_main_process: bool = True,
        *,
        start_tbar: bool = True,
        debug: bool = False,
    ) -> tuple[Progress, Union[list[int], int]]:
        """get a rich progress bar

        Args:
            task_desciptions (list[str]): list of task descriptions of `len(task_desciptions)` tasks
            task_total (list[int]): list of length each task
            tbar_kwargs (dict, optional): kwargs for progress bar. Defaults to {}.
            task_kwargs (list[dict[str, Union[str, int]]], optional): task kwargs for each task. Defaults to None.
            is_main_process (bool, optional): if is main process. Defaults to True.
            start_tbar (bool, optional): start running progress bar when init. Defaults to True.
            debug (bool, optional): debug mode, set progress bar to be unvisible. Defaults to False.

        Returns:
            tuple[Progress, Union[list[int], int]]: Progress bar and task ids
        """

        def _add_task_ids(tbar: Progress, task_desciptions, task_total, task_kwargs):
            task_ids = []
            if task_kwargs is None:
                task_kwargs = [{"visible": False}] * len(task_desciptions)
            for task_desciption, task_total, id_task_kwargs in zip(
                task_desciptions, task_total, task_kwargs
            ):
                if task_desciption in list(EasyProgress.task_desp_ids.keys()):
                    task_id = EasyProgress.task_desp_ids[task_desciption]
                    cls.tbar.start_task(task_id)
                    task_ids.append(task_id)
                else:
                    task_id = tbar.add_task(
                        task_desciption, total=task_total, **id_task_kwargs
                    )
                    task_ids.append(task_id)
                    EasyProgress.task_desp_ids[task_desciption] = task_id

            return task_ids if len(task_ids) > 1 else task_ids[0]

        def _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                if task_kwargs is not None:
                    assert len(task_desciptions) == len(task_total) == len(task_kwargs)
                else:
                    assert len(task_desciptions) == len(task_total)

                # if (console := tbar_kwargs.pop('console', None)) is not None:
                #     console._color_system = console._detect_color_system()

                # if 'console' in tbar_kwargs:
                #     tbar_kwargs['console']._color_system = tbar_kwargs['console']._detect_color_system()

                tbar = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    SpinnerColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    **tbar_kwargs,
                )
                EasyProgress.tbar = tbar

                task_ids = _add_task_ids(
                    tbar, task_desciptions, task_total, task_kwargs
                )

                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(
                    task_desciptions
                ) > 1 else None

        def _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                tbar = EasyProgress.tbar

                task_ids = []
                if task_kwargs is None:
                    task_kwargs = [{"visible": False}] * len(task_desciptions)

                task_ids = _add_task_ids(
                    tbar, task_desciptions, task_total, task_kwargs
                )

                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(
                    task_desciptions
                ) > 1 else None

        cls._is_debug = debug
        if not debug:
            if EasyProgress.tbar is not None:
                rets = _cached_tbar_with_new_task_ids(
                    task_desciptions, task_total, task_kwargs
                )
            else:
                rets = _new_tbar_with_task_ids(
                    task_desciptions, task_total, task_kwargs
                )
            if start_tbar and is_main_process and not EasyProgress.tbar.live._started:
                EasyProgress.tbar.start()
            return rets
        else:
            return nullcontext(), [None] * len(task_desciptions) if len(
                task_desciptions
            ) > 1 else None

    @classmethod
    def start_progress_bar(cls):
        if isinstance(EasyProgress.tbar, Progress):
            if (
                EasyProgress.tbar is not None
                and not EasyProgress.tbar.live._started
                and not cls.is_debug()
            ):
                EasyProgress.tbar.start()

    @classmethod
    def is_debug(cls, warn: bool = True):
        if cls._is_debug:
            cls.logger.warning("in debug mode, options on Progress bar will be ignored")
            return cls._is_debug

    @classmethod
    def reset_task_id(cls, task_id: int):
        if cls.is_debug():
            return
        if task_id not in cls.tbar.task_ids:
            cls.logger.warning(f"task id {task_id} not in tbar")
            return

        cls.tbar.reset(task_id)

    @classmethod
    def update_task_id(
        cls,
        task_id: int,
        *,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields,
    ):
        if cls.is_debug():
            return
        if task_id not in cls.tbar.task_ids:
            cls.logger.warning(f"task id {task_id} not in tbar")
            return

        cls.tbar.update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

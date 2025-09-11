import os
import re
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch

from utils.log_utils import easy_logger
from utils.metric_fusion import AnalysisFusionAcc, MetricsByTask

logger = easy_logger(func_name="fusion_metric")


def read_tensor_img_from_path(path, mode):
    # return read_image(path).float()[None].cuda() / 255.
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)
    img = torch.tensor(np.array(img)).float() / 255.0
    if img.ndim == 2:
        img = img[None, None]
    else:
        img = img.permute(2, 0, 1)[None]
    return img.cuda()


def calculate_VISIR_metric_from_file(
    fused_path, ir_path, vi_path, metric: AnalysisFusionAcc
):
    if not metric.only_on_y_component:
        read_mode = "RGB"
    else:
        read_mode = None
    fused = read_tensor_img_from_path(fused_path, read_mode)
    ir = read_tensor_img_from_path(ir_path, read_mode)
    vi = read_tensor_img_from_path(vi_path, read_mode)

    metric((vi, ir), fused)


def only_contain_imgs(path):
    path = Path(path)
    _N = 20
    i = 0
    for file in path.iterdir():
        if file.suffix not in [".png", ".jpg", ".bmp"]:
            return False
        i += 1
        if i > _N:
            return True
    return True


def fused_name_to_int_str_name(fused_name):
    str_num = re.compile(r"(\d+)").findall(fused_name)[0]
    # unzfill str
    fused_name = str(int(str_num))

    return fused_name + ".png"


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from tqdm import tqdm

    torch.cuda.set_device("cuda:0")

    dataset_paths_by_task = {
        "TNO": [
            "data/VIF/VIF-TNO/new_test_data/vi",
            "data/VIF/VIF-TNO/new_test_data/ir",
        ],
        "RoadScene": [
            "data/VIF/VIF-RoadSceneFusion/test/vi test",
            "data/VIF/VIF-RoadSceneFusion/test/ir test",
        ],
        "MSRS": [
            "data/VIF/VIF-MSRS/test/raw_png/vi",
            "data/VIF/VIF-MSRS/test/raw_png/ir",
        ],
        "MEFB": [
            "data/MEF/MEF-MEFB/OVER",
            "data/MEF/MEF-MEFB/UNDER",
        ],
        "SICE": [
            "data/MEF/MEF-SICE/over",
            "data/MEF/MEF-SICE/under",
        ],
        "MFF_WHU": [
            "data/MFF/MFI-WHU/source_2",
            "data/MFF/MFI-WHU/source_1",
        ],
        "Lytro": [
            "data/MFF/MFF-Lytro/FAR",
            "data/MFF/MFF-Lytro/NEAR",
        ],
        "RealMFF": [
            "data/MFF/MFF-RealMFF/FAR",
            "data/MFF/MFF-RealMFF/NEAR",
        ],
        "MedHarvard": [
            "data/MIF/MedHarvard/xmu/SPECT-MRI/SPECT",
            "data/MIF/MedHarvard/xmu/SPECT-MRI/MRI",
        ],
        "M3FD": [
            "data/VIF/VIF-M3FD/M3FD_Fusion/vi",
            "data/VIF/VIF-M3FD/M3FD_Fusion/ir",
        ],
    }["Lytro"]

    fused_dir = "/Data3/cao/ZiHanCao/exps/panformer/visualized_img/RWKVFusion_v12_RWKVFusion/lytro_11_22_v1"
    vi_dir = dataset_paths_by_task[0]
    ir_dir = dataset_paths_by_task[1]
    implem_by = "torch"

    # ================================ metric config ================================
    metrics_by_task = MetricsByTask.ALL
    metric = AnalysisFusionAcc(
        only_on_y_component=False, implem_by=implem_by, test_metrics=metrics_by_task
    )
    _test_metrics = metric.tested_metrics
    _metrics_better_order = metric.metrics_better_order
    # =================================================================================

    # table
    console = Console()
    table = Table(title="Image Fusion Metrics Table")

    table.add_column("Methods", style="cyan", no_wrap=True, header_style="bold")
    for metric_name, metric_better_order in zip(_test_metrics, _metrics_better_order):
        table.add_column(
            f"{metric_name}({metric_better_order})", style="cyan", no_wrap=True
        )

    # if fuse_dir is the test dir
    if only_contain_imgs(fused_dir):
        logger.info(f"only contain images")
        fused_dir = Path(fused_dir)
        logger.info(f"start calculating VIS IR metric for {fused_dir}")
        _error_flag = False
        pbar = tqdm(os.listdir(fused_dir), desc="Calculating VIS IR metric")
        for fused_name in pbar:
            try:
                fused_path = os.path.join(fused_dir, fused_name)
                ir_path = os.path.join(ir_dir, fused_name)
                vi_path = os.path.join(vi_dir, fused_name)
                calculate_VISIR_metric_from_file(fused_path, ir_path, vi_path, metric)
            except Exception as e:
                logger.warning(
                    f"error met {e} in calculating VIS IR metric for {fused_path}, {ir_path}, {vi_path}",
                    "try using different extension",
                )
                _error_flag = True

            # try different extension
            if _error_flag:
                ext_ori = Path(fused_name).suffix
                EXTENSIONS_TRY = [".png", ".jpg", ".bmp", ".tiff", ".tif"]
                for ext in EXTENSIONS_TRY:
                    fused_name_tmp = fused_name.replace(ext_ori, ext)
                    if fused_name_tmp in os.listdir(vi_dir):
                        fused_name = fused_name_tmp
                        ir_path = os.path.join(ir_dir, fused_name)
                        vi_path = os.path.join(vi_dir, fused_name)
                        try:
                            calculate_VISIR_metric_from_file(
                                fused_path, ir_path, vi_path, metric
                            )
                            _error_flag = False
                            break
                        except Exception as e:
                            logger.warning(
                                f"error {e} met in calculating VIS IR metric for {fused_path}, {ir_path}, {vi_path}"
                            )
                            _error_flag = True
                            continue

                # if still error, skip
                if _error_flag:
                    logger.error(
                        f"error still exists in calculating VIS IR metric for {fused_path}, {ir_path}, {vi_path}"
                    )
                    break

            pbar.set_description(
                f"Calculating VIS IR metric [{metric._call_n}/{len(os.listdir(fused_dir))}]"
            )

        if not _error_flag and metric._call_n == len(os.listdir(fused_dir)):
            logger.info(f"VIS IR metric: {metric}")

            acc = metric.acc_ave
            table.add_row(
                Path(fused_dir).name,
                *[
                    str(round(acc[m], 4)) if acc.get(m, None) is not None else "N/A"
                    for m in _test_metrics
                ],
            )

            console.print(table)
            logger.info(f"finish computing VIS IR metric")
        else:
            logger.error(
                f"error in calculating VIS IR metric for {fused_dir} with {metric._call_n} images/total {len(os.listdir(fused_dir))}"
            )
            raise RuntimeError(
                f"error in calculating VIS IR metric for {fused_dir} with {metric._call_n} images/total {len(os.listdir(fused_dir))}"
            )
        exit()

    # if sub dir is method dir
    missed_methods = []
    if Path(fused_dir).is_dir():
        for method in Path(fused_dir).iterdir():
            if not method.is_dir() or method.name.startswith("__"):
                continue

            logger.info(f"start calculating VIS IR metric for {method.name}")
            for fused_name in tqdm(
                os.listdir(method), desc="Calculating VIS IR metric"
            ):
                fused_path = os.path.join(method, fused_name)
                # same name but different extension
                if fused_name not in os.listdir(vi_dir):
                    ext_ori = Path(fused_name).suffix
                    EXTENSIONS_TRY = [".png", ".jpg", ".bmp", ".tif"]
                    _not_found = True
                    for ext in EXTENSIONS_TRY:
                        fused_name_tmp = fused_name.replace(ext_ori, ext)
                        if fused_name_tmp in os.listdir(vi_dir):
                            fused_name = fused_name_tmp
                            _not_found = False
                            break
                    if _not_found:
                        logger.warning(
                            f"{fused_name} of {method.name} not found vi_dir, skip it"
                        )
                        break

                ir_path = os.path.join(ir_dir, fused_name)
                vi_path = os.path.join(vi_dir, fused_name)

                try:
                    calculate_VISIR_metric_from_file(
                        fused_path, ir_path, vi_path, metric
                    )
                    _error_flag = False
                except Exception as e:
                    logger.warning(
                        f"error {e} met in calculating VIS IR metric for {fused_path}, {ir_path}, {vi_path}"
                    )
                    _error_flag = True
                    continue

            if metric._call_n == 0 or _error_flag:
                logger.warning(f"skip {method.name} because no image found or error")
                missed_methods.append(method.name)
                continue

            logger.info(f"VIS IR metric: {metric}")

            acc = metric.acc_ave
            table.add_row(
                method.name,
                *[
                    str(round(acc[m], 4)) if acc.get(m, None) is not None else "N/A"
                    for m in _test_metrics
                ],
            )
            console.print(table)

            # clear metric fn
            metric.clear()

    if len(missed_methods) > 0:
        logger.warning(f"missed methods: {missed_methods}")

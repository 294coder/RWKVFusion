import os
import os.path as osp
from pathlib import Path
from typing import Literal, Tuple, TypeAlias, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float
from kornia.io import ImageLoadType, load_image
from kornia.io import write_image as write_image_k
from scipy.io import savemat
from torch import Tensor
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png

from utils.log_utils import easy_logger
from utils.misc import to_device, to_numpy, to_tensor

logger = easy_logger("visualize")


def get_rgb_channel_by_dataset_name(tensor, dataset_name: str):
    if dataset_name in ("wv3", "wv2"):
        return tensor[:, [4, 2, 0], ...]
    elif dataset_name in ("gf2", "qb"):
        return tensor[:, :3, ...]
    elif dataset_name in ("gf5", "gf5-gf1"):
        return tensor[:, [40, 30, 20], ...]
    elif dataset_name == "houston":
        return tensor[:, [39, 29, 19], ...]
    elif "cave" in dataset_name or "harvard" in dataset_name:
        return tensor[:, [29, 19, 9], ...]
    else:
        return tensor[:, :3, ...]


def permute_dim(*args):
    d = [i.permute(2, 0, 1) for i in args]
    return d


def normalize(img, to_uint8=True):
    """
    centering image to show
    :param img: numpy array, shape [H, W, C]
    :return: uint8 type image
    """
    img = img - img.min((0, 1))
    img = img / img.max((0, 1))
    if to_uint8:
        img *= 255
        img = img.astype("uint8")
    return img


def invert_normalized(
    norm_img: Union[Tensor, np.ndarray],
    mean: Union[Tensor, np.ndarray],
    std: Union[Tensor, np.ndarray],
    *,
    change_back_dim=True,
):
    """
    invert image normalized to unnormalized image
    Args:
        norm_img: Tensor: [C, H, W] or [B, C, H, W]
        mean: Tensor, [C, ]
        std: Tensor, [C, ]
        change_back_dim: bool, change channel dim back

    Returns: Ndarray, unnormalized image

    """
    if isinstance(norm_img, Tensor):
        norm_img = to_numpy(norm_img)[0]
    if isinstance(mean, Tensor):
        mean = to_numpy(mean)[0]
    if isinstance(std, Tensor):
        std = to_numpy(std)[0]
    if norm_img.ndim == 4:
        _dim_trans = [0, 2, 3, 1]
        _dim_back_trans = [0, -1, 1, 2]
    else:
        _dim_trans = [1, 2, 0]
        _dim_back_trans = [2, 0, 1]
    norm_img = norm_img.transpose(_dim_trans)

    assert norm_img.shape[-1] == mean.size == std.size
    unnormed_img = norm_img * std + mean  # [H, W, C] or [B, H, W, C]
    if change_back_dim:
        unnormed_img = unnormed_img.transpose(_dim_back_trans)
    return unnormed_img


def hist_equal(img):
    """
    equalize an image
    :param img: numpy array, shape [H, W, C], C can be any int
    :return:
    """
    if img.ndim == 3:
        for i in range(img.shape[-1]):
            img[..., i] = cv2.equalizeHist(img[..., i])
    else:
        img = cv2.equalizeHist(img)
    return img


def res_image(gt: Tensor, sr: Tensor, *, exaggerate_ratio: int = None) -> torch.Tensor:
    # shape [B, C, H, W]
    ratio = exaggerate_ratio if exaggerate_ratio is not None else 1.0
    res = torch.abs(gt - sr).mean(1, keepdim=True) * ratio
    return res


def get_spectral_image_ready(
    batch_image: Tensor,
    tensor_name: str,
    task: str = None,
    ds_name: Literal[
        "wv3",
        "wv2",
        "gf2",
        "qb",
        "gf5",
        "gf5-gf1",
        "flir",
        "tno",
        "msrs",
        "m3fd",
        "llvip",
        "medical_harvard",
    ] = None,
) -> Tensor:
    """
    get image channels ready, extracting image channels
        - fusion task: no modified
        - sharpening task: extract RGB channels
    Args:
        batch_image (Tensor): the input image to be processed, shape [B, C, H, W]
        tensor_name (str): the name of the input tensor, should be one of ("pan", "ms", "sr", "lms", "residual", "res")
        task (str, optional): the task type, should be one of ("fusion", "sharpening"). Defaults to None.
        ds_name (str): the name of the dataset, used to extract RGB channels for sharpening task. Defaults to None.

    Returns:
        Tensor: image ready for visualization, shape [B, C, H, W]

    """

    # batch_image: [B, C, H, W]
    img_arrs = batch_image.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]

    _default_tensor_name = ("pan", "ms", "sr", "lms", "residual", "res")
    if tensor_name not in _default_tensor_name:
        logger.warn_once(
            "tensor_name should be one of {_default_tensor_name}, but got {tensor_name}",
            warn_once_id="tensor_name",
        )

    if task == "fusion":
        transform_fn = lambda x, const: torch.as_tensor(x, dtype=torch.float32)
    elif task == "sharpening":
        if tensor_name in ("lms", "ms", "sr"):  # for pansharpening and HISR tasks
            batch_image = get_rgb_channel_by_dataset_name(batch_image, ds_name)
        elif tensor_name == "pan" and batch_image.shape[1] > 3:
            batch_image = batch_image[:, :3]
        else:
            raise ValueError(f"Invalid tensor name: {tensor_name}")

        transform_fn = lambda x, const: torch.as_tensor(
            normalize(x, to_uint8=False) * const
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    if "res" in tensor_name:
        _res_amplitude = 10
        equalized_img = [
            transform_fn(i, _res_amplitude).permute(-1, 0, 1)[None, ...]
            for i in img_arrs
        ]  # [1, C, H, W]
    else:
        equalized_img = [
            transform_fn(i, 1).permute(-1, 0, 1)[None, ...] for i in img_arrs
        ]

    cat_imgs = torch.cat(equalized_img, dim=0)

    return cat_imgs


def viz_batch(
    img: Tensor, base_path="./visualized_img", suffix=None, start_index=1, format="jpg"
):
    assert suffix is not None, "arg @suffix can not be None"
    assert suffix in [
        "pan",
        "ms",
        "sr",
        "gt",
        "residual",
    ], "arg @suffix should only be pan, ms or sr"
    img_arrs = img.permute(0, 2, 3, 1).numpy()
    if suffix == "residual":
        equalized_img = [i for i in img_arrs]
    else:
        # equalized_img = [hist_equal(normalize(i)) for i in img_arrs]
        equalized_img = [normalize(i) for i in img_arrs]

    path = osp.join(base_path, suffix)
    if not osp.exists(path):
        os.makedirs(path)
    # all_path = path + '.mat'
    # savemat(all_path, {f'{suffix}': img_arrs})

    # fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    # ax: plt.Axes
    # fig: plt.Figure
    for i, img in enumerate(equalized_img, start_index):
        h, w = img.shape[-2:]
        img_path1 = osp.join(path, str(i) + "." + format)
        # plt.cla()
        if suffix == "pan" or suffix == "residual":
            # ax.imshow(img, cmap='gray')
            cv2.imwrite(img_path1, img)
        # elif suffix == 'residual':
        #     ax.imshow(img)
        else:
            try:
                # ax.imshow(img[..., [0, 2, 4]])
                cv2.imwrite(img_path1, img[..., [0, 2, 4]])
            except:
                # ax.imshow(img[..., :3])
                cv2.imwrite(img_path1, img[..., :3])
    #     ax.set_axis_off()
    #     fig.set_size_inches(h, w)
    #     fig.savefig(img_path1, format=format, dpi=50, bbox_inches='tight', pad_inches=0.)
    # plt.close()


def show_details(
    img: np.ndarray,
    cpos_ratio: Tuple[float, float],
    area_pixels: Tuple[int, int],
    interp_ratio: int = 3,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    place: str = None,
) -> np.ndarray:
    """select a patch in raw image which decided by @cpos_ratio and @area_pixels,
    the function will interpolate @interp_ratio times and paste it in a corner.

    Args:
        img (np.ndarray): raw image needed to detailed, format [H, W, C]
        cpos_ratio (Tuple[float, float]): selected patch's centroid, from 0 to 1
        area_pixels (Tuple[int, int]): pixel area of the patch
        interp_ratio (int, optional): interpolate ratio. Defaults to 3.
        color (Tuple[int, int, int], optional): color of the box. Defaults to (0, 255, 0).
                                                recommend colors:
                                                    (236,229,240)
                                                    (233,138,21)
                                                    (0,59,54)
        thickness (int, optional): thickness of the box. Defaults to 2.
        place(str, optional): where to place the interpolated patch.
    """
    assert (
        0 < cpos_ratio[0] < 1 and 0 < cpos_ratio[1] < 1
    ), "@cpos_ratio can only be range (0, 1)"

    # to array
    if img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img_size = np.array(img.shape[:2])
    cpos_pixels = np.array(cpos_ratio) * img_size
    cpos_ratio = np.array(cpos_ratio)
    area_pixels = np.array(area_pixels)
    paste_pixels = area_pixels * interp_ratio
    img = img.astype("uint8")
    img = np.ascontiguousarray(img)

    # bound check
    bound = []
    for i, j in zip((-1, 1), (-1, 1)):
        bound.append([i * area_pixels[0] / 2, j * area_pixels[1] / 2])
    bound = np.array(bound)
    bound = np.repeat(cpos_pixels[np.newaxis, ...], 2, axis=0) + bound
    assert not np.bitwise_or(
        bound[0] < 0, bound[1] > img_size
    ).any(), f"selected range out of image size, image size {img_size} but get selected range {bound}"

    # find furthest corner to paste the interpolated patch
    if place is None:
        furthest_pos_ratio = None
        furthest_dis = 0.0
        for i in (0, 1):
            for j in (0, 1):
                d = (i - cpos_ratio[0]) ** 2 + (j - cpos_ratio[1]) ** 2
                if d > furthest_dis:
                    furthest_pos_ratio = (i, j)
                    furthest_dis = d
    else:
        assert place in (
            "lt",
            "rt",
            "lb",
            "rb",
        ), "@place should be one of [lt, rt, lb, rb]"
        place_dict = {"lt": (0, 0), "rt": (0, 1), "lb": (1, 0), "rb": (1, 1)}
        furthest_pos_ratio = place_dict[place]

    bound = bound.astype("int")
    patch = img[bound[0, 0] : bound[1, 0], bound[0, 1] : bound[1, 1], :]
    interp_img = cv2.resize(patch, dsize=paste_pixels[::-1])

    box_edge_point = []
    cv2.rectangle(img, bound[0][::-1], bound[1][::-1], color, thickness=thickness)
    box_pre_thick = thickness // 2
    if furthest_pos_ratio == (0, 0):
        img[: paste_pixels[0], : paste_pixels[1], :] = interp_img
        box_edge_point = [[box_pre_thick, box_pre_thick], paste_pixels[::-1]]
    elif furthest_pos_ratio == (1, 0):
        img[-paste_pixels[0] :, : paste_pixels[1], :] = interp_img
        box_edge_point = [
            [box_pre_thick, img_size[0] - paste_pixels[0]],
            [paste_pixels[1], img_size[0] - box_pre_thick],
        ]
    elif furthest_pos_ratio == (0, 1):
        img[: paste_pixels[0], -paste_pixels[1] :, :] = interp_img
        box_edge_point = [
            [img_size[1] - paste_pixels[1], box_pre_thick],
            [img_size[1] - box_pre_thick, paste_pixels[0]],
        ]
    else:
        img[-paste_pixels[0] :, -paste_pixels[1] :, :] = interp_img
        box_edge_point = [
            [img_size[1] - paste_pixels[1], img_size[0] - paste_pixels[0]],
            [img_size[1] - box_pre_thick, img_size[0] - box_pre_thick],
        ]
    cv2.rectangle(img, box_edge_point[0], box_edge_point[1], color, thickness)

    return img


# *==============================================================
# * Image IO utilities
# *==============================================================


ImageType: TypeAlias = (
    Float[Tensor, "B C H W"]
    | Float[Tensor, "C H W"]
    | Float[Tensor, "B H W C"]
    | Float[Tensor, "H W C"]
)
ImagePathType: TypeAlias = str | Path


@beartype
def write_image(
    img: ImageType | list[ImageType],
    img_path: ImagePathType | list[ImagePathType],
    jpeg_quality: int = 95,
    png_compression_lvl: int = 6,
    backend: Literal["tv", "k", "cv2", "tiff", "mat"] = "tv",
    verbose: bool = False,
    not_force_rgb: bool = False,
    mat_file_key: str = "img",
):
    """
    Writes one or more images to disk.

    Args:
        img: The image(s) to write. Can be a single image or a list of images.
            Supported image types are:
                - torch.Tensor with shape (B, C, H, W) or (C, H, W)
                - numpy.ndarray with shape (B, H, W, C) or (H, W, C)
        img_path: The path(s) to write the image(s) to. Can be a single path or a list of paths.
        jpeg_quality: The quality of the JPEG image. Only used when writing JPEG images.
        png_compression_lvl: The compression level of the PNG image. Only used when writing PNG images.
        backend: The backend to use for writing the image.
            - 'tv': TorchVision
            - 'k': Kornia
            - 'cv2': OpenCV
            - 'tiff': PyLibTiff
            - 'mat': SciPy.io
        verbose: Whether to print verbose output.
        not_force_rgb: Whether to avoid forcing the image to be RGB.
        mat_file_key: The key to use when writing the image to a MAT file.
    """

    # Check and convert input
    if not isinstance(img, (Tensor, np.ndarray)):
        raise TypeError(f"img type {type(img)} not supported")

    # Create output directory if needed
    if isinstance(img_path, (str, Path)):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    else:
        for p in img_path:
            os.makedirs(os.path.dirname(p), exist_ok=True)

    # Channel check after converting to correct format
    def check_channels(x):
        if isinstance(x, Tensor):
            c = x.shape[-3] if x.dim() >= 3 else 1
        else:  # numpy array
            x: np.ndarray
            c = x.shape[-1] if x.ndim >= 3 else 1

        if not not_force_rgb and backend not in ("mat", "tiff"):
            assert c in (1, 3), "image should be either grayscale or RGB"

    if isinstance(img, (list, tuple)):
        for i in img:
            check_channels(i)
    else:
        check_channels(img)

    assert 0 < jpeg_quality <= 100, "quality should be range (0, 100]"
    assert 0 <= png_compression_lvl <= 9, "png_compression_lvl should be in [0, 9]"

    def shape_handle(
        img: Tensor | np.ndarray, to_array: bool = False, to_range_255: bool = True
    ):
        if isinstance(img, Tensor):
            if to_range_255:
                img = img.mul(255.0).add_(0.5).clip_(0, 255).to(dtype=torch.uint8)
            img = img.cpu()
            if to_array:
                img = img.numpy()
                img = rearrange(img, "... c h w -> ... h w c")
        elif isinstance(img, np.ndarray):
            if to_range_255:
                img = ((img * 255) + 0.5).clip(0, 255).astype(np.uint8)
            if not to_array:
                img = torch.as_tensor(img, dtype=torch.uint8, device="cpu")
                img = rearrange(img, "... h w c -> ... c h w")
        else:
            raise TypeError(f"img type {type(img)} does not supported")
        return img

    def save_kornia_fn(img: Tensor | np.ndarray, path: str | Path, quality: int = 95):
        assert (
            Path(path).suffix == ".jpg"
        ), "kornia only support jpeg format for its rust backend"
        img = shape_handle(img, to_array=False)
        write_image_k(
            path, img
        )  # kornia rust jpeg backend does not support quality argument

    def save_torchvision_fn(
        img: Tensor, path: str | Path, quality: int = 95, png_compression_lvl: int = 6
    ):
        assert (suffix := Path(path).suffix) in (
            ".jpg",
            ".png",
        ), "torchvision only support jpeg and png format"
        img = shape_handle(img, to_array=False)
        if suffix == ".jpg":
            write_jpeg(img, str(img_path), quality=quality)
        elif suffix == ".png":
            write_png(img, str(img_path), compression_level=png_compression_lvl)
        else:
            raise ValueError(f"unsupported image format {suffix}")

    def save_cv2_fn(
        img: Tensor | np.ndarray,
        path: str | Path,
        quality: int = 95,
        png_compression_lvl: int = 6,
    ):
        suffix = Path(path).suffix

        img = shape_handle(img, to_array=True, to_range_255=not not_force_rgb)
        # RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if suffix == ".jpg":
            saved = cv2.imwrite(
                str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
        elif suffix == ".png":
            saved = cv2.imwrite(
                str(path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_lvl]
            )
        else:  # e.g, tiff file
            saved = cv2.imwrite(str(path), img)

        return saved

    def save_tiff_fn(img: Tensor | np.ndarray, path: str | Path):
        img = shape_handle(img, to_array=True, to_range_255=not not_force_rgb)
        tifffile.imwrite(path, img)

    def save_mat_fn(img: Tensor | np.ndarray, path: str | Path, mat_key: str):
        # mat file do not change the shape
        assert Path(path).suffix == ".mat", "mat file only support mat format"
        img = to_numpy(img)
        savemat(path, {mat_key: img})

    def single_img_save(img, path):
        nonlocal jpeg_quality, png_compression_lvl, verbose

        if backend == "k":
            save_kornia_fn(img, path, quality=jpeg_quality)
        elif backend == "tv":
            save_torchvision_fn(
                img, path, quality=jpeg_quality, png_compression_lvl=png_compression_lvl
            )
        elif backend == "cv2":
            saved = save_cv2_fn(
                img, path, quality=jpeg_quality, png_compression_lvl=png_compression_lvl
            )
            if not saved:
                print(f"[Warn]: opencv backend save {path} failed")
        elif backend == "tiff":
            save_tiff_fn(img, path)
        elif backend == "mat":
            save_mat_fn(img, path, mat_file_key)
        else:
            raise ValueError(f"unsupported backend {backend}")

        if verbose:
            print(f"save image {path}")

    is_batched = img.dim() == 4 if isinstance(img, Tensor) else np.ndim(img) == 4
    if not is_batched:
        single_img_save(img, img_path)
    else:
        assert isinstance(
            img_path, list
        ), "img_path should be a list when img is a batched image"
        assert (
            len(img) == len(img_path)
        ), f"img and img_path should have the same length, but got {len(img)} and {len(img_path)}"
        for img, path in zip(img, img_path):
            single_img_save(img, path)


def plt_plot_img_without_white_margin(img, *args, **kwargs):
    """

    :param img: format [H, W, C]
    :param args: plt.imshow args
    :param kwargs: plt.imshow kwargs
    :return:
    """
    width, height = img.shape[:2]

    ax = plt.imshow(img, *args, **kwargs)

    fig = plt.gcf()
    fig.set_size_inches(width / 100, height / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().set_axis_off()

    return fig, ax


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import PIL.Image as Image

    # img = Image.open('../visualized_img/sr/8.eps')
    # img = np.asarray(img)
    # img = show_details(img, cpos_ratio=(0.2, 0.8), area_pixels=(50, 50), thickness=2)

    # plt_plot_img_without_white_margin(img)
    # plt.show()

    img = torch.randn(1, 3, 256, 256)
    write_image(
        img,
        img_path=["/Data4/cao/ZiHanCao/exps/panformer/visualized_img/test.mat"],
        backend="mat",
        verbose=True,
    )

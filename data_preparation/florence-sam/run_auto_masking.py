import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import spaces
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm

from utils.florence import (
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
    FLORENCE_DETAILED_CAPTION_TASK,
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
    load_florence_model,
    run_florence_inference,
)
from utils.modes import (
    IMAGE_CAPTION,
    IMAGE_CAPTION_GROUNDING_MASKS_MODE,
    IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
)
from utils.sam import load_sam_image_model, load_sam_video_model, run_sam_inference
from utils.video import create_directory, delete_directory, generate_unique_name

MARKDOWN = """
# Florence2 + SAM2 🔥

<div>
    <a href="https://github.com/facebookresearch/segment-anything-2">
        <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block;">
    </a>
    <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block;">
    </a>
    <a href="https://blog.roboflow.com/what-is-segment-anything-2/">
        <img src="https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg" alt="Roboflow" style="display:inline-block;">
    </a>
    <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y">
        <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="display:inline-block;">
    </a>
</div>

This demo integrates Florence2 and SAM2 by creating a two-stage inference pipeline. In
the first stage, Florence2 performs tasks such as object detection, open-vocabulary
object detection, image captioning, or phrase grounding. In the second stage, SAM2
performs object segmentation on the image.
"""

IMAGE_PROCESSING_EXAMPLES = [
    [
        IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
        "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
        "straw, white napkin, black napkin, hair",
    ],
    [
        IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
        "https://media.roboflow.com/notebooks/examples/dog-3.jpeg",
        "tail",
    ],
    [
        IMAGE_CAPTION_GROUNDING_MASKS_MODE,
        "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
        None,
    ],
    [
        IMAGE_CAPTION_GROUNDING_MASKS_MODE,
        "https://media.roboflow.com/notebooks/examples/dog-3.jpeg",
        None,
    ],
]
VIDEO_PROCESSING_EXAMPLES = [
    [
        "videos/clip-07-camera-1.mp4",
        "player in white outfit, player in black outfit, ball, rim",
    ],
    [
        "videos/clip-07-camera-2.mp4",
        "player in white outfit, player in black outfit, ball, rim",
    ],
    [
        "videos/clip-07-camera-3.mp4",
        "player in white outfit, player in black outfit, ball, rim",
    ],
]

VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700", "#32CD32", "#8A2BE2"]
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
# BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
BOX_ANNOTATOR = sv.BoundingBoxAnnotator(
    color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX
)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5,
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX
)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


def on_mode_dropdown_change(text):
    return [
        gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
        gr.Textbox(visible=text == IMAGE_CAPTION_GROUNDING_MASKS_MODE),
    ]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    mode_dropdown, image_input, text_input
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None

    if mode_dropdown == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input:
            gr.Info("Please enter a text prompt.")
            return None, None

        texts = [prompt.strip() for prompt in text_input.split(",")]
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text,
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image_input.size
            )
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        return annotate_image(image_input, detections), None

    if mode_dropdown == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            # task=FLORENCE_CAPTION_TASK
            task=FLORENCE_DETAILED_CAPTION_TASK,
        )
        # caption = result[FLORENCE_CAPTION_TASK]
        caption = result[FLORENCE_DETAILED_CAPTION_TASK]
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
            text=caption,
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        return annotate_image(image_input, detections), caption, detections

    if mode_dropdown == IMAGE_CAPTION:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_DETAILED_CAPTION_TASK,
        )
        caption = result[FLORENCE_DETAILED_CAPTION_TASK]

        return caption


@spaces.GPU(duration=300)
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_video(
    video_input, text_input, progress=gr.Progress(track_tqdm=True)
) -> Optional[str]:
    if not video_input:
        gr.Info("Please upload a video.")
        return None

    if not text_input:
        gr.Info("Please enter a text prompt.")
        return None

    frame_generator = sv.get_video_frames_generator(video_input)
    frame = next(frame_generator)
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    texts = [prompt.strip() for prompt in text_input.split(",")]
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=frame,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text,
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=frame.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)

    if len(detections.mask) == 0:
        gr.Info(
            "No objects of class {text_input} found in the first frame of the video. "
            "Trim the video to make the object appear in the first frame or try a "
            "different text prompt."
        )
        return None

    name = generate_unique_name()
    frame_directory_path = os.path.join(VIDEO_TARGET_DIRECTORY, name)
    frames_sink = sv.ImageSink(
        target_dir_path=frame_directory_path, image_name_pattern="{:05d}.jpeg"
    )

    video_info = sv.VideoInfo.from_video_path(video_input)
    video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
    video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

    frames_generator = sv.get_video_frames_generator(video_input)
    with frames_sink:
        for frame in tqdm(
            frames_generator,
            total=video_info.total_frames,
            desc="splitting video into frames",
        ):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            frames_sink.save_image(frame)

    inference_state = SAM_VIDEO_MODEL.init_state(
        video_path=frame_directory_path, device=DEVICE
    )

    for mask_index, mask in enumerate(detections.mask):
        _, object_ids, mask_logits = SAM_VIDEO_MODEL.add_new_mask(
            inference_state=inference_state, frame_idx=0, obj_id=mask_index, mask=mask
        )

    video_path = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")
    frames_generator = sv.get_video_frames_generator(video_input)
    masks_generator = SAM_VIDEO_MODEL.propagate_in_video(inference_state)
    with sv.VideoSink(video_path, video_info=video_info) as sink:
        for frame, (_, tracker_ids, mask_logits) in zip(
            frames_generator, masks_generator
        ):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
            if len(masks.shape) == 4:
                masks = np.squeeze(masks, axis=1)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=np.array(tracker_ids),
            )
            annotated_frame = frame.copy()
            annotated_frame = MASK_ANNOTATOR.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = BOX_ANNOTATOR.annotate(
                scene=annotated_frame, detections=detections
            )
            sink.write_frame(annotated_frame)

    delete_directory(frame_directory_path)
    return video_path


def main_auto_masking():
    import argparse

    argparser = argparse.ArgumentParser(
        description="Run auto masking on a directory of images."
    )
    argparser.add_argument("--img_dir", type=str, required=True)
    argparser.add_argument("--dataset", type=str, default="MFF-MFFW", required=False)
    argparser.add_argument(
        "--saved_path", type=str, default="results/masks", required=False
    )
    args = argparser.parse_args()

    img_dir = args.img_dir
    dataset = args.dataset
    print("dataset:", dataset)

    caption_df = pd.DataFrame()

    # mask path
    saved_path = Path(args.saved_path)
    mask_path = saved_path / dataset / "mask"
    mask_path.mkdir(parents=True, exist_ok=True)

    # annotation path
    annotation_path = saved_path / dataset / "annotated"
    annotation_path.mkdir(parents=True, exist_ok=True)

    # json path
    json_path = saved_path / dataset / f"{dataset}.json"
    f = open(json_path, "w+")

    detection_files = {}

    ## MFF / MEF
    img_lst = list(img_dir.glob("*"))
    img_lst.sort()
    print(f"found {len(img_lst)} images")
    for index, img_path in enumerate(img_lst, 1):
        img = Image.open(img_path.as_posix())
        img = img.convert("RGB")
        annotated_img, caption, detections = process_image(
            mode_dropdown=IMAGE_CAPTION_GROUNDING_MASKS_MODE,
            image_input=img,
            text_input="",
        )
        mask = detections.mask  # [n_detections, h, w]
        class_ = detections.data["class_name"]  # [n_detections]

        img_name = img_path.stem
        detection_files[img_name] = {
            "size": str(img.size),
            "caption": caption,
            "xyxy": detections.xyxy.tolist(),
            "class_name": class_.tolist(),
        }
        # save annotated image
        annotated_img.save(annotation_path / f"{img_name}.jpg")

        # save mask
        # bg to 0, objects to index
        new_mask = np.zeros(mask.shape[-2:])
        for i, mask_i in enumerate(mask, 1):
            new_mask[mask_i] = i
        new_mask = new_mask.astype(np.uint8)
        Image.fromarray(new_mask).save(mask_path / f"{img_name}.png")

        # save json
        json.dump(detection_files, f)

        print(f"{index}/{len(img_lst)} - save masks for {img_path.name}")

    f.close()


if __name__ == "__main__":
    # ! run on transformers==4.45.1
    main_auto_masking()

import argparse
import os

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

CACHE_DIR = os.path.join(os.path.dirname(__file__), "pretrained_ckpts")


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length, cache_dir=CACHE_DIR
            )
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                version, max_length=max_length, cache_dir=CACHE_DIR
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def load_t5(
    device: str | torch.device = "cuda",
    max_length: int = 512,
    model_name: str = "google-t5/t5-small",
) -> HFEmbedder:
    # "google/t5-v1_1-xxl"

    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(model_name, max_length=max_length, torch_dtype=torch.bfloat16).to(
        device
    )


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(
        "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
    ).to(device)


def prepare_txt_with_h5(
    t5: HFEmbedder,
    prompt: str | list[str],
    bs: int = 1,
    device: str | torch.device = "cuda",
):
    if isinstance(prompt, str):
        prompt = [prompt]
    # t5 text embedding
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    return txt.to(device), txt_ids.to(device)


def prepare_txt_with_clip(
    clip: HFEmbedder,
    prompt: str | list[str],
    bs: int = 1,
    device: str | torch.device = "cuda",
):
    if isinstance(prompt, str):
        prompt = [prompt]
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return vec.to(device)


def prepare_img(img: Tensor, device: str | torch.device = "cuda") -> Tensor:
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    # (h // 2, w // 2) + (h // 2, 1)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    # (h // 2, w // 2) + (1, w // 2)
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return img, img_ids.to(device)


def prepare(
    t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]
) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    device = img.device
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img, img_ids = prepare_img(img, device=device)
    txt, txt_ids = prepare_txt_with_h5(t5, prompt, bs=bs, device=device)
    vec = prepare_txt_with_clip(clip, prompt, bs=bs, device=device)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def measure_t5_params_and_flops(t5: HFEmbedder):
    import fvcore.nn as fnn

    t5.to("cuda:0").eval()

    with torch.no_grad():
        print(fnn.flop_count_table(fnn.FlopCountAnalysis(t5, ["1" * 512])))


def main_encode_csv_captions_t5_embeddings() -> None:
    parser = argparse.ArgumentParser(description="Encode CSV captions to T5 embeddings")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing captions",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save the T5 features"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=1, help="GPU ID to use (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (default: cuda)",
    )

    args = parser.parse_args()

    from pathlib import Path

    import numpy as np
    import pandas as pd
    from safetensors.numpy import save_file

    torch.cuda.set_device(args.gpu_id)

    t5 = load_t5()
    measure_t5_params_and_flops(t5)

    csv_path = Path(args.csv_path)
    name = csv_path.stem
    name = name.replace("caption", "t5_feature")
    save_dir = Path(args.save_dir)
    csv_file = pd.read_csv(csv_path, dtype=str)

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{name}.safetensors"
    print(f"save features to file: {save_path}")

    file_name = csv_file.iloc[:, 0]
    caption = csv_file["caption"]
    h5_features = {}
    for i, (f_name, prompt) in enumerate(zip(file_name, caption), 1):
        txt, txt_ids = prepare_txt_with_h5(t5, prompt, bs=1, device=args.device)
        f_name = str(f_name)
        txt = txt.to(torch.float16).cpu().numpy()
        h5_features[f_name] = txt
        print(f"[{i}/{len(file_name)}] - save {f_name} to dict")

    save_file(h5_features, save_path)
    print(f"save features to file: {save_path}")


if __name__ == "__main__":
    main_encode_csv_captions_t5_embeddings()

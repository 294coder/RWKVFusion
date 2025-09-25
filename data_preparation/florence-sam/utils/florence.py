import os
from typing import Any, Dict, Tuple, Union
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

FLORENCE_CHECKPOINT = "microsoft/Florence-2-large"
FLORENCE_OBJECT_DETECTION_TASK = "<OD>"
FLORENCE_DETAILED_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"
FLORENCE_DENSE_REGION_CAPTION_TASK = "<DENSE_REGION_CAPTION>"
FLORENCE_CAPTION_TASK = "<CAPTION>"


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_florence_model(
    device: torch.device, checkpoint: str = FLORENCE_CHECKPOINT
) -> Tuple[Any, Any]:
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint, trust_remote_code=True, cache_dir="florence_ckpt/"
            )
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, cache_dir="florence_ckpt"
        )
        return model, processor


def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image,
    task: str,
    text: str = "",
) -> Tuple[str, Dict]:
    # if task == FLORENCE_DETAILED_CAPTION_TASK and text != "":
    #     processor.task_prompts_without_inputs[FLORENCE_DETAILED_CAPTION_TASK] = \
    #         text + ', '+ 'describe with a paragraph what is shown in the image.'
    #     prompt = task
    # else:
    prompt = task + text

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text, task=task, image_size=image.size
    )
    return generated_text, response


# * florence model forward code
from typing import List, Optional, Tuple, Union


def florence_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[List[torch.FloatTensor]] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    Returns:
    Example:
    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Florence2ForConditionalGeneration
    >>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
    >>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
    >>> prompt = "<CAPTION>"
    >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=100)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "A green car parked in front of a yellow building."
    ```"""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    image_features = None
    if inputs_embeds is None:
        # 1. Extra the input embeddings
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        if pixel_values is not None:
            # (batch_size, num_image_tokens, hidden_size)
            image_features = self._encode_image(pixel_values)
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds
            )

    if inputs_embeds is not None:
        attention_mask = attention_mask.to(inputs_embeds.dtype)
    outputs = self.language_model(
        attention_mask=attention_mask,
        labels=labels,
        inputs_embeds=inputs_embeds,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask,
        decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs.logits
    logits = logits.float()
    loss = outputs.loss
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # return Florence2Seq2SeqLMOutput(
    #     loss=loss,
    #     logits=logits,
    #     past_key_values=outputs.past_key_values,
    #     decoder_hidden_states=outputs.decoder_hidden_states,
    #     decoder_attentions=outputs.decoder_attentions,
    #     cross_attentions=outputs.cross_attentions,
    #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,
    #     encoder_hidden_states=outputs.encoder_hidden_states,
    #     encoder_attentions=outputs.encoder_attentions,
    #     image_hidden_states=image_features,
    # )


def florence_generate(self, input_ids, inputs_embeds=None, pixel_values=None, **kwargs):
    if inputs_embeds is None:
        # 1. Extra the input embeddings
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        if pixel_values is not None:
            image_features = self._encode_image(pixel_values)
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds
            )

    return self.language_model.generate(
        input_ids=None, inputs_embeds=inputs_embeds, **kwargs
    )

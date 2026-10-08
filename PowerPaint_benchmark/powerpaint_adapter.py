import importlib.util
import sys

import torch
from diffusers import UniPCMultistepScheduler
from safetensors.torch import load_file, load_model

from common import ROOT

VENDOR = ROOT / "vendor" / "PowerPaint"
VENDOR_COMMIT = "5b4c3d52291709fcec2a1870d987da693fd3549c"
QUALITY_NEGATIVE = ", worst quality, low quality, normal quality, bad quality, blurry "


def load_vendor(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, VENDOR / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vendor_modules():
    pipeline = load_vendor("powerpaint_v1_pipeline", "powerpaint/pipelines/pipeline_PowerPaint.py")
    utils = load_vendor("powerpaint_v1_utils", "powerpaint/utils/utils.py")
    return pipeline, utils


def prompts_for(prompt, negative=""):
    if any(token in prompt or token in negative for token in ("P_obj", "P_ctxt", "P_shape")):
        raise ValueError("Dataset prompts must not contain reserved PowerPaint task tokens")
    return prompt + " P_obj", negative + QUALITY_NEGATIVE + "P_obj"


def check_prompt(tokenizer, text):
    length = len(tokenizer(text, truncation=False)["input_ids"])
    if length > tokenizer.model_max_length:
        raise ValueError(f"Expanded task prompt has {length} tokens, exceeding {tokenizer.model_max_length}; refusing silent task-token truncation")
    return length


def load_text_weights(model, filename):
    state = load_file(str(filename), device="cpu")
    position_key = "text_model.embeddings.position_ids"
    if position_key in state and position_key not in model.state_dict():
        expected = model.text_model.embeddings.position_ids.cpu()
        if state[position_key].shape != expected.shape or not torch.equal(state[position_key], expected):
            raise ValueError("Legacy CLIP position_ids differs from the current fixed position buffer")
        del state[position_key]
    model.load_state_dict(state, strict=True)


def load_pipeline(scheduler="unipc"):
    pipeline, utils = vendor_modules()
    base = ROOT / "models" / "base"
    pipe = pipeline.StableDiffusionInpaintPipeline.from_pretrained(
        str(base), variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True, safety_checker=None,
        requires_safety_checker=False, low_cpu_mem_usage=False,
    )
    pipe.tokenizer = utils.TokenizerWrapper(
        from_pretrained=str(ROOT / "models" / "tokenizer"), subfolder="tokenizer", local_files_only=True,
    )
    utils.add_tokens(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                     placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                     initialize_tokens=["a", "a", "a"], num_vectors_per_token=10)
    checkpoint = ROOT / "models" / "powerpaint"
    load_model(pipe.unet, str(checkpoint / "unet" / "unet.safetensors"), strict=True)
    load_text_weights(pipe.text_encoder, checkpoint / "text_encoder" / "text_encoder.safetensors")
    if pipe.unet.config.in_channels != 9:
        raise ValueError("PowerPaint v1 requires a nine-channel inpainting UNet")
    if scheduler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler != "base":
        raise ValueError(f"Unsupported scheduler: {scheduler}")
    pipe = pipe.to(device="cuda", dtype=torch.float16)
    for name in ("unet", "vae", "text_encoder"):
        getattr(pipe, name).eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe

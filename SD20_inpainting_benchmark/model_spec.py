ORIGINAL_REPOSITORY = "stabilityai/stable-diffusion-2-inpainting"
BASE_REPOSITORY = "sd2-community/stable-diffusion-2-inpainting"
BASE_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
CROSSCHECK_REPOSITORY = "alwold/stable-diffusion-2-inpainting"
CROSSCHECK_REVISION = "e8255bd00e6a3d8dd489d2a9f03955eb08262d6e"
CLIP_REPOSITORY = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
WEIGHTS_SHA256 = {
    "text_encoder/model.fp16.safetensors": "681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668",
    "unet/diffusion_pytorch_model.fp16.safetensors": "29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8",
    "vae/diffusion_pytorch_model.fp16.safetensors": "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
}
MODEL_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "text_encoder/config.json", "unet/config.json", "vae/config.json",
    "feature_extractor/preprocessor_config.json", *WEIGHTS_SHA256,
)


def validate_architecture(unet, text, scheduler, index):
    if unet.get("in_channels") != 9 or unet.get("out_channels") != 4:
        raise ValueError("Expected nine-channel SD2.0 inpainting, not a text-to-image base")
    if unet.get("cross_attention_dim") != 1024 or text.get("hidden_size") != 1024:
        raise ValueError("Expected SD2 OpenCLIP dimensions, not SD1.5")
    if not unet.get("use_linear_projection") or text.get("num_hidden_layers") != 23:
        raise ValueError("Unexpected SD2.0 architecture")
    if scheduler.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("Original SD2.0 inpainting predicts epsilon, not v_prediction")
    if index.get("_class_name") != "StableDiffusionInpaintPipeline":
        raise ValueError("Custom pipelines are not allowed")

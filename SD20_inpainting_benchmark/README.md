# 原版 SD2.0 Inpainting 数据集基准

本项目使用原 `stabilityai/stable-diffusion-2-inpainting` 的公开存档，不使用 Text2Earth-inpainting、SD2 文生图基座、PowerPaint、BrushNet 权重或任何自定义 pipeline。

## 权重身份与可追溯性

原仓库在准备时返回 HTTP 401。采用声明为其镜像的 `sd2-community/stable-diffusion-2-inpainting`，固定 revision `5f74973cbb64c8568780732c17f43eb269d63a0d`。社区镜像与 Stability AI 无隶属关系。三份 FP16 safetensors 的 LFS SHA256 已与另一公开存档 `alwold/stable-diffusion-2-inpainting` 交叉核对；无法直接从当前不可访问的原仓库再验证，故明确保留这一来源限制。

模型是从 SD2-base 继续训练的原版 inpainting 发布版：UNet 输入9通道、输出4通道、cross-attention 1024维；OpenCLIP文本编码器1024维、23层、linear projection。**预测类型是 epsilon，不是 v_prediction。** 原始默认采样器为 PNDM；为对齐已有实验，本基准使用 UniPC。权重规范见 `model_spec.py`，下载后每个文件的哈希、固定版本及镜像来源见 `models/sources.json`。模型遵循 CreativeML Open RAIL++-M，具体条款以原始发布许可证为准。

## 环境及资产准备

在此目录执行：

```bash
bash setup_environment.sh
.venv/bin/python prepare_assets.py
.venv/bin/python prepare_dataset.py
.venv/bin/python -m unittest -v test_benchmark test_sd20
```

Python 3.9 独立 `.venv`，不继承系统 site-packages。使用与 SD1.5 基准相同的 CUDA torch 2.1.2+cu121、官方 diffusers 0.27.0 及锁定依赖。`requirements.lock.txt` 为安装约束，`requirements.installed.txt` 为实际安装清单，`environment.json` 记录环境来源。

`prepare_assets.py` 仅从公开 Hugging Face 固定版本下载白名单文件，并校验权重SHA256/配置Git blob；不加载 pickle、远程代码或自定义pipeline。凭据如有需要只取环境变量 `HF_TOKEN`，默认不读取本机登录token。CLIP ViT-L/14和AlexNet从相邻 `SD15_inpainting_benchmark` 复制并校验；所有资产准备好后，推理及默认评测均离线且不依赖相邻项目。下载或安装期间不修改旧项目。

## 运行

```bash
# 四图端到端验证，报告明确标为 partial
.venv/bin/python run_benchmark.py --run-dir runs/smoke_s42 --limit 4

# 全量推理成功后自动评测；同一命令支持断点续跑
.venv/bin/python run_benchmark.py

# 单独推理或重评
.venv/bin/python infer.py
.venv/bin/python evaluate.py
```

默认正式运行目录：`runs/sd20_inpainting_unipc_s42/`。

- `generated/`：256×256 RGB PNG。
- `samples/`：逐图seed、耗时、配置和输出哈希。
- `config.json`、`scheduler_config.json`：完整实验及实际采样器配置。
- `progress.json`、`generation_summary.json`：推理进度和完整性。
- `job_status.json`：驱动/子进程PID、阶段及完成/失败状态。
- `logs/inference.log`、`logs/evaluation.log`：运行日志。
- `eval_results.json`、`eval_results.txt`、`eval_results_per_sample.json`：汇总及逐图指标。

`--limit` 是排序后前N条，只用于小批测试，不改变全量1550条定义。完整评测拒绝缺图；部分评测必须显式指定 `--allow-partial`。改变配置、推理脚本或模型规范后须使用新的 `--run-dir`，避免混合结果。逐图输出验证后才可跳过重算。运行目录使用文件锁，单GPU显存预算默认40%，不足时等待，不终止其他作业。`infer.py --scheduler base --run-dir ...` 可另测原模型的PNDM，不能和默认UniPC实验混用。

## 与现有基准一致的协议

数据 `../datasets/test_dataset_1500` 实际1550对，以 `prompts.json` 及真实文件配对为准。原图224×224，mask256×256；灰度>127的白色区域为编辑区，黑色为保留区。`dataset_info.json` 的7处文件名差异保留在数据清单中，不修改原始数据。

推理：512×512，图像BILINEAR、mask NEAREST；原始RGB直接传给官方 `StableDiffusionInpaintPipeline`，由pipeline归一化后遮罩。50步、CFG7.5、strength1.0、fp16、每张重置seed42、原样提示词和空负面提示。输出仅BILINEAR缩至256×256，不回贴、不混合、不羽化。为对齐本地遥感基准不加载safety checker，不应直接用于公众生成服务。

评测公式沿用SD1.5/BrushNet：Global/Local CLIP（ViT-L/14，fp32归一化余弦×100；Local掩码外置黑），AlexNet LPIPS v0.1（全图及区域空间图），背景MAE/PSNR/SSIM、3像素双侧边界误差，FWD（haar、level4、log_scale=False、resize256）。FID仅在 `evaluate.py --fid` 时额外计算，不属于默认协议，其第三方后端可能需要下载公开Inception权重。

**参照图是编辑前原图，而非编辑后的目标真值。LPIPS/FWD及背景/边界误差不能独立说明语义编辑成功。小批指标只验证流程，不参与全量方法排名。**

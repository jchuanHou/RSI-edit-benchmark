# 原版 SD1.5 Inpainting 基准

此项目只使用 `stable-diffusion-v1-5/stable-diffusion-inpainting`（原 `runwayml/stable-diffusion-inpainting` 的公开镜像），不使用 Text2Earth-inpainting、PowerPaint 增量权重、BrushNet 或自定义 pipeline。

## 环境与资产

在此目录执行：

```bash
bash setup_environment.sh
.venv/bin/python prepare_assets.py
.venv/bin/python prepare_dataset.py
.venv/bin/python -m unittest -v test_benchmark test_sd15
```

Python 3.9，独立 `.venv`，不继承系统 site-packages。固定依赖见 `requirements.txt`，完整安装版本见 `requirements.lock.txt`，实际来源与 GPU 见 `environment.json`。CUDA torch 2.1.2+cu121、官方 diffusers 0.27.0。安装脚本只安装依赖；`prepare_assets.py` 从相邻 `PowerPaint_benchmark` 复制并逐文件校验原始基座、CLIP 和 AlexNet，完成后推理及默认评测不再依赖相邻项目或网络。复制来源和哈希保存在 `models/sources.json`。模型权重及许可归各原始发布者。

## 运行

```bash
# 小批端到端验证，报告会明确标记 partial
.venv/bin/python run_benchmark.py --run-dir runs/smoke_s42 --limit 4

# 全量推理成功后自动评测；再次执行相同命令可以断点续跑
.venv/bin/python run_benchmark.py

# 单独推理 / 重新评测
.venv/bin/python infer.py
.venv/bin/python evaluate.py
```

默认正式输出：`runs/sd15_inpainting_unipc_s42/`。

- `generated/`：256×256 RGB PNG；`samples/`：逐图耗时、seed、配置/输出哈希。
- `config.json`、`scheduler_config.json`：完整可追溯配置。
- `progress.json`、`generation_summary.json`：进度与生成完整性。
- `job_status.json`：驱动进程/子进程 PID、阶段、成功/失败状态。
- `logs/inference.log`、`logs/evaluation.log`：推理及评测日志。
- `eval_results.json`、`eval_results.txt`、`eval_results_per_sample.json`：汇总及逐图指标。

`--limit` 只处理排序后的前 N 条，不重新定义数据集。完整评测默认拒绝缺失图像，部分测试必须显式 `--allow-partial`；只接受通过哈希、配置和尺寸验证的输出。改变推理配置或推理源文件后必须用新的 `--run-dir`，避免混合结果。运行采用文件锁；不要同时往同一目录写入。单 GPU 显存预算默认 40%，资源不足等待，不中断其他进程。

## 对齐协议

数据：`../datasets/test_dataset_1500`，实际 **1550** 对，`prompts.json` 为配对依据。原图 224×224，mask 256×256；mask 灰度 >127 为编辑区，黑色为保留区。原始 `dataset_info.json` 有 7 处文件名差异，保留原文件，差异列在 `dataset_manifest.json`。

默认推理：工作尺寸 512×512，图像 BILINEAR、mask NEAREST；官方九通道 pipeline 接收原始 RGB 并在内部归一化后遮罩；UniPC、50 步、CFG=7.5、strength=1.0、fp16，每张重置 seed=42。提示词原样使用、空负面提示；不增加 learned token、LoRA 或 adapter。输出只 BILINEAR 缩至 256×256，不额外混合/羽化/回贴。UniPC 是为对齐现有基准主动选择的，`infer.py --scheduler base --run-dir ...` 可另测底模默认 DDIM。

为与现有本地遥感基准一致，未加载 safety checker；不可将此配置直接作为公众生成服务。

评测计算沿用 `BrushNet_benchmark/evaluate.py`，仅调整模型路径、离线权重校验和运行完整性检查：

- Global / Local CLIP：OpenAI ViT-L/14，fp32 归一化余弦 ×100；Local 为掩码外置黑，不是框裁剪。
- LPIPS：AlexNet v0.1，全图标量及掩码内/外、边界空间图均值。
- 背景 MAE / PSNR / SSIM；边界为 3 像素双侧带。
- FWD：pytorchfwd，haar、level=4、log_scale=False、resize=256。
- FID 只在 `evaluate.py --fid` 时额外计算，不属于当前默认对齐协议；其第三方后端可能需要公开 Inception 权重下载。

**reference_images 是编辑前原图，不是编辑后真值。** LPIPS、FWD、背景/边界误差衡量与原图的差异，不能独立说明语义编辑是否成功。小批结果只用于验证流程，不用于与全量 BrushNet/PowerPaint 结果排名。

公共数据读取、数据验证、基础测试及评测公式参考现有 `BrushNet_benchmark`；GPU 保护和串行调度参考 `PowerPaint_benchmark`。旧项目代码、环境和输出均不修改。

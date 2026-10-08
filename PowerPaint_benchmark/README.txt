PowerPaint v1 数据集基准
=======================

项目根目录：/data/workspace/PowerPaint_benchmark
环境：.venv/bin/python（Python 3.9；继承宿主基础 PyTorch/CUDA，只在本项目安装覆盖依赖；不修改 BrushNet 环境）
主要脚本：setup_environment.sh、download_models.py、prepare_dataset.py、infer.py、evaluate.py、run_benchmark.py

模型选择
--------
使用官方原版 PowerPaint-v1，不使用基于 BrushNet 的 v2，避免与 BrushNet 对比混淆。
官方代码：open-mmlab/PowerPaint，固定 revision 见 powerpaint_adapter.py。
官方权重：JunhaoZhuang/PowerPaint-v1；底模为 SD1.5 inpainting 的公开镜像，不是普通 SD1.5。
所有模型来源、revision、文件 SHA256 见 models/sources.json。模型加载使用本地 safetensors，不启动 UI 或分享服务。
旧版文本编码器的 position_ids 与 Transformers 4.37.2 固定缓冲区不一致时拒绝加载；只有值完全相同的历史冗余字段可移除，所有学习权重仍严格校验。

数据与对齐协议
--------------
数据：../datasets/test_dataset_1500；实际 1550 对，以 prompts.json 和实际文件为准。
原图 224x224、mask 256x256；白色 >127 表示编辑。
工作分辨率 512x512，输出 256x256；图像 BILINEAR，mask NEAREST。
50 步，CFG 7.5，每张重置 seed=42，float16，batch size=1，不做原图回填或羽化。
默认 UniPC 与当前 BrushNet 对齐；这不是官方 app 原生设置（app 短边 640、默认 45 步、沿用底模 scheduler）。
infer.py --scheduler base 可单独测试底模 scheduler，但必须使用独立 --run-dir。
正提示为数据集 prompt + P_obj，负提示保留官方质量词和 P_obj。A/B 提示相同。
P_obj 等任务占位符各展开为 10 个学习向量；所有 1550 条最终提示在采样前检查长度，不允许静默截断任务 token。
PowerPaint 传入原始 RGB，由官方 pipeline 在归一化后遮挡，不套用 BrushNet 的 RGB 编辑区置黑操作。
本地遥感基准关闭 safety checker，与现有 BrushNet 一致；这不是面向用户开放的生成服务。

评测与解释
----------
evaluate.py 从已有 BrushNet 基准复制计算核心，只修改默认目录并增加显存准入。
Global/Local CLIP：OpenAI ViT-L/14 fp32，归一化余弦 x100；Local 是 mask 外置黑后的整图，不是 bbox 裁剪。
LPIPS：AlexNet v0.1，全图和 spatial 区域平均。
背景 MAE/PSNR/SSIM；双侧 3 像素边界带 MAE/LPIPS。
FWD：pytorchfwd 1.0.1，haar、max_level=4、log_scale=False、resize=256；默认启用。
FID 是可选 --fid，不作为默认指标。
reference 是编辑前原图，不是编辑后真值；LPIPS/FWD/FID 不能单独证明编辑成功。
小样本评测只能验证流程，分布指标不能作为全数据集结论。

常用命令（在项目目录执行）
--------------------------
bash setup_environment.sh
.venv/bin/python download_models.py
.venv/bin/python prepare_dataset.py
.venv/bin/python -m unittest -v test_benchmark
.venv/bin/python run_benchmark.py --run-dir runs/smoke_new --limit 3
.venv/bin/python run_benchmark.py
.venv/bin/python infer.py --run-dir runs/powerpaint_v1_unipc_s42
.venv/bin/python evaluate.py --run-dir runs/powerpaint_v1_unipc_s42

run_benchmark.py 串行执行推理和评测；后台启动可用 nohup，并重定向到 logs/。
相同配置可断点续跑，仅跳过成功且输出哈希一致的样本；代码/模型/配置变化必须用新目录。
--limit 取排序前 N 条，不是随机抽样；只完成小样本时报告 partial/smoke_complete，不冒充完整基准。
默认评测拒绝缺图；手动评测部分结果需显式 --allow-partial。

GPU 共存
--------
当前仅支持单 GPU 主机。启动前需要空闲显存 >= 自身预算 + 2048 MiB；默认自身 PyTorch 分配上限为总显存 40%。
16GB T4 对应约 6144 MiB 分配预算、8192 MiB 启动准入。每张前还检查至少 1536 MiB 空闲。
评测同样按 8192 MiB 准入及 40% 分配预算执行。不足时等待；推理首个错误立即停止释放自身资源。
这不是 GPU 硬件隔离：CUDA 上下文/第三方内存和其他进程未来增长不受此预算约束。
不会终止、暂停或修改 BrushNet 任务。GPU 算力共享会导致双方变慢，不保证整体更快。

输出和排查
----------
runs/<名称>/generated/：逐图 PNG
runs/<名称>/samples/：耗时、输出 SHA256、已分配/保留显存峰值、成功/失败信息
runs/<名称>/config.json：数据/模型/代码指纹与完整参数
runs/<名称>/scheduler_config.json：实际 scheduler 类和配置
runs/<名称>/prompt_lengths.json：全量正负提示长度
runs/<名称>/job_status.json、progress.json、generation_summary.json：任务状态和进度
runs/<名称>/logs/inference.log、evaluation.log：分阶段日志
runs/<名称>/eval_results.json、eval_results_per_sample.json、eval_results.txt：汇总及逐图指标
logs/unit-tests.log：协议测试；requirements.lock.txt：当前实际环境版本。
首次 smoke_v1_s42 留存已修复的 position_ids 兼容性失败记录，不是有效实验结果。

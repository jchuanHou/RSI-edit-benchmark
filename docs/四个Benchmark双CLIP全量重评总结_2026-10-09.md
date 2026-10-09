# 四个 Benchmark 双 CLIP 全量重评总结

评测日期：2026-10-09

状态：**8/8 组完成，独立验收通过**

完成时间：北京时间 2026-10-09 10:35:04；独立验收完成于 10:35:31。

## 1. 结论摘要

本次对 BrushNet、PowerPaint、SD1.5 Inpainting、SD2.0 Inpainting 的已有生成结果，分别使用 **OpenAI CLIP ViT-B/32** 和 **OpenAI CLIP ViT-L/14**，重新计算 Global CLIP Score 与 Local CLIP Score。

- 每个 benchmark、每个 CLIP 评测器均完整评估 **1550/1550** 个样本，无缺失、无非有限分数。
- 共完成 **4 个 benchmark × 2 个 CLIP = 8 组评测**，得到 12400 条“图像—评测器”记录、24800 个评分标量。
- **PowerPaint 在两个 CLIP 评测器上的 Global 均值均最高。**
- **BrushNet 在两个 CLIP 评测器上的 Local 均值均最高。**
- 两个评测器的 **Local 排名完全一致**；Global 的中间名次存在变化，不能认为所有方法的排序与评测器无关。
- 四个 benchmark 的 L/14 重算结果均与各自旧结果**逐样本完全一致**，Global、Local 的最大绝对差均为 0。
- 本次只重算 CLIP，不重新生成图像，不重算 LPIPS/FWD 等其他指标，也没有覆盖原评测文件。

**适用范围：**上述“最高”仅表示当前数据集、当前生成配置和单一生成 seed 下的算术均值最高。本次未进行显著性检验，不能据此宣称统计显著优势或整体编辑质量最优。

## 2. 主结果表

以下所有分数均为 **归一化图像／文本特征余弦相似度 × 100**，同一列内越高越好。每格是 **均值 ± 样本间总体标准差**，有效样本数均为 1550；粗体表示该列最高均值。

| Benchmark | B/32 Global ↑ | B/32 Local ↑ | L/14 Global ↑ | L/14 Local ↑ |
|---|---:|---:|---:|---:|
| BrushNet | 27.251125 ± 3.130783 | **22.337799 ± 3.564385** | 21.974328 ± 3.679096 | **18.765506 ± 3.482079** |
| PowerPaint | **27.513319 ± 3.118457** | 22.172065 ± 3.550650 | **22.082659 ± 3.609764** | 18.651872 ± 3.721306 |
| SD1.5 Inpainting | 27.143789 ± 3.164582 | 22.056527 ± 3.517011 | 21.767960 ± 3.702114 | 18.499143 ± 3.657636 |
| SD2.0 Inpainting | 27.255389 ± 3.259677 | 22.131987 ± 3.536901 | 21.676450 ± 3.757849 | 18.548023 ± 3.591874 |

注意：标准差采用 `ddof=0`，描述不同样本之间的分数离散程度，**不是标准误、置信区间或多次运行波动**。完整精度保留在机器可读结果中。

### 2.1 名次对照

以下为未经显著性检验的均值排序。

| 评测器／指标 | 第 1 名 | 第 2 名 | 第 3 名 | 第 4 名 |
|---|---|---|---|---|
| B/32 Global | PowerPaint | SD2.0 | BrushNet | SD1.5 |
| L/14 Global | PowerPaint | BrushNet | SD1.5 | SD2.0 |
| B/32 Local | BrushNet | PowerPaint | SD2.0 | SD1.5 |
| L/14 Local | BrushNet | PowerPaint | SD2.0 | SD1.5 |

### 2.2 BrushNet 与 PowerPaint 的差异

下表为 **PowerPaint − BrushNet**，单位为本协议下的 CLIP 分数点。

| 评测器 | Global 差值 | Local 差值 |
|---|---:|---:|
| B/32 | +0.262195 | −0.165734 |
| L/14 | +0.108330 | −0.113634 |

两种评测器均呈现相同的均值方向：PowerPaint 的全图文本匹配得分较高，BrushNet 的编辑掩码区域得分较高。但这两项指标并不等价于背景保真、边界自然度或编辑成功率，不能仅凭这张表认定某个方法全面胜出。

### 2.3 SD1.5 与 SD2.0，以及评测器敏感性

- B/32 的 Global 中，SD2.0 高于 SD1.5；L/14 的 Global 中则相反。因此，对这两者的全图语义匹配排序依赖 CLIP 评测器。
- Local 中，SD2.0 在两个评测器上均高于 SD1.5，但仍低于 BrushNet 和 PowerPaint。
- B/32 Global 中，SD2.0 与 BrushNet 的均值仅相差 **0.004264** 分，虽然形式上存在名次先后，但不宜将其解读为明确的实用优势。
- B/32 的绝对分数总体高于 L/14，**不表示 B/32 对应的生成质量更高**：图像完全相同，变化的是评测编码器。跨评测器的分数标尺不可直接混合比较，也不建议将四列直接求平均作为“综合分”。

## 3. 本次到底重评了什么

### 3.1 两个 CLIP 评测器

| 简称 | 模型 ID | 固定 revision | 本地评测目录 |
|---|---|---|---|
| B/32 | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | `SD15_inpainting_benchmark/models/clip_b32/` |
| L/14 | `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` | `SD15_inpainting_benchmark/models/clip/` |

路径均相对于 `/data/workspace/`。

四个 benchmark 原有正式全量评测使用的都是 L/14。本次新增 B/32 的全量结果，并将 L/14 也真正重新计算一遍，**不是把旧 L/14 汇总数复制进新表**。每个评测器都同时计算 Global 和 Local，并非让 Global、Local 各自使用不同模型。

B/32 官方固定版本提供的是 `pytorch_model.bin`。准备阶段校验官方权重 SHA256，使用 `torch.load(weights_only=True, map_location="cpu")` 受限读取纯张量字典，再将不变的张量转换成 `model.safetensors`。正式评测只从本地读取 safetensors，不启用远程代码。原始摘要、转换说明及各模型文件摘要保存在来源记录中。

两个 CLIP 均加载成功，没有缺失参数、额外参数、形状不匹配或加载错误。

### 3.2 四套固定生成结果

| Benchmark | 原始运行目录 |
|---|---|
| BrushNet | `BrushNet_benchmark/runs/brushnet_sd15_random_s42/` |
| PowerPaint | `PowerPaint_benchmark/runs/powerpaint_v1_unipc_s42/` |
| SD1.5 Inpainting | `SD15_inpainting_benchmark/runs/sd15_inpainting_unipc_s42/` |
| SD2.0 Inpainting | `SD20_inpainting_benchmark/runs/sd20_inpainting_unipc_s42/` |

- 评分输入来自各运行目录的 `generated/<sample_id>.png`，均为 256×256 RGB。
- 数据集为 `datasets/test_dataset_1500/`。虽然目录名包含 1500，实际正式样本数是 **1550**，本次按实际完整样本集评测。
- 四组生成配置均记录 seed=42、50 steps、CFG=7.5、UniPC；本次没有改变这些生成参数。
- 数据集包含 **250 条不同 prompt**。主结果为图像等权平均，不是 prompt 等权平均，因此不能把 1550 张图理解为 1550 条独立指令。

## 4. 统一评测口径

本次直接复用 `SD15_inpainting_benchmark/evaluate.py` 中原有的 `clip_scores`，避免另写近似实现造成口径偏移。

设生成 RGB 图像为 \(I\)，数据集原始文本为 \(T\)，二值编辑掩码为 \(M\)。设 \(\widehat{E_I}\)、\(\widehat{E_T}\) 分别表示 CLIP 投影特征经 L2 归一化后的结果。

\[
\mathrm{Global}(I,T)=100\cdot\left\langle\widehat{E_I}(I),\widehat{E_T}(T)\right\rangle
\]

\[
\mathrm{Local}(I,T,M)=100\cdot\left\langle\widehat{E_I}(I\odot M),\widehat{E_T}(T)\right\rangle
\]

### 4.1 Global

直接使用完整生成图与数据集 `prompts.json` 中该样本的原始 prompt 计算相似度。评分文本不添加 PowerPaint 的 `P_obj`，不使用生成阶段的负提示词。

### 4.2 Local

1. 将原始 mask 转为灰度。
2. 以 NEAREST 缩放到生成图尺寸。
3. 灰度值 **严格大于 127** 的像素定义为编辑区。
4. 编辑区内保留生成内容，区外 RGB 全部设为黑色 `(0, 0, 0)`。
5. 将这张完整尺寸的黑背景图交给 CLIP，文本与 Global 完全相同。

**Local 不是 bbox 紧裁剪、不是局部放大，也不是掩码内像素特征平均。** 它对掩码面积、位置和黑背景占比存在敏感性。

### 4.3 处理器、精度与聚合

- 每个 CLIP 使用自己的配套 `CLIPProcessor`，没有复用错误型号的模型参数。
- 独立验收确认：两个模型的**实际图像处理器配置相同**，对本数据集的**全部 prompt 分词 token ID 相同**。
- 图像处理：短边 BICUBIC 缩放到 224、224×224 中心裁剪、像素乘以 \(1/255\)，再执行 CLIP 标准归一化。这里仍有 CLIP 标准中心裁剪，不应把“Local 不是 bbox 裁剪”误写为“没有裁剪”。
- 归一化 mean：`[0.48145466, 0.4578275, 0.40821073]`；std：`[0.26862954, 0.26130258, 0.27577711]`。
- 文本最大长度 77 token，启用 padding 和 truncation；本数据集最长仅 **27 token**，两种评测器实际截断样本数均为 **0**。
- 模型为 FP32、`eval()`、`inference_mode()`；关闭 CUDA matmul TF32。每次同时处理一个样本的 Global 图和 Local 图，图像 batch=2。
- 分数不截断负值，不使用 `logit_scale`，不做 softmax，也不是概率。
- 对全部 1550 个样本逐图等权取算术均值，并报告总体标准差；任何缺失或非有限分数都会使验收失败。

## 5. 完成情况与可复现性校验

### 5.1 八组完成记录

| Benchmark | B/32 有效样本 | L/14 有效样本 | B/32 评分耗时 | L/14 评分耗时 |
|---|---:|---:|---:|---:|
| BrushNet | 1550/1550 | 1550/1550 | 56.05 秒 | 196.04 秒 |
| PowerPaint | 1550/1550 | 1550/1550 | 50.00 秒 | 192.46 秒 |
| SD1.5 Inpainting | 1550/1550 | 1550/1550 | 50.25 秒 | 192.67 秒 |
| SD2.0 Inpainting | 1550/1550 | 1550/1550 | 50.55 秒 | 192.11 秒 |

每组有效样本数同时适用于 Global 和 Local。表内计时包含 `clip_scores` 的数据读取、预处理和 GPU 特征计算，不包含模型加载或额外完整性校验，不应当作纯模型吞吐率对比。

评测程序记录总耗时约 **1029.98 秒，即 17 分 10 秒**，包含输入预检、模型加载、八组评分和程序内复核；不包含单独的 B/32 资产下载／转换及末尾独立审计。开始时间为北京时间 10:17:54。

### 5.2 L/14 回归检查

| Benchmark | Global 均值差 | Global 逐样本最大绝对差 | Local 均值差 | Local 逐样本最大绝对差 |
|---|---:|---:|---:|---:|
| BrushNet | 0 | 0 | 0 | 0 |
| PowerPaint | 0 | 0 | 0 | 0 |
| SD1.5 Inpainting | 0 | 0 | 0 | 0 |
| SD2.0 Inpainting | 0 | 0 | 0 | 0 |

差值均相对于对应原运行目录的旧 L/14 结果。这里的零是完整精度比较结果，而不只是表格四舍五入后的零。

### 5.3 独立验收覆盖项

- 八个“benchmark × CLIP”组合唯一且完整；各组恰好 1550 条记录，ID 不重复、不缺失。
- 全部逐样本分数为有限数值，并从逐样本记录重新计算均值、标准差，核对汇总表。
- 四组使用相同样本 ID 集合及同一数据集 manifest。
- 原始图片、mask、prompt、生成图片的摘要验证通过；数据文件的 mtime/ctime 均早于本次评测开始时间，末尾再次核对了数据摘要。
- 每张生成图均与原成功记录中的输出摘要一致，并与旧评测的输出集合一致。
- 原 `config.json`、`generation_summary.json`、`job_status.json`、旧汇总／逐样本评测文件均未改变。
- 两个 CLIP 的权重与处理器文件摘要、实际处理器配置和分词结果均经过核验。
- 四组 L/14 的旧逐样本基线同样经过 ID 唯一性与有限数值检查。
- 评测脚本摘要与结果记录一致；五项轻量单元测试通过，覆盖重复 ID、无效数值、总体标准差、Local mask 阈值和负分不截断。

**生成溯源边界：**SD1.5/SD2.0 的逐样本记录包含生成配置摘要并通过校验；BrushNet/PowerPaint 的旧记录没有该字段。本次能确认它们使用的是原成功记录及旧评测对应的同一批图片，但不会据此声称补建了它们缺失的“输出—生成配置”历史绑定。

## 6. 环境与复现入口

统一使用 `SD15_inpainting_benchmark/.venv/bin/python`，避免四套虚拟环境的细微依赖差异进入本次对比。

| 项目 | 版本／设置 |
|---|---|
| GPU | NVIDIA Tesla T4，单卡顺序执行 |
| Python | 3.9.16 |
| PyTorch / CUDA | 2.1.2+cu121 / 12.1 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.37.2 |
| tokenizers | 0.15.2 |
| huggingface-hub | 0.20.3 |
| safetensors | 0.4.2 |
| NumPy / Pillow | 1.26.3 / 10.2.0 |
| 评测精度 | FP32，CUDA matmul TF32 关闭 |

脚本均位于 `SD15_inpainting_benchmark/`：

- `prepare_clip_b32.py`：准备固定版本的 B/32 本地安全权重与来源记录。
- `evaluate_clip_comparison.py`：八组离线 CLIP-only 全量重评，直接复用旧 `clip_scores`。
- `audit_clip_comparison.py`：独立验收完整性、数据一致性和回归结果。
- `test_clip_comparison.py`：轻量口径及验收测试。

在 `/data/workspace/SD15_inpainting_benchmark/` 目录下，可用以下入口复现：

1. `.venv/bin/python prepare_clip_b32.py`：准备或校验 B/32 资产；只有资产缺失时才需要外网。
2. `.venv/bin/python evaluate_clip_comparison.py --output-dir /data/workspace/SD15_inpainting_benchmark/runs/clip_comparison_rerun --device cuda`：读取原四组生成图，全量重评。
3. `.venv/bin/python audit_clip_comparison.py --results-dir /data/workspace/SD15_inpainting_benchmark/runs/clip_comparison_rerun`：在评测完成后验收。
4. `.venv/bin/python -m unittest -v test_clip_comparison`：运行轻量测试。

`--output-dir` 必须是工作区内**尚不存在的新目录**，已有目录会被拒绝，以免覆盖结果；再次复现时应更换目录后缀。原四组生成图片、数据集和两个本地 CLIP 资产必须保留。正式评测阶段设置 Hugging Face／Transformers 离线模式，不自动下载权重。

## 7. 结果文件索引

本次结果根目录：`/data/workspace/SD15_inpainting_benchmark/runs/clip_comparison_20261009/`。

GitHub 发布说明：下表是本地原始运行文件索引，`runs/` 与日志不随源码上传。可在线查看[已发布聚合结果](benchmark_results_2026-10-09.json)及[四方法综合分析报告](Benchmark综合结果分析报告_2026-10-09.md)；后者补充了配对比例、prompt 宏平均和掩码面积敏感性，不应将本专题的图像等权结论直接推广到其他聚合协议。

| 文件 | 用途 |
|---|---|
| [summary.json](../SD15_inpainting_benchmark/runs/clip_comparison_20261009/summary.json) | 八组完整精度汇总、方法定义、环境和模型来源 |
| [audit.json](../SD15_inpainting_benchmark/runs/clip_comparison_20261009/audit.json) | 独立验收结论，状态为 `passed` |
| [status.json](../SD15_inpainting_benchmark/runs/clip_comparison_20261009/status.json) | 作业状态 `complete`、完成组数 8、退出码 0 |
| [models_metadata.json](../SD15_inpainting_benchmark/runs/clip_comparison_20261009/models_metadata.json) | 两个 CLIP 的版本、文件摘要、处理器配置、截断统计和加载信息 |
| [input_provenance.json](../SD15_inpainting_benchmark/runs/clip_comparison_20261009/input_provenance.json) | 样本／数据集／生成图摘要及旧文件保护记录 |
| `*_benchmark_b32.json`、`*_benchmark_l14.json` | 每个 benchmark、每个 CLIP 的独立汇总 |
| `*_benchmark_b32_per_sample.json`、`*_benchmark_l14_per_sample.json` | 八份逐样本分数，每份 1550 条 |
| [评测日志](../SD15_inpainting_benchmark/logs/clip_comparison_20261009.log) | 完整进度与各组完成记录 |

B/32 来源记录：`SD15_inpainting_benchmark/models/clip_b32/sources.json`。L/14 来源记录：`SD15_inpainting_benchmark/models/sources.json` 的 `clip` 字段。所有权重与模型配置的摘要均在这些记录及结果元数据中保留。

## 8. 解读限制与建议

1. **只在同一评测器、同一指标列内比较方法。** B/32 的 27 分和 L/14 的 22 分不能直接比较优劣。
2. **Global 与 Local 测量对象不同。** Global 可能受未编辑背景影响；Local 的黑背景处理会受掩码面积、位置及缩放影响。二者均不能独立验证编辑正确性。
3. **生成策略不完全相同。** PowerPaint 使用其特定正／负提示策略，其他 benchmark 未采用完全相同策略；这是现有系统配置的比较，不是严格控制一切变量的网络结构消融。
4. **本次是固定生成结果的重评，不是多 seed 重复实验。** seed 同为 42 也不保证不同方法消耗同一随机噪声序列。
5. **未报告显著性。** 样本间标准差不能替代配对置信区间；250 条不同 prompt 的重复分布也意味着不能简单假设全部图像彼此独立。
6. 对当前结果最稳妥的表述是：**在两种 CLIP 编码器下，PowerPaint 的 Global 均值优势与 BrushNet 的 Local 均值优势方向一致；Local 的完整排序一致，而 Global 的中间排序对编码器有依赖。**

如用于论文或实验报告，建议同时保留两套 CLIP 列、明确 Local 的黑背景定义，并将这里的语义匹配结论与已有背景保真／边界／感知距离结果分开陈述。

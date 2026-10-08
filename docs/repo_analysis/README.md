# Text2Earth 代码仓库分析（总览与索引）

> 仓库：https://github.com/chen-yang-liu/Text2Earth
> 本地路径：`/data/workspace/Text2Earth`
> 分析时的 HEAD：`358da75`（2026-05-22，“Update README.md”），共 42 个提交
> 分析方法：逐行阅读核心代码；**用 Git blob 哈希与 HuggingFace diffusers 主干历史逐一比对，定位作者修改过的文件**；读取 HuggingFace 上模型权重的 safetensors 头部和配置文件，核实网络结构与参数量；对增强模型权重做静态 pickle 反汇编（不执行）。

---

## 文档目录

| 文件 | 内容 |
|---|---|
| [01_仓库结构与来源.md](./01_仓库结构与来源.md) | 目录结构、代码来源（diffusers 快照）、上游基线提交定位、**作者改动的完整清单**、提交历史、安装方式 |
| [02_文生图Pipeline实现详解.md](./02_文生图Pipeline实现详解.md) | `Text2EarthDiffusionPipeline` 逐段解析；分辨率条件从提示词到 UNet 的完整数据流；CFG 实现；与 SD 原版的逐行差异 |
| [03_局部重绘与ControlNet实现详解.md](./03_局部重绘与ControlNet实现详解.md) | `Text2EarthDiffusionInpaintPipeline`（9 通道输入、软掩码）；被就地修改的 `StableDiffusionControlNetPipeline` / `ControlNetModel` |
| [04_画质增强工具解析.md](./04_画质增强工具解析.md) | `Tools/visual_enhancement.py`：RRDBNet 网络结构、权重格式、推理流程及注意事项 |
| [05_模型权重与论文代码对照.md](./05_模型权重与论文代码对照.md) | HF 权重的组件/参数量/精度核实；论文中每个概念对应的代码位置；**哪些已实现、哪些未开源** |
| [06_问题清单与使用建议.md](./06_问题清单与使用建议.md) | 已发现的 Bug、隐患、安全问题；在本服务器（Tesla T4）上的部署和使用建议 |

---

## 一页速览

### 这个仓库是什么

**它是 HuggingFace `diffusers` 的一个完整快照（版本号 `0.29.0.dev0`，对应上游主干 2024-06-07 左右），作者在里面加入了两个 Text2Earth 专用 pipeline，并对 ControlNet 相关代码做了少量修改。**
仓库里**没有训练代码和评测代码**，只能用于推理；另外附带一个独立的数据集画质增强脚本和权重。

```
Text2Earth/
├── src/diffusers/          ← diffusers 快照（412 个文件，作者只改了其中 12 个）
│   └── pipelines/stable_diffusion/
│       ├── pipeline_text2earth_diffusion.py          ★ 文生图（分辨率可控）
│       └── pipeline_text2earth_diffusion_inpaint.py  ★ 局部重绘 / 编辑
├── Tools/
│   ├── visual_enhancement.py        ★ Git-10M 画质增强（RRDBNet ×2）
│   └── best_PSNR_iter_22000_new.pth ★ 增强模型权重（67 MB）
├── images/                 ← README 配图 + 重绘示例图和掩码
├── tests/ utils/ Makefile setup.py pyproject.toml ...  ← diffusers 原样文件
└── 0.2.0 / .idea/          ← 误提交的 pip 日志和 IDE 配置
```

### 作者实际写了 / 改了什么

| 类别 | 文件 | 改动 |
|---|---|---|
| **新增** | `pipeline_text2earth_diffusion.py` | 复制自 `StableDiffusionPipeline`，**新增约 35 行**：解析 `{Level}_GOOGLE_LEVEL_` 前缀 → 构造分辨率张量 → 作为 `class_labels` 送入 UNet，无条件分支置 0 |
| **新增** | `pipeline_text2earth_diffusion_inpaint.py` | 复制自 `StableDiffusionInpaintPipeline`，同样的分辨率逻辑；另外**关闭掩码二值化**，默认 `guidance_scale` 从 7.5 改为 3.5 |
| 修改 | `pipelines/controlnet/pipeline_controlnet.py` | **就地修改**官方 ControlNet pipeline，加入相同的分辨率逻辑（对应论文的图像翻译实验，但没有发布 ControlNet 权重） |
| 修改 | `models/controlnet.py` | 1 行：`class_labels` 的 dtype 对齐 |
| 修改 | 3 个 `__init__.py` | 注册并导出两个新 pipeline |
| 仅格式 | 5 个文件 | 只有空白、换行差异，逻辑不变 |
| **新增** | `Tools/visual_enhancement.py` + 权重 | 数据集画质增强（Real-ESRGAN 风格的 RRDBNet，2 倍超分） |

### 核心机制（一句话）

> 把提示词里的 Google 地图缩放级别（10 ~ 18）解析出来，作为 **UNet 的 `class_labels`** 传入。UNet 配置为 `class_embed_type="timestep"`，于是这个整数先经过与时间步相同的正弦编码，再经过一个独立的 MLP，**加到时间步嵌入上**，从而在每一个去噪步、每一个 ResNet 块里调节生成尺度。CFG 的无条件分支使用“空文本 + Level 0”。

### 关键发现

1. **代码改动极小**：真正有逻辑意义的新增代码只有几十行，绝大部分能力来自 diffusers 内置的 `class_embed_type="timestep"` 和模型权重本身。
2. **模型基于 SD2**：权重配置显示两个模型分别从 `stable-diffusion-2-base` 和 `stable-diffusion-2-inpainting` 初始化；总参数量 1.292B，与论文的 1.3B 一致。
3. **存在若干 Bug 和隐患**：例如只传 `prompt_embeds` 时 `res` 未定义；`num_images_per_prompt` 被强制为 1；默认输出尺寸为 512（训练尺寸是 256）；包名仍是 `diffusers`，安装后会覆盖环境里的官方 diffusers；官方 ControlNet pipeline 被就地修改等。详见 [06](./06_问题清单与使用建议.md)。
4. **README 与代码有出入**：增强脚本的文件名和参数名不一致；增强脚本实际输出 512×512（2 倍超分），README 没有说明；README 声称 MIT 许可，但 `LICENSE` 文件是 Apache-2.0。

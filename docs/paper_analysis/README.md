# Text2Earth 论文分析（总览与索引）

> 论文：**Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model**
> 作者：Chenyang Liu, Keyan Chen, Rui Zhao, Zhengxia Zou, Zhenwei Shi（通讯作者）—— 北京航空航天大学宇航学院 / 上海人工智能实验室 / 新加坡国立大学
> 发表：IEEE Geoscience and Remote Sensing Magazine (GRSM), 2025, pp. 2–23, DOI: 10.1109/MGRS.2025.3560455
> 预印本：arXiv:2501.00895（本分析基于 v2，2025-03-20，共 19 页）
> 项目主页：https://chen-yang-liu.github.io/Text2Earth/
> 本地 PDF：`/data/workspace/paper/text2earth.pdf`

---

## 文档目录

| 文件 | 内容 |
|---|---|
| [01_背景动机与相关工作.md](./01_背景动机与相关工作.md) | 研究问题、遥感文生图的特殊性、现有数据集/模型的局限、GAN / 自回归 / 扩散三条技术路线及遥感领域前序工作 |
| [02_Git-10M数据集.md](./02_Git-10M数据集.md) | 数据来源、采集策略、画质增强、GPT-4o 自动标注流水线、数据统计分析 |
| [03_Text2Earth模型与方法.md](./03_Text2Earth模型与方法.md) | 模型结构（VAE / UNet / OpenCLIP / 分辨率嵌入）、扩散公式、分辨率引导机制、掩码条件编辑、动态条件自适应 DCA（训练与采样算法） |
| [04_实验设置与结果.md](./04_实验设置与结果.md) | 训练配置、评测指标、RSICD 对比、引导系数消融、图像编辑、无界场景构建、跨模态生成、数据增强、Git-RSCLIP |
| [05_评述_局限与思考.md](./05_评述_局限与思考.md) | 贡献评价、方法本质、论文内部不一致之处、可复现性、局限性、后续研究方向 |

---

## 一页速览

### 1. 要解决什么问题

遥感图像是“上帝视角”的，覆盖范围广、场景多样、**分辨率跨度极大（0.5 m ~ 百米级/像素）**。因此一个实用的遥感文生图系统需要同时具备：

1. **全球尺度（Global-scale）**：能生成世界各地各种地貌；
2. **多分辨率可控（Multi-resolution controllable）**：用户能指定 GSD（地面采样距离）；
3. **无界生成（Unbounded）**：不受固定画布尺寸约束，能拼接出任意大的场景。

现有工作在两方面受限：
- **数据**：UCM（2.1k）、RSICD（10.9k）、NWPU-Captions（31.5k）、RS5M-RS3（2M）规模小、地域/场景单一、**普遍不带分辨率信息**；
- **模型**：GAN/Transformer 类方法难以刻画全球复杂地物结构，忽视分辨率属性，且只能做固定尺寸的基础文生图，泛化到编辑、外扩等任务的能力弱。

### 2. 两大贡献

| 贡献 | 要点 |
|---|---|
| **Git-10M 数据集** | 1050 万图文对，比此前最大的数据集大约 5 倍；全球分布；带**地理位置 + 分辨率**元数据；分辨率覆盖 0.5 m ~ 128 m/pixel；文本由 GPT-4o 自动生成（平均约 52 词，总计超过 5.5 亿词——论文原文写作 “5.5 billion”，与 10.5M × 52 不符，见 05 篇）；所有图像经过自研增强模型去噪/去伪影 |
| **Text2Earth 基础模型** | 13 亿参数的潜空间扩散模型（VAE + OpenCLIP ViT-H 文本编码器 + 带交叉注意力的 UNet + **分辨率嵌入模块**）；提出**分辨率引导机制**与**动态条件自适应（DCA）**策略；同时发布文生图版 `Text2Earth` 与编辑版 `Text2Earth-inpainting` |

### 3. 核心技术点

- **分辨率引导**：把分辨率 \(I_s\) 经投影层编码为 \(\rho = f_\theta(I_s)\)，与时间步嵌入相加：\(c_{st} = \rho + g_\theta(t)\)，在每个去噪步注入 UNet。
  实际实现中分辨率以“Google 地图缩放级别 Level”表示，\(\text{Resolution} = 2^{17-\text{Level}}\)，Level ∈ [10, 18] 对应 128 m ~ 0.5 m。
- **DCA 训练**：以一定概率分别随机丢弃文本条件与分辨率条件，让模型同时学会条件生成与无条件生成。
- **DCA 采样**：类 Classifier-Free Guidance，\(\epsilon_g = (1+\omega)\,\epsilon_\theta(z_t,t,\tau,\rho) - \omega\,\epsilon_\theta(z_t,t,\tau_\varnothing,\rho_\varnothing)\)，**文本与分辨率被联合引导**。
- **掩码条件编辑**：将被掩码图像的潜变量 \(z_m\) 与噪声潜变量 \(z_t\) 在通道维拼接，实现局部重绘、去云、外扩（outpainting）。

### 4. 主要结果

- **RSICD 基准**（LoRA 微调后）：FID **24.49**（此前最好的 CRS-Diff 为 50.72，提升 26.23）、零样本分类 OA **90.26%**（提升 20.95 个百分点）、CLIP Score **25.62**。
- **引导系数**：ω = 3.0 时 FID 与 OA 折中最好。
- **零样本文生图**：可生成山脉、河流、城市、森林、农田、冰川、港口等多种场景；能只给分辨率、或只给文本生成。
- **下游能力**：图像编辑（去云、替换/添加地物）、无界场景构建（如 3500×1100、3700×1300 像素）、Text2SAR / Text2NIR / Text2PAN（LoRA）、图像翻译（PAN2RGB、NIR2RGB、超分、去雾，用 ControlNet 类模块实现）。
- **数据增强**：用合成图扩充 RSICD 分类训练集，VGG-19 / ResNet-18 / ViT-B-16 / Swin-S 精度提升 3.9 ~ 6.6 个百分点。
- **Git-RSCLIP**：在 Git-10M 上做对比学习预训练，6 个数据集零样本分类平均 72.80%，高于 RemoteCLIP（65.60%）和 GeoRSCLIP（67.97%）。

### 5. 一句话评价

> 本文的核心价值主要来自**数据工程**（千万级、带分辨率的全球遥感图文数据集）和**基础模型的工程化落地**；在方法层面，它是对 Stable Diffusion 2 框架的**轻量、有效的领域化扩展**：把分辨率当作“类别/时间步式”的全局条件注入，再配合 CFG 风格的条件丢弃。实验覆盖面广，但训练与评测代码尚未开源，部分实验细节（如 LoRA 配置、ControlNet 结构、外扩流程）描述较粗，复现门槛主要在数据与算力上。

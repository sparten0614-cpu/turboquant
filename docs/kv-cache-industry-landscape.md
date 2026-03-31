# KV Cache优化行业全景：论文、生产部署与我们的定位

**Date:** 2026-03-31
**Author:** 宁宁
**Purpose:** 摸清行业baseline，确定优化目标

---

## 一、大模型KV Cache的内存规模（基本面）

### Claude/GPT-4级别模型（~200B参数，1M上下文）

**不压缩（FP16）的理论计算：**

| 参数 | 典型值 | 说明 |
|------|--------|------|
| hidden_dim | 16,384 | 200B模型典型值 |
| num_layers | 80 | |
| num_kv_heads | 8 | GQA（已是标配） |
| head_dim | 128 | |
| 精度 | FP16 (2 bytes) | |

```
每token KV = 2(K+V) × 80层 × 8头 × 128维 × 2字节 = 327,680 bytes ≈ 320KB
1M tokens = 320KB × 1,000,000 = 320GB
```

**注意：GQA已经内置了压缩。** 如果用MHA（64个KV头），则：
```
MHA: 1M tokens = 2 × 80 × 64 × 128 × 2 × 1M = 2.56TB
GQA (8头): 320GB — 已经是8倍压缩
```

### 行业实际情况

| 层级 | 技术 | 压缩倍数 | 1M ctx KV大小 | 状态 |
|------|------|---------|-------------|------|
| 架构层 | MHA→GQA | 8x | 320GB | 所有新模型标配 |
| 架构层 | MHA→MLA (DeepSeek) | 28-93x | 3-9GB | DeepSeek独家 |
| 量化层 | FP8 | 2x (vs FP16) | 160GB | 行业主流（TRT-LLM, vLLM） |
| 量化层 | INT4/FP4 | 4x | 80GB | NVIDIA NVFP4（最新） |
| 量化层 | TurboQuant 3-bit | 6x | 53GB | 论文阶段，未大规模部署 |
| 淘汰层 | H2O/SnapKV等 | 5-20x | 16-64GB | 研究为主 |
| 组合 | GQA + FP8 | 16x | 20GB | 当前行业baseline |
| 组合 | GQA + FP4 | 32x | 10GB | NVIDIA前沿 |
| 组合 | MLA (DeepSeek) | 28-93x | 3-9GB | 最激进的生产部署 |

**关键结论：当前一线厂商的KV cache实际大小约为理论值的1/16到1/32。**

---

## 二、一线厂商部署情况

### OpenAI
- **架构：** GQA（GPT-oss确认64 query heads, 8 KV heads）
- **KV量化：** 未公开。推测FP8（行业标准）
- **Prefix Caching：** 已部署，缓存命中减50%费用，保留5-10分钟
- **GPT-4.1/5.1系列：** 24小时KV缓存保留，offload到GPU本地SSD
- **内部推理栈完全不透明**

### Google
- **TurboQuant论文：** ICLR 2026，6x压缩，3-bit，零质量损失
- **生产部署：** **未确认用于Gemini。** 论文测试用的是开源模型（Gemma, Mistral, Llama）
- **博客说法：** "designed for solving the KV cache bottleneck in models like Gemini"——但没有明确说已部署
- **SGLang/llama.cpp正在集成中**

### Anthropic (Claude)
- **架构+量化：** 完全不公开
- **Prompt Caching：** 已部署，缓存读取仅10%价格（90%折扣），延迟降低85%
- **Claude Code实测92%缓存命中率，81%成本降低**

### Meta (Llama)
- **GQA：** Llama-2 70B开始，现在是标配
- **KV cache量化：** 未公开Meta内部推理栈细节
- **开源贡献：** GQA成为行业标准的推动者

### DeepSeek
- **MLA（Multi-head Latent Attention）：** 最激进的KV压缩
- **原理：** 不存储完整K/V，而是压缩成512维latent向量，推理时重建
- **DeepSeek-V3：** KV cache从213.5GB→7.6GB（**28倍压缩**），吞吐量5.76x
- **代价：** 需要从头训练模型，不能后装
- **这是目前行业最强的KV压缩方案**

### Mistral
- **滑动窗口注意力(SWA)：** Mistral 7B使用4096窗口
- **后续大模型（Large, NeMo）放弃了SWA，改回全注意力**
- **GQA + 标准KV cache**

### NVIDIA (TensorRT-LLM)
- **FP8 KV cache：** 推荐方案，2x压缩，33%吞吐提升
- **NVFP4 KV cache：** 最新，4x压缩，<1%质量损失
- **FP8注意力计算：** 支持在FP8精度下直接做attention
- **Blackwell GPU：** NVFP4原生支持

---

## 三、推理框架默认配置

| 框架 | 默认KV精度 | 支持的量化 | 说明 |
|------|-----------|-----------|------|
| **vLLM** | BF16/FP16 | FP8 only | 不支持INT8（!），PagedAttention标配 |
| **TensorRT-LLM** | FP16 | INT8, FP8, NVFP4 | 最完整，推荐FP8 |
| **SGLang** | BF16 | FP8, FP4(开发中) | RadixAttention（前缀重用），MLA支持 |
| **TGI (HuggingFace)** | FP16 | INT2/4(quanto), INT2/4/8(HQQ) | KIVI启发 |
| **llama.cpp** | F16 | Q8_0, Q4_0 | Q8_0正在成为默认 |
| **DeepSpeed** | FP16 | KV offload到CPU/NVMe | 不是量化，是分层存储 |

---

## 四、论文全景（80+篇）

### 按技术类型分类

**量化方法（~25篇）**

| 论文 | 会议 | 压缩比 | 核心 |
|------|------|--------|------|
| KIVI | ICML 2024 | 2.6x | 2-bit非对称（K per-channel, V per-token） |
| KVQuant | NeurIPS 2024 | 4.8-8x | Pre-RoPE + 非均匀量化 + 稀疏outlier |
| QJL | AAAI 2025 | >5x | 1-bit JL变换，无偏内积 |
| TurboQuant | ICLR 2026 | 6x | PolarQuant + QJL，3-bit零损失 |
| KVTC | ICLR 2026 (NVIDIA) | 20x | JPEG风格：PCA去相关+自适应量化+熵编码 |
| ZipCache | NeurIPS 2024 | ~5x | 显著token识别+混合精度 |
| QServe | MLSys 2025 | 4x | W4A8KV4，SmoothAttention，已生产部署 |
| Atom | MLSys 2024 | 4x | 4-bit系统级协同设计 |
| QuaRot | NeurIPS 2024 | 4x | Hadamard旋转消除outlier |
| SpinQuant | ICLR 2025 | varies | 学习旋转矩阵 |
| RotateKV | IJCAI 2025 | 8x (2-bit) | Channel重排序+自适应旋转 |
| KVLinC | preprint | varies | K不旋转+线性校正 |
| GEAR | preprint | varies | 低秩量化残差近似 |
| FireQ | arXiv 2025 | 4x | INT4-FP8 Hopper协同设计 |
| QuantSpec | Apple 2025 | 4x | 自推测解码+4-bit KV |
| KVSplit | 2025 | 2-3x | K=8-bit, V=4-bit非对称 |

**淘汰/剪枝方法（~25篇）**

| 论文 | 会议 | 压缩比 | 核心 |
|------|------|--------|------|
| H2O | NeurIPS 2023 | 5x | 累积注意力"重击者" |
| StreamingLLM | ICLR 2024 | 常数内存 | Attention sinks + 滑动窗口 |
| ScissorHands | NeurIPS 2023 | 5x (20x+4bit) | 重要性持续性假设 |
| SnapKV | NeurIPS 2024 | 3.6x速度/8.2x内存 | 观测窗口投票 |
| PyramidKV | ACL 2024 | 88%压缩 | 金字塔形层级预算 |
| Quest | ICML 2024 | 7x自注意力加速 | 查询感知页级稀疏 |
| RazorAttention | arXiv 2024 | >70%缩减 | 检索头vs非检索头 |
| Ada-KV | NeurIPS 2025 | varies | 头级自适应预算 |
| ShadowKV | ICML 2025 Spotlight | 6x batch | 低秩K+V offload |
| RocketKV | ICML 2025 (NVIDIA) | 400x | 两阶段：粗粒度淘汰→混合稀疏注意力 |
| DynamicKV | EMNLP 2025 | 59x (1.7% KV) | 任务自适应 |
| ChunkKV | ICLR 2025 | varies | 语义块作为压缩单位 |
| FastGen | 2024 | 17x prefill加速 | 每头定制策略 |
| HashEvict | arXiv 2025 | 1.4-3x | LSH余弦相似度 |

**架构创新（~10篇）**

| 论文 | 会议 | 压缩比 | 核心 |
|------|------|--------|------|
| GQA | 2023 | 8x | 分组查询注意力（标配） |
| MQA | 2019 | Nx | 单KV头 |
| MLA | DeepSeek 2024 | 28-93x | 低秩latent压缩（最强） |
| YOCO | NeurIPS 2024 Oral | O(1)/层 | 解码器-解码器架构 |
| CLA | 2024 | 2-4x | 跨层KV共享 |
| LCKV | ACL 2024 | 2-4x | 层浓缩KV |
| xKV | arXiv 2025 | significant | 跨层SVD |
| TransMLA | arXiv 2025 | — | MLA可转换为标准MHA |

**低秩分解（~8篇）**

| 论文 | 会议 | 压缩比 | 核心 |
|------|------|--------|------|
| Palu | ICLR 2025 | 91%+量化 | SVD截断+在线重建 |
| DeltaKV | arXiv 2025 | 3.4x | 残差KV（减共享分量） |
| KVSculpt | arXiv 2025 | varies | KV压缩即蒸馏 |
| OjaKV | arXiv 2025 | varies | 在线低秩压缩 |

**混合/统一方法（~8篇）**

| 论文 | 会议 | 核心 |
|------|------|------|
| MiniKV | ACL 2025 | 2-bit量化+自适应淘汰，FlashAttention兼容 |
| MiKV | 2024 | "软淘汰"：淘汰=INT2残差，非删除 |
| EVICPRESS | arXiv 2025 | 压缩+淘汰联合优化 |
| Q-Hitter | MLSys 2024 | 稀疏-量化KV oracle |
| ThinKV | arXiv 2025 | 推理模型的思维自适应压缩 |

**系统级（~15篇）**

| 技术 | 状态 | 核心 |
|------|------|------|
| PagedAttention (vLLM) | 生产标配 | OS式分页内存管理 |
| FlashAttention v1/2/3 | 标配 | 融合注意力kernel |
| FlashInfer | 生产级 | 自适应kernel调度 |
| RadixAttention (SGLang) | 生产级 | 基数树前缀缓存 |
| FP8 KV cache | 生产标配 | 2x压缩 |
| NVFP4 KV cache | 最新 | 4x压缩 |
| Prefix caching | 标配 | 跨请求KV复用 |

---

## 五、TurboQuant是否已被大规模使用？

**答案：没有。**

证据：
1. 论文2026年初发表（ICLR 2026），太新
2. Google博客说"designed for Gemini"但没说"deployed in Gemini"
3. SGLang有feature request（issue #21618），但尚未合并
4. llama.cpp有讨论帖（discussion #20969），社区在探索
5. 没有任何一线厂商公开宣布使用TurboQuant

**行业主流仍是：GQA（架构） + FP8/FP4（量化） + Prefix caching（系统）**

---

## 六、竞争格局与我们的定位

### 当前行业水平线

| 组合方案 | 总压缩倍数(vs MHA+FP16) | 质量损失 | 部署难度 |
|---------|----------------------|---------|---------|
| GQA + FP16 | 8x | 0 | 需训练时 |
| GQA + FP8 | 16x | <0.1% | 简单，框架支持 |
| GQA + FP4 | 32x | <1% | NVIDIA最新 |
| GQA + TurboQuant 6-bit | 21x | <0.1% | 需集成 |
| GQA + TurboQuant 3-bit | 43x | ~0 (论文claim) | 需集成 |
| MLA (DeepSeek) | 28-93x | 0 | 需从头训练 |
| GQA + 淘汰(SnapKV) + FP8 | 80-160x | 可控 | 复杂 |
| GQA + KVTC 20x | 160x | 待验证 | NVIDIA研究 |

### 我们的目标定位

**短期（可立即做）：** 证明TurboQuant在GQA模型上实现更好的压缩/质量trade-off than FP8/FP4。

**中期（3-6个月）：** 组合量化+淘汰+非对称K/V，在开源框架（llama.cpp/vLLM/SGLang）实现：
- 目标：GQA + TurboQuant + 淘汰 = **50-100x总压缩**（vs MHA+FP16）
- 对标：NVIDIA KVTC的20x claim

**长期：** 成为开源LLM推理的KV cache优化标准方案，替代FP8成为新默认。

### 关键差异化

| 竞争者 | 我们的优势 |
|--------|----------|
| NVIDIA FP4/KVTC | 我们支持Apple Silicon（Metal），不依赖CUDA |
| KIVI/KVQuant | 我们有更好的理论基础（PolarQuant数学最优） |
| MLA (DeepSeek) | MLA需从头训练；我们是post-training，可用于任何模型 |
| 纯淘汰方法 | 我们是"量化+淘汰"统一框架，不丢信息 |

---

## 七、三份Survey论文（推荐精读）

1. **"A Survey on LLM Acceleration based on KV Cache Management"** — TMLR 2025 (arXiv:2412.19442)
   - 最全面，有配套GitHub: TreeAI-Lab/Awesome-KV-Cache-Management

2. **"KV Cache Optimization Strategies for Scalable and Efficient LLM Inference"** — arXiv 2025 (arXiv:2603.20397)
   - 最新（2026年3月），五个方向分类

3. **"SCBench: A KV Cache-Centric Analysis of Long-Context Methods"** — ICLR 2025
   - 最系统的benchmark框架

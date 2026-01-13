# 注意力机制实现

本仓库包含多种注意力机制的PyTorch实现，适用于自然语言处理和深度学习任务。

## 包含的注意力机制

### 1. 缩放点积注意力 (Scaled Dot-Product Attention)
- Transformer中的标准注意力机制
- 使用点积计算注意力分数，并进行缩放
- 支持掩码操作

### 2. 多头注意力 (Multi-Head Attention)
- 将注意力分割成多个头
- 每个头学习不同的表示
- 最后合并所有头的输出

### 3. 自注意力 (Self-Attention)
- 输入序列关注自身
- 包含残差连接和层归一化
- 用于捕捉序列内部依赖

### 4. 交叉注意力 (Cross-Attention)
- 一个序列关注另一个序列
- 用于编码器-解码器架构
- 支持掩码操作

### 5. 加性注意力 (Additive Attention)
- 也称为Bahdanau注意力
- 使用加性方式计算注意力分数
- 常用于Seq2Seq模型

### 6. 位置编码 (Positional Encoding)
- 为Transformer添加位置信息
- 使用正弦和余弦函数
- 支持不同长度的序列

## 文件结构

```
attention_mechanism/
├── attention.py          # 主要注意力机制实现
├── test_attention.py     # 单元测试
├── demo.py              # 使用演示
└── README.md           # 说明文档
```

## 快速开始

### 安装依赖
```bash
pip install torch
```

### 基本使用

```python
import torch
from attention import SelfAttention

# 创建自注意力层
d_model = 512
n_heads = 8
self_attention = SelfAttention(d_model, n_heads)

# 创建输入
batch_size = 2
seq_len = 10
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output, attention_weights = self_attention(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

### 使用掩码

```python
from attention import create_padding_mask

# 创建填充掩码
seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
mask = create_padding_mask(seq, pad_token_id=0)

# 使用掩码
output, attention_weights = self_attention(x, mask=mask)
```

## 运行测试

```bash
# 运行所有测试
python test_attention.py

# 运行演示
python demo.py
```

## 示例

### 示例1：创建Transformer风格的自注意力

```python
from attention import SelfAttention, PositionalEncoding

# 创建模型
d_model = 256
n_heads = 4
seq_len = 20

# 位置编码
pe = PositionalEncoding(d_model)
# 自注意力
attention = SelfAttention(d_model, n_heads)

# 处理输入
x = torch.randn(1, seq_len, d_model)
x_with_pe = pe(x)
output, weights = attention(x_with_pe)
```

### 示例2：机器翻译中的交叉注意力

```python
from attention import CrossAttention

# 编码器输出（源语言）
encoder_output = torch.randn(1, 15, 512)  # [batch, src_len, d_model]
# 解码器查询（目标语言）
decoder_query = torch.randn(1, 10, 512)   # [batch, tgt_len, d_model]

# 交叉注意力
cross_attention = CrossAttention(512, 8)
output, alignment = cross_attention(decoder_query, encoder_output)
```

### 示例3：Seq2Seq中的加性注意力

```python
from attention import AdditiveAttention

# 编码器隐藏状态
encoder_states = torch.randn(1, 20, 256)  # [batch, seq_len, hidden]
# 解码器当前隐藏状态
decoder_state = torch.randn(1, 256)       # [batch, hidden]

# 加性注意力
additive_attention = AdditiveAttention(256)
context, weights = additive_attention(decoder_state, encoder_states)
```

## 注意力机制比较

| 机制 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 缩放点积注意力 | 计算高效，可并行 | 需要大量内存 | Transformer基础 |
| 多头注意力 | 捕捉多种关系，表示能力强 | 参数多，计算复杂 | 需要丰富表示的任务 |
| 自注意力 | 捕捉长距离依赖，并行计算 | 二次复杂度 | 文本分类，语言建模 |
| 交叉注意力 | 跨序列对齐，灵活 | 需要两个输入序列 | 机器翻译，问答 |
| 加性注意力 | 简单直观，可解释性强 | 计算较慢，不能并行 | 传统Seq2Seq模型 |

## 掩码类型

1. **填充掩码**：忽略填充位置
2. **前瞻掩码**：防止看到未来信息（用于解码器）
3. **组合掩码**：同时使用填充和前瞻掩码

## 性能优化建议

1. 使用`torch.bmm`进行批量矩阵乘法
2. 合理设置`d_model`和`n_heads`的比例
3. 使用混合精度训练（FP16）
4. 对于长序列，考虑使用稀疏注意力或局部注意力

## 扩展功能

可以扩展的功能：
- 稀疏注意力（Sparse Attention）
- 局部注意力（Local Attention）
- 相对位置编码（Relative Positional Encoding）
- 线性注意力（Linear Attention）
- 核注意力（Kernel Attention）

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
3. Luong et al. "Effective Approaches to Attention-based Neural Machine Translation" (2015)

## 许可证

MIT License
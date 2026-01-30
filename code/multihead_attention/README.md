# 多头注意力机制 (Multi-Head Attention)

多头注意力机制是Transformer架构的核心组件，首次在论文《Attention Is All You Need》中提出。它允许模型同时关注输入序列的不同位置的不同表示子空间。

## 核心概念

### 1. 注意力机制 (Attention Mechanism)
注意力机制模拟了人类在处理信息时的选择性关注能力。在深度学习中，它允许模型在处理序列数据时，动态地关注输入的不同部分。

### 2. 缩放点积注意力 (Scaled Dot-Product Attention)
这是多头注意力的基础计算单元：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- **Q** (Query): 查询向量，表示我们想要关注什么
- **K** (Key): 键向量，表示输入的特征
- **V** (Value): 值向量，表示实际要提取的信息
- **d_k**: 键的维度，缩放因子防止梯度消失

### 3. 多头注意力 (Multi-Head Attention)
多头注意力将输入分割成多个"头"，每个头独立进行注意力计算：

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
其中 headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

## 文件结构

```
multihead_attention/
├── multihead_attention.py    # 核心实现
├── simple_example.py         # 简单示例
├── test_multihead_attention.py # 测试文件
└── README.md                 # 说明文档
```

## 核心组件

### 1. ScaledDotProductAttention
缩放点积注意力类，实现基本的注意力计算。

**主要功能：**
- 计算查询和键的点积
- 缩放点积结果
- 应用softmax得到注意力权重
- 将权重应用到值上

### 2. MultiHeadAttention
多头注意力类，实现完整的多头注意力机制。

**主要功能：**
- 线性投影到查询、键、值空间
- 分割输入为多个注意力头
- 每个头独立进行注意力计算
- 合并多个头的输出
- 最终线性投影

### 3. PositionalEncoding
位置编码类，为序列添加位置信息。

**为什么需要位置编码？**
- 注意力机制本身没有位置信息
- 需要告诉模型输入元素的顺序
- 使用正弦和余弦函数生成位置编码

## 使用方法

### 基本用法

```python
import torch
from multihead_attention import MultiHeadAttention

# 参数设置
d_model = 512      # 模型维度
n_heads = 8        # 注意力头数量
batch_size = 4     # 批次大小
seq_len = 32       # 序列长度

# 创建多头注意力层
attention = MultiHeadAttention(d_model, n_heads)

# 创建输入数据
x = torch.randn(batch_size, seq_len, d_model)

# 应用注意力
output, attention_weights = attention(x, x, x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

### 使用掩码

```python
from multihead_attention import create_padding_mask, create_look_ahead_mask

# 创建填充掩码
seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 0表示填充
padding_mask = create_padding_mask(seq)

# 创建前瞻掩码（用于解码器）
look_ahead_mask = create_look_ahead_mask(seq_len=10)

# 应用带掩码的注意力
output, weights = attention(x, x, x, mask=padding_mask)
```

### 添加位置编码

```python
from multihead_attention import PositionalEncoding

# 创建位置编码
pos_encoding = PositionalEncoding(d_model)

# 为输入添加位置编码
x_with_pos = pos_encoding(x)
```

## 数学原理

### 注意力计算

1. **线性投影**：
   ```
   Q = XW^Q, K = XW^K, V = XW^V
   ```

2. **分割多头**：
   ```
   Q_i = split_head(Q), K_i = split_head(K), V_i = split_head(V)
   ```

3. **缩放点积注意力**：
   ```
   Attention(Q_i, K_i, V_i) = softmax(Q_iK_i^T / √d_k) V_i
   ```

4. **合并多头**：
   ```
   MultiHead = Concat(head₁, ..., headₕ)W^O
   ```

### 位置编码公式

对于位置 `pos` 和维度 `i`：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 应用场景

### 1. 自注意力 (Self-Attention)
查询、键、值来自同一输入序列，用于编码器。

```python
# 自注意力
output, weights = attention(x, x, x)
```

### 2. 交叉注意力 (Cross-Attention)
查询来自一个序列，键值来自另一个序列，用于解码器。

```python
# 交叉注意力
output, weights = attention(query, key_value, key_value)
```

### 3. 掩码注意力 (Masked Attention)
使用掩码控制注意力范围。

```python
# 填充掩码：忽略填充标记
# 前瞻掩码：防止看到未来信息（用于语言模型）
```

## 优势

1. **并行计算**：多个注意力头可以并行计算
2. **表示能力**：每个头学习不同的关注模式
3. **可解释性**：注意力权重可视化帮助理解模型决策
4. **长距离依赖**：直接建模序列中任意两个位置的关系

## 运行示例

### 运行简单示例

```bash
cd ./code/multihead_attention
python simple_example.py
```

### 运行测试

```bash
cd ./code/multihead_attention
python test_multihead_attention.py
```

### 输出示例

```
============================================================
多头注意力机制演示
============================================================

参数设置:
批次大小: 2
序列长度: 8
模型维度: 64
注意力头数量: 4
每个头的维度: d_k = d_v = 16

创建输入张量...
输入形状: torch.Size([2, 8, 64])

创建多头注意力层...
总参数数量: 33,280
可训练参数数量: 33,280

测试多头注意力...
输入形状: torch.Size([2, 8, 64])
输出形状: torch.Size([2, 8, 64])
注意力权重形状: torch.Size([2, 4, 8, 8])
```

## 参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| d_model | 模型维度 | 512, 768, 1024 |
| n_heads | 注意力头数量 | 8, 12, 16 |
| d_k | 每个头的键/查询维度 | d_model / n_heads |
| d_v | 每个头的值维度 | d_model / n_heads |
| dropout | Dropout概率 | 0.1 |

## 常见问题

### 1. 为什么需要缩放因子 √d_k？
防止点积结果过大导致softmax梯度消失。

### 2. 多头注意力的优势是什么？
- 并行计算提高效率
- 每个头学习不同的表示
- 增强模型表达能力

### 3. 位置编码为什么使用正弦和余弦？
- 可以表示绝对位置和相对位置
- 可以外推到比训练时更长的序列
- 具有周期性，适合表示循环模式

### 4. 掩码的作用是什么？
- **填充掩码**：忽略填充标记，提高计算效率
- **前瞻掩码**：防止解码器看到未来信息，保证自回归性

## 扩展阅读

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
3. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 代码实现

## 许可证

本项目代码仅供学习使用，遵循MIT许可证。
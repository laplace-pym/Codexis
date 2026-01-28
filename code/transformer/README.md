# Transformer 实现

基于 "Attention Is All You Need" 论文的完整Transformer模型实现。

## 文件结构

- `transformer.py` - 核心Transformer实现
- `example.py` - 训练和推理示例
- `demo.py` - 各个组件的演示
- `README.md` - 说明文档

## 模型架构

### 主要组件

1. **位置编码 (PositionalEncoding)**
   - 正弦和余弦函数编码位置信息
   - 支持不同频率的位置编码

2. **多头注意力 (MultiHeadAttention)**
   - 并行计算多个注意力头
   - 缩放点积注意力
   - 支持mask机制

3. **前馈网络 (PositionwiseFeedForward)**
   - 每个位置独立的前馈网络
   - 两层线性变换 + ReLU激活

4. **编码器层 (EncoderLayer)**
   - 自注意力子层
   - 前馈网络子层
   - 残差连接 + 层归一化

5. **解码器层 (DecoderLayer)**
   - 自注意力子层（带look-ahead mask）
   - 交叉注意力子层
   - 前馈网络子层
   - 残差连接 + 层归一化

6. **完整Transformer (Transformer)**
   - 编码器堆叠
   - 解码器堆叠
   - 词嵌入 + 位置编码
   - 输出线性层

## 使用方法

### 1. 基本使用

```python
from transformer import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=10000,  # 源语言词汇表大小
    tgt_vocab_size=10000,  # 目标语言词汇表大小
    d_model=512,           # 模型维度
    n_heads=8,             # 注意力头数
    num_encoder_layers=6,  # 编码器层数
    num_decoder_layers=6,  # 解码器层数
    d_ff=2048,             # 前馈网络维度
    max_seq_length=100,    # 最大序列长度
    dropout=0.1            # dropout率
)

# 前向传播
src = torch.randint(0, 10000, (32, 20))  # [batch_size, seq_len]
tgt = torch.randint(0, 10000, (32, 20))
output = model(src, tgt[:, :-1])  # 训练时使用shifted right

# 生成序列
generated = model.generate(src, max_len=50)
```

### 2. 训练示例

```python
# 训练循环示例
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    for src, tgt in dataloader:
        # 准备输入输出
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # 前向传播
        output = model(src, tgt_input)
        
        # 计算损失
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. 运行演示

```bash
# 运行组件演示
python demo.py

# 运行训练示例
python example.py

# 测试模型
python -c "from transformer import test_transformer; test_transformer()"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `src_vocab_size` | 源语言词汇表大小 | - |
| `tgt_vocab_size` | 目标语言词汇表大小 | - |
| `d_model` | 模型维度 | 512 |
| `n_heads` | 注意力头数 | 8 |
| `num_encoder_layers` | 编码器层数 | 6 |
| `num_decoder_layers` | 解码器层数 | 6 |
| `d_ff` | 前馈网络维度 | 2048 |
| `max_seq_length` | 最大序列长度 | 100 |
| `dropout` | dropout率 | 0.1 |

## 特性

- ✅ 完整实现论文中的Transformer架构
- ✅ 支持训练和推理模式
- ✅ 包含mask机制（padding mask + look-ahead mask）
- ✅ 残差连接和层归一化
- ✅ 位置编码
- ✅ 多头注意力
- ✅ 示例训练代码
- ✅ 组件演示

## 扩展建议

1. **优化**：
   - 添加学习率调度器
   - 实现标签平滑
   - 添加梯度裁剪

2. **功能增强**：
   - 支持不同的位置编码方式
   - 添加相对位置编码
   - 实现不同的注意力机制

3. **应用场景**：
   - 机器翻译
   - 文本摘要
   - 对话生成
   - 代码生成

## 注意事项

1. 确保 `d_model` 能被 `n_heads` 整除
2. 训练时使用 `tgt[:, :-1]` 作为输入，`tgt[:, 1:]` 作为目标
3. 推理时使用 `generate()` 方法进行自回归生成
4. 使用适当的mask防止信息泄露

## 引用

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```
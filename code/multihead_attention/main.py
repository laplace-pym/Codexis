import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    
    参数:
        d_model: 输入维度
        num_heads: 注意力头的数量
        dropout: dropout概率
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key: 键张量 [batch_size, seq_len_k, d_model]
            value: 值张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量 [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_q, seq_len_k]
        
        返回:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并分割为多头
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # 2. 重塑为多头格式
        # [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算缩放点积注意力
        # Q: [batch_size, num_heads, seq_len_q, d_k]
        # K: [batch_size, num_heads, seq_len_k, d_k]
        # 注意力分数: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. 应用掩码（如果有）
        if mask is not None:
            # 确保掩码维度正确
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. 应用注意力权重到值
        # attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        # V: [batch_size, num_heads, seq_len_k, d_k]
        # output: [batch_size, num_heads, seq_len_q, d_k]
        output = torch.matmul(attention_weights, V)
        
        # 7. 合并多头
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_heads, d_k]
        output = output.transpose(1, 2).contiguous()
        # [batch_size, seq_len_q, num_heads, d_k] -> [batch_size, seq_len_q, d_model]
        output = output.view(batch_size, -1, self.d_model)
        
        # 8. 最终线性变换
        output = self.W_o(output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    位置编码，用于为序列添加位置信息
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        返回:
            添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """
    完整的Transformer块，包含多头注意力和前馈网络
    """
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        
        返回:
            处理后的张量
        """
        # 1. 多头注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


def create_padding_mask(seq, pad_token_id=0):
    """
    创建填充掩码
    
    参数:
        seq: 序列张量 [batch_size, seq_len]
        pad_token_id: 填充token的ID
    
    返回:
        掩码张量 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器）
    
    参数:
        size: 序列长度
    
    返回:
        掩码张量 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def test_multihead_attention():
    """测试多头注意力机制"""
    print("测试多头注意力机制...")
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建随机输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attention_weights = mha(query, key, value)
    
    print(f"输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"注意力权重形状错误: {attention_weights.shape}"
    
    print("✓ 测试通过!")
    
    return output, attention_weights


def test_transformer_block():
    """测试Transformer块"""
    print("\n测试Transformer块...")
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建Transformer块
    transformer_block = TransformerBlock(d_model, num_heads)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attn_weights = transformer_block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    
    print("✓ Transformer块测试通过!")
    
    return output, attn_weights


def test_with_masks():
    """测试带掩码的多头注意力"""
    print("\n测试带掩码的多头注意力...")
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建随机输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 创建填充掩码
    # 假设序列长度为10，前5个是真实token，后5个是填充
    seq = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])
    padding_mask = create_padding_mask(seq)
    
    # 前向传播带掩码
    output, attention_weights = mha(query, key, value, padding_mask)
    
    print(f"序列: {seq}")
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"带掩码的输出形状: {output.shape}")
    
    # 创建前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len)
    print(f"前瞻掩码形状: {look_ahead_mask.shape}")
    
    print("✓ 掩码测试通过!")
    
    return output, attention_weights


def main():
    """主函数"""
    print("=" * 60)
    print("多头注意力机制实现")
    print("=" * 60)
    
    # 测试1: 基本多头注意力
    test_multihead_attention()
    
    # 测试2: Transformer块
    test_transformer_block()
    
    # 测试3: 带掩码的注意力
    test_with_masks()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    
    # 演示如何使用
    print("\n使用示例:")
    print("1. 创建多头注意力层:")
    print("   mha = MultiHeadAttention(d_model=512, num_heads=8)")
    print("2. 前向传播:")
    print("   output, attn_weights = mha(query, key, value, mask)")
    print("3. 创建Transformer块:")
    print("   block = TransformerBlock(d_model=512, num_heads=8)")
    print("4. 添加位置编码:")
    print("   pe = PositionalEncoding(d_model=512)")
    print("   x_with_pe = pe(x)")


if __name__ == "__main__":
    main()
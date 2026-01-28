"""
Transformer模型实现
基于"Attention Is All You Need"论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [seq_len, batch_size, d_model]"""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性变换并分割为多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到value
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出线性变换
        output = self.W_o(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力子层（带mask）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 交叉注意力子层
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src, tgt):
        """生成源序列和目标序列的mask"""
        # 源序列mask（padding mask）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # 目标序列mask（padding mask + look-ahead mask）
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        """编码器前向传播"""
        # 词嵌入 + 位置编码
        src_embedded = self.dropout(self.positional_encoding(self.src_embedding(src).transpose(0, 1)))
        src_embedded = src_embedded.transpose(0, 1)
        
        # 编码器层
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)
        
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """解码器前向传播"""
        # 词嵌入 + 位置编码
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_embedding(tgt).transpose(0, 1)))
        tgt_embedded = tgt_embedded.transpose(0, 1)
        
        # 解码器层
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def forward(self, src, tgt):
        """完整的前向传播"""
        # 生成mask
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_linear(decoder_output)
        
        return output
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """生成序列（推理时使用）"""
        self.eval()
        batch_size = src.size(0)
        
        # 初始化目标序列
        tgt = torch.ones(batch_size, 1).fill_(start_token).long().to(src.device)
        
        # 编码源序列
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        encoder_output = self.encode(src, src_mask)
        
        # 自回归生成
        for i in range(max_len - 1):
            # 生成mask
            tgt_mask = self.generate_mask(src, tgt)[1]
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # 获取下一个token
            output = self.output_linear(decoder_output[:, -1:, :])
            next_token = output.argmax(-1)
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果所有序列都生成了结束符，则停止
            if (next_token == end_token).all():
                break
        
        return tgt


def create_sample_data(vocab_size=100, batch_size=4, seq_len=10):
    """创建示例数据"""
    # 源序列和目标序列
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(3, vocab_size, (batch_size, seq_len))
    
    # 添加开始和结束标记
    src[:, 0] = 1  # 开始标记
    tgt[:, 0] = 1  # 开始标记
    src[:, -1] = 2  # 结束标记
    tgt[:, -1] = 2  # 结束标记
    
    return src, tgt


def test_transformer():
    """测试Transformer模型"""
    print("测试Transformer模型...")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_seq_length=20
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    src, tgt = create_sample_data()
    
    # 前向传播
    output = model(src, tgt[:, :-1])
    
    print(f"输入形状: src={src.shape}, tgt={tgt[:, :-1].shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试生成
    generated = model.generate(src, max_len=15)
    print(f"生成序列形状: {generated.shape}")
    
    print("测试完成！")
    return model


if __name__ == "__main__":
    # 运行测试
    model = test_transformer()
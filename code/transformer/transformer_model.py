import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_heads(self, x):
        """将输入分割为多个头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """合并多个头"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分割为多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 注意力计算
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并头
        attn_output = self.combine_heads(attn_output)
        
        # 输出线性变换
        output = self.W_o(attn_output)
        
        return output, attn_probs

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力子层（带掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力子层
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_seq_len=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """生成用于解码器的掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        # 词嵌入 + 位置编码
        src_embedded = self.dropout(self.positional_encoding(self.src_embedding(src).transpose(0, 1)))
        
        # 编码器层
        enc_output = src_embedded.transpose(0, 1)  # [batch_size, seq_len, d_model]
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        return enc_output
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """解码器前向传播"""
        # 词嵌入 + 位置编码
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_embedding(tgt).transpose(0, 1)))
        
        # 解码器层
        dec_output = tgt_embedded.transpose(0, 1)  # [batch_size, seq_len, d_model]
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """完整的前向传播"""
        enc_output = self.encode(src, src_mask)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        return output
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """生成序列（推理时使用）"""
        self.eval()
        batch_size = src.size(0)
        
        # 编码源序列
        enc_output = self.encode(src)
        
        # 初始化目标序列
        tgt = torch.ones(batch_size, 1).fill_(start_token).long().to(src.device)
        
        for i in range(max_len - 1):
            # 生成掩码
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
            
            # 解码
            output = self.decode(tgt, enc_output, tgt_mask=tgt_mask)
            
            # 获取最后一个时间步的预测
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都结束了
            if (next_token == end_token).all():
                break
        
        return tgt
"""
使用注意力机制的文本分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class AttentionTextClassifier(nn.Module):
    """
    基于注意力机制的文本分类器
    使用自注意力捕捉句子中重要词语
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding(embedding_dim, 512)
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _create_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        return pe
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 添加位置编码
        if seq_len <= self.positional_encoding.size(1):
            embeddings = embeddings + self.positional_encoding[:, :seq_len]
        
        # 自注意力层
        attention_output, attention_weights = self.self_attention(
            embeddings, embeddings, embeddings,
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
        )
        
        # 残差连接和层归一化
        embeddings = self.layer_norm1(embeddings + attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward(embeddings)
        
        # 残差连接和层归一化
        embeddings = self.layer_norm2(embeddings + ff_output)
        
        # 池化：使用注意力加权的平均池化
        # 计算每个位置的注意力分数
        attention_scores = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len]
        
        # 应用softmax得到权重
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights_pooling = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        
        # 加权平均池化
        pooled = torch.sum(embeddings * attention_weights_pooling, dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits, attention_weights


class SimpleAttentionClassifier(nn.Module):
    """
    简化的注意力文本分类器
    更容易理解和实现
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 双向LSTM编码器
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入token IDs [batch_size, seq_len]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        # 词嵌入
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM编码
        lstm_output, _ = self.lstm(embeddings)  # [batch_size, seq_len, hidden_dim*2]
        
        # 计算注意力分数
        attention_scores = self.attention(lstm_output).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # 加权平均池化
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(lstm_output * attention_weights_expanded, dim=1)  # [batch_size, hidden_dim*2]
        
        # 分类
        logits = self.classifier(context_vector)
        
        return logits, attention_weights


def create_sample_data(vocab_size: int = 10000, max_len: int = 50):
    """创建示例数据"""
    batch_size = 4
    seq_len = torch.randint(10, max_len, (batch_size,))
    
    # 创建输入序列
    inputs = []
    for length in seq_len:
        seq = torch.randint(0, vocab_size, (length,))
        # 填充到最大长度
        if length < max_len:
            padding = torch.zeros(max_len - length, dtype=torch.long)
            seq = torch.cat([seq, padding])
        inputs.append(seq)
    
    inputs = torch.stack(inputs)
    
    # 创建注意力掩码
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, length in enumerate(seq_len):
        attention_mask[i, :length] = 1
    
    # 创建标签
    labels = torch.randint(0, 3, (batch_size,))
    
    return inputs, attention_mask, labels


def train_example():
    """训练示例"""
    print("训练注意力文本分类器示例")
    print("=" * 50)
    
    # 设置参数
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 3
    batch_size = 32
    
    # 创建模型
    model = SimpleAttentionClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
    
    # 创建示例数据
    inputs, attention_mask, labels = create_sample_data(vocab_size)
    
    print(f"输入形状: {inputs.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 前向传播
    logits, attention_weights = model(inputs)
    
    print(f"\n模型输出:")
    print(f"Logits形状: {logits.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    
    print(f"\n计算损失: {loss.item():.4f}")
    
    # 预测
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    
    print(f"预测准确率: {accuracy.item():.2%}")
    
    return model, inputs, attention_weights


def visualize_attention():
    """可视化注意力权重"""
    print("\n" + "=" * 50)
    print("注意力权重可视化示例")
    print("=" * 50)
    
    # 创建示例句子
    sentences = [
        "I love this movie it is amazing",
        "This product is terrible and waste of money",
        "The service was okay nothing special"
    ]
    
    # 模拟tokenization
    vocab = {"I": 0, "love": 1, "this": 2, "movie": 3, "it": 4, "is": 5, 
             "amazing": 6, "product": 7, "terrible": 8, "and": 9, "waste": 10,
             "of": 11, "money": 12, "The": 13, "service": 14, "was": 15,
             "okay": 16, "nothing": 17, "special": 18, "[PAD]": 19}
    
    # 模拟注意力权重
    attention_weights = [
        [0.1, 0.3, 0.05, 0.2, 0.05, 0.05, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 正面评论
        [0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.0, 0.2, 0.15, 0.05, 0.1, 0.05, 0.05],  # 负面评论
        [0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.25]  # 中性评论
    ]
    
    print("句子和注意力权重:")
    for i, (sentence, weights) in enumerate(zip(sentences, attention_weights)):
        print(f"\n句子 {i+1}: {sentence}")
        print("注意力权重分布:")
        
        words = sentence.split()
        for word, weight in zip(words, weights[:len(words)]):
            print(f"  {word}: {weight:.3f}")
        
        # 找出最重要的词
        max_idx = weights.index(max(weights[:len(words)]))
        print(f"  最重要的词: '{words[max_idx]}' (权重: {weights[max_idx]:.3f})")
    
    print("\n注意力机制帮助模型:")
    print("1. 识别句子中的关键词语")
    print("2. 理解词语之间的依赖关系")
    print("3. 提高分类准确性")


def main():
    """主函数"""
    print("注意力文本分类器示例")
    print("=" * 60)
    
    # 运行训练示例
    model, inputs, attention_weights = train_example()
    
    # 可视化注意力
    visualize_attention()
    
    print("\n" + "=" * 60)
    print("总结:")
    print("1. 注意力机制让模型能够关注输入中的重要部分")
    print("2. 在文本分类中，注意力可以帮助识别关键词语")
    print("3. 注意力权重可以可视化，增加模型的可解释性")
    print("4. 注意力机制广泛应用于各种NLP任务")
    
    print("\n实际应用:")
    print("- 情感分析")
    print("- 文本分类")
    print("- 命名实体识别")
    print("- 机器翻译")
    print("- 问答系统")


if __name__ == "__main__":
    main()
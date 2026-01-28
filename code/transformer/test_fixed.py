import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 测试导入
try:
    from transformer import Transformer, MultiHeadAttention, PositionalEncoding
    print("✅ 成功导入Transformer模块")
    
    # 创建一个小型模型测试
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        n_layers=2,  # 注意：参数名是 n_layers，不是 num_layers
        n_heads=4,
        d_ff=256
    )
    print("✅ 成功创建Transformer模型")
    
    # 测试前向传播
    import torch
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    src = torch.randint(1, 100, (batch_size, src_seq_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_seq_len))
    
    output = model(src, tgt)
    print(f"✅ 前向传播成功，输出形状: {output.shape}")
    
    # 测试生成
    generated = model.generate(src, max_len=15)
    print(f"✅ 序列生成成功，生成形状: {generated.shape}")
    
    # 测试注意力机制
    attention = MultiHeadAttention(d_model=64, n_heads=4)
    query = torch.randn(batch_size, src_seq_len, 64)
    key = torch.randn(batch_size, src_seq_len, 64)
    value = torch.randn(batch_size, src_seq_len, 64)
    
    attn_output, attn_weights = attention(query, key, value)
    print(f"✅ 注意力机制成功，输出形状: {attn_output.shape}")
    
    # 测试位置编码
    pe = PositionalEncoding(d_model=64, max_len=100)
    x = torch.randn(batch_size, src_seq_len, 64)
    x_with_pe = pe(x)
    print(f"✅ 位置编码成功，输出形状: {x_with_pe.shape}")
    
    print("\n🎉 所有测试通过！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
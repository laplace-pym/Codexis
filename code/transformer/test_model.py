import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch

# 测试导入
try:
    from transformer import Transformer
    print("✓ 成功导入Transformer模型")
    
    # 创建小模型测试
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_len=50
    )
    
    print(f"✓ 成功创建模型，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, 100, (batch_size, src_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_len))
    
    output = model(src, tgt)
    print(f"✓ 前向传播成功，输出形状: {output.shape}")
    
    # 测试编码器
    encoder_output = model.encode(src)
    print(f"✓ 编码器成功，输出形状: {encoder_output.shape}")
    
    print("\n✅ 所有测试通过！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
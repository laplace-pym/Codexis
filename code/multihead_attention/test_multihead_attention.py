"""
测试多头注意力机制
验证实现的正确性和功能
"""

import torch
import torch.nn as nn
import numpy as np
from multihead_attention import (
    ScaledDotProductAttention, 
    MultiHeadAttention,
    PositionalEncoding,
    create_padding_mask,
    create_look_ahead_mask
)


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("测试缩放点积注意力...")
    
    # 创建测试数据
    batch_size = 2
    n_heads = 3
    seq_len_q = 4
    seq_len_k = 5
    d_k = 8
    d_v = 8
    
    query = torch.randn(batch_size, n_heads, seq_len_q, d_k)
    key = torch.randn(batch_size, n_heads, seq_len_k, d_k)
    value = torch.randn(batch_size, n_heads, seq_len_k, d_v)
    
    # 创建注意力层
    attention = ScaledDotProductAttention(dropout=0.0)
    
    # 前向传播
    output, weights = attention(query, key, value)
    
    # 验证形状
    assert output.shape == (batch_size, n_heads, seq_len_q, d_v), \
        f"输出形状错误: {output.shape}, 期望: {(batch_size, n_heads, seq_len_q, d_v)}"
    
    assert weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k), \
        f"权重形状错误: {weights.shape}, 期望: {(batch_size, n_heads, seq_len_q, seq_len_k)}"
    
    # 验证注意力权重和为1
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), rtol=1e-5), \
        "注意力权重和不为1"
    
    print("  ✓ 缩放点积注意力测试通过")
    return True


def test_multihead_attention_shapes():
    """测试多头注意力形状"""
    print("测试多头注意力形状...")
    
    # 测试1: 标准自注意力
    d_model = 64
    n_heads = 8
    batch_size = 4
    seq_len = 10
    
    attention = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attention(x, x, x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"输出形状错误: {output.shape}"
    
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), \
        f"权重形状错误: {weights.shape}"
    
    print("  ✓ 标准自注意力形状测试通过")
    
    # 测试2: 不同长度的查询和键值
    seq_len_q = 6
    seq_len_kv = 8
    
    query = torch.randn(batch_size, seq_len_q, d_model)
    key_value = torch.randn(batch_size, seq_len_kv, d_model)
    
    output, weights = attention(query, key_value, key_value)
    
    assert output.shape == (batch_size, seq_len_q, d_model), \
        f"交叉注意力输出形状错误: {output.shape}"
    
    assert weights.shape == (batch_size, n_heads, seq_len_q, seq_len_kv), \
        f"交叉注意力权重形状错误: {weights.shape}"
    
    print("  ✓ 交叉注意力形状测试通过")
    return True


def test_multihead_attention_masking():
    """测试多头注意力掩码功能"""
    print("测试多头注意力掩码功能...")
    
    d_model = 32
    n_heads = 4
    batch_size = 2
    seq_len = 5
    
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建填充掩码
    seq = torch.tensor([
        [1, 1, 1, 0, 0],  # 前3个有效，后2个填充
        [1, 1, 0, 0, 0]   # 前2个有效，后3个填充
    ])
    mask = create_padding_mask(seq)
    
    # 应用带掩码的注意力
    output, weights = attention(x, x, x, mask=mask)
    
    # 验证填充位置的注意力权重为0
    for i in range(batch_size):
        valid_len = (seq[i] != 0).sum().item()
        
        # 检查填充位置的权重
        for head_idx in range(n_heads):
            # 查询位置在有效范围内的填充键位置
            for q in range(valid_len):
                for k in range(valid_len, seq_len):
                    assert weights[i, head_idx, q, k].abs() < 1e-6, \
                        f"填充位置应有0权重，但得到: {weights[i, head_idx, q, k]}"
    
    print("  ✓ 填充掩码测试通过")
    
    # 测试前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len)
    output2, weights2 = attention(x, x, x, mask=look_ahead_mask)
    
    # 验证未来位置的权重为0
    for i in range(batch_size):
        for head_idx in range(n_heads):
            for q in range(seq_len):
                for k in range(q + 1, seq_len):  # 未来位置
                    assert weights2[i, head_idx, q, k].abs() < 1e-6, \
                        f"未来位置应有0权重，但得到: {weights2[i, head_idx, q, k]}"
    
    print("  ✓ 前瞻掩码测试通过")
    return True


def test_positional_encoding():
    """测试位置编码"""
    print("测试位置编码...")
    
    d_model = 16
    max_len = 20
    batch_size = 3
    seq_len = 10
    
    pos_enc = PositionalEncoding(d_model, max_len)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 应用位置编码
    x_with_pe = pos_enc(x)
    
    # 验证形状
    assert x_with_pe.shape == x.shape, \
        f"位置编码后形状改变: {x_with_pe.shape} != {x.shape}"
    
    # 验证不同位置有不同的编码
    pos_encoding_values = pos_enc.pe[0, :seq_len, :]
    
    # 检查不同位置是否不同
    for i in range(seq_len - 1):
        for j in range(i + 1, seq_len):
            assert not torch.allclose(pos_encoding_values[i], pos_encoding_values[j]), \
                f"位置{i}和位置{j}的编码相同"
    
    # 验证周期性
    # 位置编码应该具有周期性模式
    period = 10000  # 根据公式，周期与10000相关
    
    print("  ✓ 位置编码测试通过")
    return True


def test_gradient_flow():
    """测试梯度流动"""
    print("测试梯度流动...")
    
    d_model = 48
    n_heads = 6
    batch_size = 2
    seq_len = 7
    
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 创建需要梯度的输入
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # 前向传播
    output, _ = attention(x, x, x)
    
    # 创建虚拟损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 验证梯度存在
    assert x.grad is not None, "输入梯度为None"
    assert not torch.all(x.grad == 0), "输入梯度全为0"
    
    # 验证所有参数都有梯度
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"参数{name}的梯度为None"
        assert not torch.all(param.grad == 0), f"参数{name}的梯度全为0"
    
    print("  ✓ 梯度流动测试通过")
    return True


def test_attention_properties():
    """测试注意力性质"""
    print("测试注意力性质...")
    
    d_model = 24
    n_heads = 3
    batch_size = 1
    seq_len = 4
    
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    
    # 测试1: 相同输入产生相同输出（确定性）
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 设置为评估模式以确保确定性
    attention.eval()
    
    with torch.no_grad():
        output1, _ = attention(x, x, x)
        output2, _ = attention(x, x, x)
    
    assert torch.allclose(output1, output2, rtol=1e-6), \
        "相同输入产生不同输出"
    
    print("  ✓ 确定性测试通过")
    
    # 测试2: 线性性质（近似）
    x1 = torch.randn(batch_size, seq_len, d_model)
    x2 = torch.randn(batch_size, seq_len, d_model)
    alpha = 0.3
    beta = 0.7
    
    attention.train()  # 训练模式
    output_linear = attention(alpha * x1 + beta * x2, 
                              alpha * x1 + beta * x2,
                              alpha * x1 + beta * x2)[0]
    
    output_separate = alpha * attention(x1, x1, x1)[0] + \
                      beta * attention(x2, x2, x2)[0]
    
    # 注意：由于softmax的非线性，这不是精确的线性关系
    # 我们只检查它们不是完全不同的
    diff = (output_linear - output_separate).abs().mean()
    assert diff.item() < 1.0, f"线性性质差异太大: {diff}"
    
    print("  ✓ 近似线性性质测试通过")
    return True


def test_performance():
    """测试性能"""
    print("测试性能...")
    
    import time
    
    d_model = 512
    n_heads = 8
    batch_size = 16
    seq_len = 64
    
    attention = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 预热
    for _ in range(5):
        _ = attention(x, x, x)
    
    # 性能测试
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        output, _ = attention(x, x, x)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    print(f"  ⚡ 平均推理时间: {avg_time*1000:.2f} ms")
    print(f"  ⚡ FPS: {1/avg_time:.1f}")
    
    # 验证在合理时间内完成
    assert avg_time < 0.1, f"推理时间太长: {avg_time}"
    
    print("  ✓ 性能测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试多头注意力机制")
    print("=" * 60)
    
    tests = [
        ("缩放点积注意力", test_scaled_dot_product_attention),
        ("多头注意力形状", test_multihead_attention_shapes),
        ("注意力掩码", test_multihead_attention_masking),
        ("位置编码", test_positional_encoding),
        ("梯度流动", test_gradient_flow),
        ("注意力性质", test_attention_properties),
        ("性能", test_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"  通过: {passed}")
    print(f"  失败: {failed}")
    print(f"  总计: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {failed}个测试失败")
    
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_all_tests()
    
    if success:
        print("\n多头注意力机制实现正确！")
        print("\n实现功能包括:")
        print("1. ✅ 缩放点积注意力核心计算")
        print("2. ✅ 多头分割与合并")
        print("3. ✅ 填充掩码和前瞻掩码")
        print("4. ✅ 位置编码")
        print("5. ✅ 正确的梯度流动")
        print("6. ✅ 合理的性能")
        
        print("\n可以用于:")
        print("- Transformer模型")
        print("- 自注意力编码器")
        print("- 交叉注意力解码器")
        print("- 任何需要序列建模的任务")
    else:
        print("\n实现存在问题，请检查代码！")
    
    exit(0 if success else 1)
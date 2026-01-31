"""
CNN网络使用示例
"""

import torch
import torch.nn as nn
from main import SimpleCNN, AdvancedCNN

def example_usage():
    """使用示例"""
    print("CNN网络使用示例")
    print("=" * 50)
    
    # 1. 创建简单CNN模型
    print("\n1. 创建简单CNN模型:")
    simple_cnn = SimpleCNN(num_classes=10, input_channels=3)
    print(f"模型类型: {type(simple_cnn).__name__}")
    
    # 创建随机输入数据 (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32)
    print(f"输入形状: {dummy_input.shape}")
    
    # 前向传播
    output = simple_cnn(dummy_input)
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output[0][:5]}")  # 打印第一个样本的前5个输出
    
    # 2. 创建高级CNN模型
    print("\n2. 创建高级CNN模型（带残差连接）:")
    advanced_cnn = AdvancedCNN(num_classes=10, input_channels=3)
    print(f"模型类型: {type(advanced_cnn).__name__}")
    
    # 前向传播
    output = advanced_cnn(dummy_input)
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output[0][:5]}")
    
    # 3. 模型参数统计
    print("\n3. 模型参数统计:")
    
    for model_name, model in [("简单CNN", simple_cnn), ("高级CNN", advanced_cnn)]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{model_name}:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 各层参数统计
        print(f"  各层参数:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.numel():,}")
    
    # 4. 保存和加载模型
    print("\n4. 模型保存和加载示例:")
    
    # 保存模型
    torch.save(simple_cnn.state_dict(), 'simple_cnn_model.pth')
    print("  模型已保存到: simple_cnn_model.pth")
    
    # 加载模型
    loaded_model = SimpleCNN(num_classes=10, input_channels=3)
    loaded_model.load_state_dict(torch.load('simple_cnn_model.pth'))
    loaded_model.eval()
    print("  模型已加载")
    
    # 验证加载的模型
    with torch.no_grad():
        original_output = simple_cnn(dummy_input)
        loaded_output = loaded_model(dummy_input)
        
        # 检查输出是否相同
        if torch.allclose(original_output, loaded_output, rtol=1e-5):
            print("  ✓ 加载的模型输出与原始模型一致")
        else:
            print("  ✗ 加载的模型输出与原始模型不一致")
    
    # 5. 模型推理示例
    print("\n5. 模型推理示例:")
    
    # 设置模型为评估模式
    simple_cnn.eval()
    
    # 创建测试数据
    test_input = torch.randn(1, 3, 32, 32)  # 单个样本
    
    with torch.no_grad():
        predictions = simple_cnn(test_input)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"  输入形状: {test_input.shape}")
        print(f"  预测类别: {predicted_class.item()}")
        print(f"  类别概率: {probabilities[0][predicted_class].item():.4f}")
        
        # 显示所有类别的概率
        print(f"  所有类别概率:")
        for i, prob in enumerate(probabilities[0]):
            print(f"    类别 {i}: {prob.item():.4f}")
    
    print("\n" + "=" * 50)
    print("示例完成！")

def quick_demo():
    """快速演示"""
    print("快速演示：创建和使用CNN模型")
    
    # 创建模型
    model = SimpleCNN(num_classes=10, input_channels=3)
    
    # 创建随机输入
    input_tensor = torch.randn(2, 3, 32, 32)
    
    # 前向传播
    output = model(input_tensor)
    
    print(f"输入: {input_tensor.shape}")
    print(f"输出: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # 运行完整示例
    example_usage()
    
    print("\n\n快速演示:")
    quick_demo()
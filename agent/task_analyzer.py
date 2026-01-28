"""
Task Analyzer - 任务复杂度智能判断

根据任务描述快速判断任务是简单还是复杂，用于优化执行策略。
"""

from dataclasses import dataclass
from typing import Optional
from llm.base import BaseLLM, Message


@dataclass
class TaskComplexity:
    """任务复杂度评估结果"""
    is_simple: bool
    confidence: float  # 0-1，置信度
    reason: str  # 判断理由
    suggested_max_iterations: int  # 建议的最大迭代次数


class TaskAnalyzer:
    """
    任务复杂度分析器
    
    策略：
    1. 先用快速规则（token 长度、关键词）初步判断
    2. 如果不确定，再用 LLM 精确判断（可选）
    """
    
    # 简单任务的典型特征（关键词）
    SIMPLE_KEYWORDS = [
        "打印", "输出", "计算", "求和", "乘法表", "hello world",
        "斐波那契", "排序", "反转", "查找", "替换",
        "单个函数", "一个类", "简单脚本"
    ]
    
    # 复杂任务的典型特征（关键词）
    COMPLEX_KEYWORDS = [
        "系统", "平台", "架构", "前后端", "数据库", "API",
        "微服务", "部署", "多模块", "分布式", "爬虫",
        "web应用", "网站", "管理系统", "框架",
        # 深度学习相关
        "transformer", "bert", "gpt", "cnn", "rnn", "lstm",
        "神经网络", "模型", "训练", "深度学习", "机器学习",
        "attention", "embedding", "encoder", "decoder"
    ]
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        初始化分析器
        
        Args:
            llm: 可选的 LLM 实例，用于精确判断（如果不提供，只用规则判断）
        """
        self.llm = llm
    
    def analyze(self, task: str, use_llm: bool = False) -> TaskComplexity:
        """
        分析任务复杂度
        
        Args:
            task: 任务描述
            use_llm: 是否使用 LLM 进行精确判断（默认 False，只用规则）
        
        Returns:
            TaskComplexity 评估结果
        """
        # 步骤 1：快速规则判断
        rule_result = self._analyze_by_rules(task)
        
        # 如果规则判断很有信心，直接返回
        if rule_result.confidence >= 0.8:
            return rule_result
        
        # 步骤 2：如果不确定且提供了 LLM，用 LLM 精确判断
        if use_llm and self.llm:
            return self._analyze_by_llm(task, rule_result)
        
        # 否则返回规则判断结果
        return rule_result
    
    def _analyze_by_rules(self, task: str) -> TaskComplexity:
        """基于规则的快速判断"""
        task_lower = task.lower()
        
        # 规则 1：token 长度（简单粗暴但有效）
        token_count = len(task)
        
        if token_count > 200:
            # 任务描述超过 200 字符，很可能是复杂任务
            return TaskComplexity(
                is_simple=False,
                confidence=0.9,
                reason=f"任务描述过长（{token_count} 字符），判定为复杂任务",
                suggested_max_iterations=10
            )
        
        # 规则 2：关键词匹配
        simple_matches = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in task_lower)
        complex_matches = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in task_lower)
        
        if complex_matches > 0:
            # 包含复杂关键词
            return TaskComplexity(
                is_simple=False,
                confidence=0.85,
                reason=f"包含复杂任务关键词（匹配 {complex_matches} 个）",
                suggested_max_iterations=10
            )
        
        if simple_matches > 0 and token_count < 100:
            # 包含简单关键词且描述简短
            return TaskComplexity(
                is_simple=True,
                confidence=0.9,
                reason=f"包含简单任务关键词（匹配 {simple_matches} 个）且描述简短",
                suggested_max_iterations=5
            )
        
        # 规则 3：句子结构分析
        has_multiple_requirements = any(sep in task for sep in ["，并且", "，然后", "和", "以及", "包括"])
        
        if has_multiple_requirements and token_count > 80:
            return TaskComplexity(
                is_simple=False,
                confidence=0.7,
                reason="任务包含多个子需求",
                suggested_max_iterations=8
            )
        
        # 默认：短任务倾向于简单
        if token_count < 50:
            return TaskComplexity(
                is_simple=True,
                confidence=0.75,
                reason="任务描述简短",
                suggested_max_iterations=5
            )
        
        # 不确定的情况：中等复杂度
        return TaskComplexity(
            is_simple=False,
            confidence=0.6,
            reason="无法明确判断，按中等复杂度处理",
            suggested_max_iterations=5
        )
    
    def _analyze_by_llm(self, task: str, rule_result: TaskComplexity) -> TaskComplexity:
        """使用 LLM 进行精确判断"""
        if not self.llm:
            return rule_result
        
        prompt = f"""请判断以下编程任务是"简单"还是"复杂"。

简单任务示例：
- 写一个计算两数之和的函数
- 打印九九乘法表
- 反转一个字符串
- 实现斐波那契数列

复杂任务示例：
- 开发一个用户管理系统（前后端）
- 实现一个 Web 爬虫框架
- 创建一个微服务架构
- 搭建数据分析平台

任务：{task}

请只回复：
1. 判断结果（简单 / 复杂）
2. 理由（一句话）

格式：判断结果｜理由"""
        
        try:
            response = self.llm.chat_sync(
                [Message.user(prompt)],
                temperature=0.3,
            )
            
            content = response.content.strip()
            
            # 解析回复
            if "|" in content or "｜" in content:
                parts = content.replace("｜", "|").split("|")
                result_str = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else "LLM 判断"
                
                is_simple = "简单" in result_str
                
                return TaskComplexity(
                    is_simple=is_simple,
                    confidence=0.95,
                    reason=f"LLM 判断：{reason}",
                    suggested_max_iterations=5 if is_simple else 10
                )
        except Exception:
            pass
        
        # LLM 判断失败，返回规则判断结果
        return rule_result

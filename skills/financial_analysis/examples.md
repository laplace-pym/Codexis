# 金融分析工具使用示例

## 场景 1：选股分析

### 问题
"我在考虑投资苹果公司，帮我分析一下是否值得买入。"

### Claude 会调用
```json
{
  "tool": "analyze_stock",
  "parameters": {
    "ticker": "AAPL",
    "analysis_type": "comprehensive"
  }
}
```

### 预期输出
```
AAPL 综合分析报告：

基本面：
- P/E 比率：25.3（估值合理）
- ROE：28.5%（盈利能力强）
- 债务权益比：1.57（财务杠杆适中）
- 利润率：23.4%（盈利能力优秀）

技术面：
- 趋势：上涨
- RSI：62.5（不过热）
- MACD：看涨交叉

投资建议：
AAPL 目前估值合理，基本面稳健，技术面呈现上涨趋势，建议适度配置。
```

---

## 场景 2：公司财务分析

### 问题
"这家公司去年营收 100亿，净利润 20亿，总资产 500亿，股东权益 200亿，帮我看看财务状况怎么样。"

### Claude 会调用
```json
{
  "tool": "calculate_financial_ratios",
  "parameters": {
    "revenue": 10000000000,
    "net_income": 2000000000,
    "total_assets": 50000000000,
    "total_equity": 20000000000
  }
}
```

### 预期输出
```
财务比率分析：
- 利润率：20%（优秀）
- ROE：10%（一般）
- ROA：4%（一般）
- 债务权益比：1.5（适中）
- 资产周转率：0.2（较低）

综合评估：
✓ 盈利能力：优秀（利润率 20%）
✓ 财务杠杆：稳健（债务比 < 2）
⚠ 运营效率：有待提升（资产周转率偏低）

建议：公司盈利能力不错，但资产使用效率有提升空间。
```

---

## 场景 3：投资组合优化

### 问题
"我的投资组合包括：苹果 30%（年收益15%）、特斯拉 20%（年收益25%）、微软 30%（年收益12%）、谷歌 20%（年收益18%），帮我评估一下。"

### Claude 会调用
```json
{
  "tool": "portfolio_analysis",
  "parameters": {
    "holdings": [
      {"ticker": "AAPL", "weight": 0.3, "return": 0.15},
      {"ticker": "TSLA", "weight": 0.2, "return": 0.25},
      {"ticker": "MSFT", "weight": 0.3, "return": 0.12},
      {"ticker": "GOOGL", "weight": 0.2, "return": 0.18}
    ]
  }
}
```

### 预期输出
```
投资组合分析报告：

基本信息：
- 持仓数量：4 只股票
- 加权平均收益率：16.1%

评估：
✓ 多样化：良好（4 只股票，科技行业集中）
✓ 收益表现：优秀（年化 16.1%）
⚠ 行业集中度：较高（全部为科技股）

建议：
组合收益不错，但过度集中在科技行业。建议增加其他行业配置以降低系统性风险。
可以考虑配置 10-15% 的防御性资产（如消费、医疗）。
```

---

## 场景 4：风险管理

### 问题
"我想知道这个投资策略的风险有多大，过去 12 个月的收益率是：5%, -2%, 8%, 3%, -5%, 10%, 4%, -3%, 7%, 2%, -4%, 6%"

### Claude 会调用
```json
{
  "tool": "risk_assessment",
  "parameters": {
    "returns": [0.05, -0.02, 0.08, 0.03, -0.05, 0.10, 0.04, -0.03, 0.07, 0.02, -0.04, 0.06],
    "confidence_level": 0.95
  }
}
```

### 预期输出
```
风险评估报告：

关键指标：
- 平均月收益率：2.58%
- 波动率（年化）：18.2%
- VaR (95% 置信)：-4.5%
- 风险等级：中等

解读：
该策略的波动性适中（年化波动率 18.2%）。在 95% 的情况下，单月最大损失不会超过 4.5%。

建议：
- 适合风险承受能力中等的投资者
- 建议保持 10-20% 的现金储备
- 设置 -5% 的止损线
- 定期再平衡（每季度一次）
```

---

## 场景 5：综合投资决策

### 问题
"我有 50 万预算想投资科技股，请帮我：
1. 分析 AAPL、MSFT、NVDA、GOOGL 哪些值得投资
2. 设计一个投资组合
3. 评估整体风险"

### Claude 会连续调用多个工具

#### Step 1：分析各股票
```json
[
  {"tool": "analyze_stock", "parameters": {"ticker": "AAPL"}},
  {"tool": "analyze_stock", "parameters": {"ticker": "MSFT"}},
  {"tool": "analyze_stock", "parameters": {"ticker": "NVDA"}},
  {"tool": "analyze_stock", "parameters": {"ticker": "GOOGL"}}
]
```

#### Step 2：设计投资组合
```json
{
  "tool": "portfolio_analysis",
  "parameters": {
    "holdings": [
      {"ticker": "AAPL", "weight": 0.30, "return": 0.15},
      {"ticker": "MSFT", "weight": 0.35, "return": 0.12},
      {"ticker": "NVDA", "weight": 0.20, "return": 0.30},
      {"ticker": "GOOGL", "weight": 0.15, "return": 0.18}
    ]
  }
}
```

#### Step 3：风险评估
```json
{
  "tool": "risk_assessment",
  "parameters": {
    "returns": [...],
    "confidence_level": 0.95
  }
}
```

### 预期输出
```
综合投资方案：

1️⃣ 股票分析总结：
- AAPL：★★★★☆ 估值合理，业绩稳定，推荐
- MSFT：★★★★★ 基本面优秀，云业务增长强劲，强烈推荐
- NVDA：★★★★☆ AI 热潮受益，但估值偏高，谨慎配置
- GOOGL：★★★★☆ 搜索和云双驱动，值得配置

2️⃣ 推荐投资组合：
┌──────────┬────────┬─────────┬──────────┐
│ 股票     │ 权重   │ 金额    │ 预期收益 │
├──────────┼────────┼─────────┼──────────┤
│ MSFT     │ 35%    │ 17.5万  │ 12%      │
│ AAPL     │ 30%    │ 15.0万  │ 15%      │
│ NVDA     │ 20%    │ 10.0万  │ 30%      │
│ GOOGL    │ 15%    │  7.5万  │ 18%      │
└──────────┴────────┴─────────┴──────────┘

- 预期年化收益：16.8%
- 多样化评分：良好

3️⃣ 风险评估：
- 整体风险等级：中等
- 年化波动率：22%
- VaR (95%)：-6%
- 最大预期回撤：12-15%

4️⃣ 操作建议：
✓ 分批建仓（3-4 周完成）
✓ 设置止盈/止损：+30% / -8%
✓ 季度再平衡
✓ 保留 10% 现金应对机会

⚠️ 风险提示：
- 组合集中于科技行业，受行业波动影响大
- NVDA 估值较高，注意控制仓位
- 建议关注美联储货币政策变化
```

---

## 日志中的工具调用轨迹

在 `~/.claude-reverse-logs/` 的日志文件中，你会看到完整的调用过程：

```json
{
  "type": "request",
  "body": {
    "messages": [...],
    "tools": [
      {
        "name": "analyze_stock",
        "description": "分析股票的基本面和技术指标...",
        "input_schema": {...}
      },
      ...
    ]
  }
}

{
  "type": "response",
  "body": {
    "content": [
      {
        "type": "tool_use",
        "name": "analyze_stock",
        "input": {
          "ticker": "AAPL",
          "analysis_type": "comprehensive"
        }
      }
    ]
  }
}

{
  "type": "request",
  "body": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "tool_result",
            "tool_use_id": "...",
            "content": "{\"analysis\": ...}"
          }
        ]
      }
    ]
  }
}
```

这样你就能清晰地看到 Claude Code 是如何调用你的金融分析 skill 的！

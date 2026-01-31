#!/usr/bin/env python3
"""
金融分析 MCP Server
为 Claude Code 提供金融分析工具
"""

import json
import asyncio
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent


# 创建 MCP server
app = Server("financial-analysis")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的金融分析工具"""
    return [
        Tool(
            name="analyze_stock",
            description="分析股票的基本面和技术指标。输入股票代码，返回详细的分析报告。",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "股票代码，例如：AAPL, TSLA, BABA"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["fundamental", "technical", "comprehensive"],
                        "description": "分析类型：基本面、技术面或综合分析"
                    }
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="calculate_financial_ratios",
            description="计算关键财务比率，包括 P/E、P/B、ROE、债务比率等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "revenue": {
                        "type": "number",
                        "description": "营业收入"
                    },
                    "net_income": {
                        "type": "number",
                        "description": "净利润"
                    },
                    "total_assets": {
                        "type": "number",
                        "description": "总资产"
                    },
                    "total_equity": {
                        "type": "number",
                        "description": "股东权益"
                    },
                    "total_liabilities": {
                        "type": "number",
                        "description": "总负债"
                    },
                    "market_cap": {
                        "type": "number",
                        "description": "市值"
                    }
                },
                "required": ["revenue", "net_income", "total_assets", "total_equity"]
            }
        ),
        Tool(
            name="portfolio_analysis",
            description="分析投资组合的风险收益特征，包括多样化程度、夏普比率、最大回撤等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "weight": {"type": "number"},
                                "return": {"type": "number"}
                            }
                        },
                        "description": "投资组合持仓列表"
                    }
                },
                "required": ["holdings"]
            }
        ),
        Tool(
            name="risk_assessment",
            description="评估投资风险，包括波动率、VaR（风险价值）、Beta 系数等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "历史收益率序列"
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "置信水平（0-1），用于计算 VaR"
                    }
                },
                "required": ["returns"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """执行工具调用"""
    
    if name == "analyze_stock":
        ticker = arguments["ticker"]
        analysis_type = arguments.get("analysis_type", "comprehensive")
        
        # 模拟股票分析
        result = {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "analysis": {
                "fundamental": {
                    "pe_ratio": 25.3,
                    "pb_ratio": 8.2,
                    "roe": 0.285,
                    "debt_to_equity": 1.57,
                    "profit_margin": 0.234
                },
                "technical": {
                    "trend": "上涨趋势",
                    "support_level": 150.0,
                    "resistance_level": 180.0,
                    "rsi": 62.5,
                    "macd": "看涨交叉"
                },
                "recommendation": f"{ticker} 目前估值合理，技术面呈现上涨趋势，建议适度配置。"
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "calculate_financial_ratios":
        revenue = arguments["revenue"]
        net_income = arguments["net_income"]
        total_assets = arguments["total_assets"]
        total_equity = arguments["total_equity"]
        total_liabilities = arguments.get("total_liabilities", total_assets - total_equity)
        market_cap = arguments.get("market_cap", 0)
        
        # 计算财务比率
        ratios = {
            "profit_margin": net_income / revenue if revenue > 0 else 0,
            "roe": net_income / total_equity if total_equity > 0 else 0,
            "roa": net_income / total_assets if total_assets > 0 else 0,
            "debt_to_equity": total_liabilities / total_equity if total_equity > 0 else 0,
            "asset_turnover": revenue / total_assets if total_assets > 0 else 0
        }
        
        if market_cap > 0:
            ratios["pe_ratio"] = market_cap / net_income if net_income > 0 else 0
            ratios["pb_ratio"] = market_cap / total_equity if total_equity > 0 else 0
        
        result = {
            "财务比率分析": ratios,
            "评估": {
                "盈利能力": "优秀" if ratios["roe"] > 0.15 else "一般",
                "财务杠杆": "稳健" if ratios["debt_to_equity"] < 2 else "较高",
                "运营效率": "良好" if ratios["asset_turnover"] > 0.5 else "有待提升"
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "portfolio_analysis":
        holdings = arguments["holdings"]
        
        # 计算组合指标
        total_weight = sum(h["weight"] for h in holdings)
        weighted_return = sum(h["weight"] * h["return"] for h in holdings) / total_weight
        
        result = {
            "投资组合分析": {
                "持仓数量": len(holdings),
                "加权平均收益率": f"{weighted_return:.2%}",
                "多样化评分": "良好" if len(holdings) >= 5 else "需要改善",
                "风险水平": "中等",
                "建议": "组合配置合理，建议定期再平衡以维持目标权重。"
            },
            "详细持仓": holdings
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "risk_assessment":
        returns = arguments["returns"]
        confidence_level = arguments.get("confidence_level", 0.95)
        
        # 计算风险指标
        import statistics
        mean_return = statistics.mean(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]
        
        result = {
            "风险评估": {
                "平均收益率": f"{mean_return:.2%}",
                "波动率（标准差）": f"{volatility:.2%}",
                f"VaR ({confidence_level:.0%} 置信水平)": f"{var:.2%}",
                "风险等级": "高" if volatility > 0.3 else "中" if volatility > 0.15 else "低",
                "建议": "根据风险承受能力调整仓位，高波动性资产应控制比例。"
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """启动 MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

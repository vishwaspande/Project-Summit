#!/usr/bin/env python3
"""
Enhanced AI-Powered Indian Stock Market Agent with Anthropic Tool Use API

This is an enhanced version of the original agent that includes:
- Proper agent reasoning loop using Anthropic's tool use API
- Autonomous decision making and action taking
- Multi-step reasoning with tool chaining
- Memory system integration
- Notification system integration

Features for Indian markets:
- NSE/BSE stock analysis with real-time monitoring
- Indian market hours (9:15 AM - 3:30 PM IST)
- Rupee-denominated analysis with currency impact
- Indian sector analysis (IT, Banking, Pharma, Auto, Infrastructure)
- Autonomous scheduling for morning and evening runs
- Telegram notifications for alerts and briefings
"""

import anthropic
import json
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import our custom modules
from config import AgentConfig, load_config, setup_logging
from memory import MemoryManager
from notifications import NotificationManager
from tools import StockMarketTools, ToolResult

warnings.filterwarnings('ignore')

@dataclass
class AgentThought:
    """Structure for agent's reasoning process."""
    step: int
    thought: str
    action: str
    tool_name: str = ""
    tool_args: Dict[str, Any] = None
    tool_result: Any = None
    observation: str = ""
    next_action: str = ""
    confidence: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
        if self.tool_args is None:
            self.tool_args = {}

@dataclass
class AgentResponse:
    """Complete agent response with reasoning trace."""
    task: str
    final_answer: str
    thoughts: List[AgentThought]
    tools_used: List[str]
    total_steps: int
    success: bool
    error: str = ""
    execution_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

class EnhancedIndianStockMarketAgent:
    """
    Enhanced AI agent with autonomous reasoning and tool use capabilities.

    This agent can:
    - Receive tasks and break them down into steps
    - Use available tools to gather data
    - Reason through complex multi-step problems
    - Make autonomous decisions
    - Take actions based on analysis
    - Learn from past decisions stored in memory
    """

    def __init__(self, config: AgentConfig = None):
        """Initialize the enhanced agent."""

        # Load configuration
        self.config = config or load_config()

        # Setup logging
        self.logger = setup_logging(self.config)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.config.anthropic_api_key
        )

        # Initialize core components
        self.memory = MemoryManager(self.config.database_path)
        self.notifications = NotificationManager(self.config)
        self.tools = StockMarketTools(self.config, self.memory, self.notifications)

        # Agent state
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.current_task = ""
        self.reasoning_trace = []
        self.max_iterations = 15  # Prevent infinite loops
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Tool registry
        self.available_tools = self._build_tool_registry()

        self.logger.info("Enhanced Indian Stock Market Agent initialized successfully")
        self.logger.info(f"Available tools: {', '.join(self.available_tools.keys())}")

    def _build_tool_registry(self) -> Dict[str, Any]:
        """Build the tool registry for the agent."""
        return {
            'get_stock_price': self.tools.get_stock_price,
            'get_multiple_stocks': self.tools.get_multiple_stocks,
            'get_stock_fundamentals': self.tools.get_stock_fundamentals,
            'get_technical_indicators': self.tools.get_technical_indicators,
            'get_market_news': self.tools.get_market_news,
            'get_fii_dii_data': self.tools.get_fii_dii_data,
            'get_usd_inr_rate': self.tools.get_usd_inr_rate,
            'check_portfolio': self.tools.check_portfolio,
            'send_alert': self.tools.send_alert,
            'save_to_memory': self.tools.save_to_memory,
            'read_from_memory': self.tools.read_from_memory
        }

    async def run_agent_loop(self, task: str, max_iterations: int = None) -> AgentResponse:
        """
        Main agent reasoning loop using Anthropic's tool use API.

        The agent will:
        1. Understand the task
        2. Plan the approach
        3. Execute tools in sequence
        4. Reason through results
        5. Make decisions
        6. Take actions
        7. Provide final analysis

        Args:
            task (str): The task for the agent to complete
            max_iterations (int): Maximum number of reasoning steps

        Returns:
            AgentResponse: Complete response with reasoning trace
        """
        start_time = datetime.now()
        self.current_task = task
        self.reasoning_trace = []
        max_iter = max_iterations or self.max_iterations

        try:
            self.logger.info(f"🤖 Starting agent loop for task: {task}")

            # Initial system prompt with Indian market context
            system_prompt = self._build_system_prompt()

            # Initialize conversation
            messages = [
                {
                    "role": "user",
                    "content": f"""
                    Task: {task}

                    Current Indian market context:
                    - Time: {datetime.now(self.indian_timezone).strftime('%Y-%m-%d %H:%M:%S IST')}
                    - Market Status: {'OPEN' if self._is_market_open() else 'CLOSED'}
                    - Your available tools: {', '.join(self.available_tools.keys())}

                    Please analyze this task step by step, use the appropriate tools to gather data,
                    reason through the information, and provide actionable insights for Indian investors.

                    Think through this systematically and use multiple tools if needed to provide
                    comprehensive analysis.
                    """
                }
            ]

            iteration = 0
            final_answer = ""
            tools_used = []

            while iteration < max_iter:
                iteration += 1

                try:
                    # Call Claude with tool definitions
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        system=system_prompt,
                        messages=messages,
                        tools=self.tools.get_tool_definitions()
                    )

                    # Process response
                    if response.stop_reason == "tool_use":
                        # Agent wants to use tools
                        assistant_message = {
                            "role": "assistant",
                            "content": response.content
                        }
                        messages.append(assistant_message)

                        # Extract tool calls and execute them
                        tool_results = []
                        for content_block in response.content:
                            if content_block.type == "tool_use":
                                tool_name = content_block.name
                                tool_args = content_block.input
                                tool_id = content_block.id

                                self.logger.info(f"🔧 Using tool: {tool_name} with args: {tool_args}")

                                # Execute tool
                                tool_result = await self._execute_tool(tool_name, tool_args)
                                tools_used.append(tool_name)

                                # Record reasoning step
                                thought = AgentThought(
                                    step=iteration,
                                    thought=f"Using {tool_name} to gather data",
                                    action=f"Call {tool_name}",
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_result=tool_result,
                                    observation=str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                                )
                                self.reasoning_trace.append(thought)

                                # Prepare tool result for Claude
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": json.dumps(tool_result, default=str)
                                })

                        # Add tool results to conversation
                        if tool_results:
                            user_message = {
                                "role": "user",
                                "content": tool_results
                            }
                            messages.append(user_message)

                    elif response.stop_reason in ["end_turn", "max_tokens"]:
                        # Agent has provided final answer
                        final_answer = ""
                        for content_block in response.content:
                            if content_block.type == "text":
                                final_answer += content_block.text

                        self.logger.info(f"✅ Agent completed task after {iteration} iterations")
                        break

                    else:
                        # Handle other stop reasons
                        self.logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                        break

                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}")
                    break

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create response object
            agent_response = AgentResponse(
                task=task,
                final_answer=final_answer,
                thoughts=self.reasoning_trace,
                tools_used=list(set(tools_used)),
                total_steps=iteration,
                success=bool(final_answer),
                execution_time=execution_time
            )

            # Store the agent's decision in memory
            self.memory.store_agent_decision(
                decision_type="task_completion",
                context=task,
                reasoning=f"Completed task in {iteration} steps using tools: {', '.join(tools_used)}",
                action_taken=f"Generated analysis with {len(final_answer)} characters",
                outcome="Success" if final_answer else "Failed",
                confidence=0.8 if final_answer else 0.3
            )

            self.logger.info(f"🎯 Task completed. Success: {agent_response.success}, Time: {execution_time:.2f}s")

            return agent_response

        except Exception as e:
            self.logger.error(f"Error in agent loop: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResponse(
                task=task,
                final_answer="",
                thoughts=self.reasoning_trace,
                tools_used=list(set(tools_used)) if 'tools_used' in locals() else [],
                total_steps=iteration if 'iteration' in locals() else 0,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool asynchronously."""
        try:
            tool_function = self.available_tools.get(tool_name)
            if not tool_function:
                return {"error": f"Tool {tool_name} not found"}

            # Execute tool (most tools are sync, but we'll handle async ones too)
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(**tool_args)
            else:
                # Run in executor to prevent blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: tool_function(**tool_args)
                )

            # Convert ToolResult to dict if needed
            if isinstance(result, ToolResult):
                return {
                    'success': result.success,
                    'data': result.data,
                    'error': result.error,
                    'source': result.source,
                    'timestamp': result.timestamp.isoformat() if result.timestamp else None
                }

            return result

        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}

    def _build_system_prompt(self) -> str:
        """Build the system prompt with Indian market context."""
        return f"""
        You are an expert Indian Stock Market AI Agent with deep knowledge of NSE/BSE markets,
        Indian economy, and investment strategies suitable for Indian retail and institutional investors.

        CURRENT CONTEXT:
        - Time: {datetime.now(self.indian_timezone).strftime('%Y-%m-%d %H:%M:%S IST')}
        - Market Status: {'OPEN (9:15 AM - 3:30 PM IST)' if self._is_market_open() else 'CLOSED'}
        - Currency: All analysis in Indian Rupees (₹)
        - Focus: Indian market conditions, regulations, and investor needs

        YOUR CAPABILITIES:
        1. Analyze Indian stocks, ETFs, and market indices
        2. Provide fundamental and technical analysis
        3. Assess market sentiment and news impact
        4. Monitor portfolio performance and risk
        5. Generate actionable investment recommendations
        6. Send alerts and notifications to investors
        7. Store and retrieve analysis from memory

        ANALYSIS FRAMEWORK:
        - Always consider Indian market hours and holidays
        - Factor in currency impact (USD/INR) on different sectors
        - Consider FII/DII flows and their market impact
        - Provide risk assessment suitable for Indian investors
        - Give recommendations in Indian investment context

        REASONING APPROACH:
        1. Break down complex tasks into steps
        2. Use appropriate tools to gather comprehensive data
        3. Analyze data with Indian market perspective
        4. Consider multiple scenarios and risk factors
        5. Provide clear, actionable recommendations
        6. Store important decisions and learnings

        SAFETY GUIDELINES:
        - Never execute actual trades (only recommend)
        - Always mention risks and disclaimers
        - Consider investor's risk tolerance and goals
        - Provide balanced analysis (not just bullish/bearish)
        - Store analysis for learning and improvement

        Use the available tools systematically to provide comprehensive,
        well-reasoned analysis that helps Indian investors make informed decisions.
        """

    def _is_market_open(self) -> bool:
        """Check if Indian stock market is currently open."""
        now = datetime.now(self.indian_timezone)
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()  # 0=Monday, 6=Sunday

        # Market closed on weekends
        if current_day >= 5:  # Saturday or Sunday
            return False

        # Check if current time is within market hours (9:15 AM - 3:30 PM IST)
        if "09:15" <= current_time <= "15:30":
            return True

        return False

    async def run_analysis(self, analysis_request: str) -> str:
        """
        Run a single analysis request through the agent loop.

        Args:
            analysis_request (str): The analysis request

        Returns:
            str: Analysis result
        """
        response = await self.run_agent_loop(analysis_request)
        return response.final_answer if response.success else f"Analysis failed: {response.error}"

    async def autonomous_morning_analysis(self) -> AgentResponse:
        """
        Run autonomous morning analysis before market opens.

        This will:
        1. Check overnight global developments
        2. Analyze watchlist stocks
        3. Review FII/DII flows
        4. Generate morning briefing
        5. Send notifications
        """
        task = f"""
        Conduct comprehensive morning analysis for Indian markets before opening:

        1. Check overnight global market impact and USD/INR movement
        2. Analyze key watchlist stocks: {', '.join(self.config.default_watchlist)}
        3. Review FII/DII flow data and its implications
        4. Get latest market news and sentiment
        5. Identify top stocks to watch today
        6. Assess sector-wise outlook
        7. Generate actionable morning briefing for investors
        8. Send appropriate alerts for significant overnight moves

        Focus on pre-market preparation and day trading opportunities.
        Consider both technical setups and fundamental factors.
        """

        return await self.run_agent_loop(task)

    async def autonomous_evening_analysis(self) -> AgentResponse:
        """
        Run autonomous evening analysis after market closes.

        This will:
        1. Review today's market performance
        2. Analyze portfolio performance
        3. Check global market setup
        4. Generate evening briefing
        5. Plan for tomorrow
        """
        task = f"""
        Conduct comprehensive evening analysis for Indian markets after closing:

        1. Review today's market performance (indices, sectors, individual stocks)
        2. Check portfolio performance using default portfolio
        3. Analyze day's FII/DII activity and its impact
        4. Review significant news and events that moved markets
        5. Assess global market setup for tomorrow (US pre-market, currency, commodities)
        6. Identify tomorrow's key levels and events to watch
        7. Generate evening briefing with tomorrow's strategy
        8. Send portfolio performance update if material changes

        Focus on performance review and next-day preparation.
        Provide both tactical and strategic insights.
        """

        return await self.run_agent_loop(task)

    async def analyze_stock_comprehensive(self, symbol: str) -> AgentResponse:
        """
        Run comprehensive stock analysis using the agent loop.

        Args:
            symbol (str): Stock symbol to analyze

        Returns:
            AgentResponse: Complete analysis response
        """
        task = f"""
        Conduct comprehensive analysis of {symbol} stock:

        1. Get current price, volume, and basic metrics
        2. Analyze fundamental metrics (P/E, market cap, growth rates)
        3. Perform technical analysis (moving averages, RSI, MACD)
        4. Review recent news and market sentiment
        5. Compare with sector peers and benchmarks
        6. Assess FII/DII interest if material
        7. Consider currency impact if relevant
        8. Provide investment recommendation with target price
        9. Highlight key risks and catalysts
        10. Store analysis for future reference

        Provide actionable insights suitable for Indian investors.
        Include both short-term and long-term perspectives.
        """

        return await self.run_agent_loop(task)

    async def portfolio_health_check(self, portfolio: Dict[str, Dict[str, float]] = None) -> AgentResponse:
        """
        Run comprehensive portfolio analysis.

        Args:
            portfolio (Dict): Portfolio to analyze (uses default if not provided)

        Returns:
            AgentResponse: Portfolio analysis response
        """
        portfolio_to_use = portfolio or self.config.default_portfolio

        task = f"""
        Conduct comprehensive portfolio health check:

        1. Analyze current portfolio performance and P&L
        2. Check individual stock performance and contribution
        3. Assess portfolio risk and diversification
        4. Compare portfolio performance with Nifty/Sensex
        5. Identify overweight/underweight sectors
        6. Review correlation and concentration risks
        7. Get latest news for portfolio stocks
        8. Suggest rebalancing if needed
        9. Highlight stocks requiring attention
        10. Store portfolio snapshot for tracking

        Provide specific actionable recommendations for portfolio optimization.
        Consider current market conditions and outlook.

        Portfolio to analyze: {json.dumps(portfolio_to_use, indent=2)}
        """

        return await self.run_agent_loop(task)

    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """Get the current reasoning trace as a list of dictionaries."""
        return [asdict(thought) for thought in self.reasoning_trace]

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics and performance metrics."""
        try:
            # Get recent decisions from memory
            recent_decisions = self.memory.get_agent_decisions(days_back=30)

            # Calculate success rate
            total_decisions = len(recent_decisions)
            successful_decisions = sum(1 for d in recent_decisions if d.get('outcome', '').lower().startswith('success'))
            success_rate = (successful_decisions / total_decisions * 100) if total_decisions > 0 else 0

            # Get database stats
            db_stats = self.memory.get_database_stats()

            return {
                'agent_version': '2.0 Enhanced',
                'total_decisions': total_decisions,
                'success_rate': round(success_rate, 1),
                'tools_available': len(self.available_tools),
                'notifications_enabled': self.notifications.enabled,
                'database_records': db_stats,
                'current_task': self.current_task,
                'reasoning_steps': len(self.reasoning_trace),
                'last_updated': datetime.now(self.indian_timezone).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting agent stats: {e}")
            return {'error': str(e)}

# Async main function for running the agent
async def main():
    """Demo of the Enhanced Indian Stock Market Agent."""

    print("🇮🇳 Enhanced Indian Stock Market AI Agent Demo")
    print("=" * 60)

    try:
        # Load configuration
        config = load_config()

        if not config.anthropic_api_key:
            print("❌ API key not found!")
            print("Please set ANTHROPIC_API_KEY environment variable")
            return

        # Initialize enhanced agent
        agent = EnhancedIndianStockMarketAgent(config)

        print(f"\n📊 Agent Status:")
        print(f"⏰ Market Status: {'OPEN' if agent._is_market_open() else 'CLOSED'}")
        print(f"🔧 Available Tools: {len(agent.available_tools)}")
        print(f"📱 Notifications: {'✅ Enabled' if agent.notifications.enabled else '❌ Disabled'}")

        # Example 1: Stock Analysis
        print(f"\n{'='*50}")
        print("1. COMPREHENSIVE STOCK ANALYSIS")
        print("="*50)

        response = await agent.analyze_stock_comprehensive('RELIANCE')
        print(f"✅ Analysis completed in {response.execution_time:.1f}s")
        print(f"🔧 Tools used: {', '.join(response.tools_used)}")
        print(f"📝 Steps: {response.total_steps}")
        print(f"\n📊 Analysis:\n{response.final_answer[:800]}...")

        # Example 2: Portfolio Health Check
        print(f"\n{'='*50}")
        print("2. PORTFOLIO HEALTH CHECK")
        print("="*50)

        response = await agent.portfolio_health_check()
        print(f"✅ Portfolio analysis completed in {response.execution_time:.1f}s")
        print(f"🔧 Tools used: {', '.join(response.tools_used)}")
        print(f"📝 Steps: {response.total_steps}")
        print(f"\n📊 Portfolio Analysis:\n{response.final_answer[:800]}...")

        # Example 3: Morning Analysis (if before market hours)
        if not agent._is_market_open():
            print(f"\n{'='*50}")
            print("3. AUTONOMOUS MORNING ANALYSIS")
            print("="*50)

            response = await agent.autonomous_morning_analysis()
            print(f"✅ Morning analysis completed in {response.execution_time:.1f}s")
            print(f"🔧 Tools used: {', '.join(response.tools_used)}")
            print(f"📝 Steps: {response.total_steps}")
            print(f"\n🌅 Morning Briefing:\n{response.final_answer[:800]}...")

        # Show agent statistics
        print(f"\n{'='*50}")
        print("4. AGENT STATISTICS")
        print("="*50)

        stats = agent.get_agent_stats()
        for key, value in stats.items():
            if key != 'database_records':
                print(f"{key.replace('_', ' ').title()}: {value}")

        print("\n🚀 Enhanced Agent Demo Complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
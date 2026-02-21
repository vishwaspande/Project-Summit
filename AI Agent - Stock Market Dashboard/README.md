# 🇮🇳 Indian Stock Market AI Agent Dashboard

A sophisticated, autonomous AI agent for Indian stock market analysis and investment decision-making. Built with Anthropic's Claude API and featuring advanced tool use, autonomous scheduling, and real-time notifications.

## 🎯 Key Features

### 🤖 Autonomous AI Agent
- **Agent Reasoning Loop**: Multi-step reasoning using Anthropic's tool use API
- **Tool Chaining**: Agent can use multiple tools in sequence to solve complex problems
- **Autonomous Decision Making**: Agent reasons through data and makes investment recommendations
- **Explainable AI**: Complete reasoning trace for all decisions

### 📊 Indian Market Focus
- **NSE/BSE Integration**: Real-time data for Indian stock exchanges
- **Currency Aware**: All analysis in INR with USD/INR impact assessment
- **Market Hours**: Respects Indian market timing (9:15 AM - 3:30 PM IST)
- **Sector Analysis**: Deep understanding of Indian sectors (IT, Banking, Pharma, etc.)
- **FII/DII Flows**: Foreign and domestic institutional investor flow analysis

### 🔧 Agent Tools
- `get_stock_price` - Live NSE/BSE stock prices
- `get_multiple_stocks` - Batch stock data retrieval
- `get_stock_fundamentals` - P/E, market cap, financials
- `get_technical_indicators` - SMA, RSI, MACD analysis
- `get_market_news` - Latest market news and sentiment
- `get_fii_dii_data` - Institutional flow data
- `get_usd_inr_rate` - Currency rates and impact
- `check_portfolio` - Portfolio performance analysis
- `send_alert` - Telegram notifications
- `save_to_memory` / `read_from_memory` - Persistent learning

### 🕐 Autonomous Scheduling
- **Morning Briefing** (8:30 AM IST): Pre-market analysis and watchlist screening
- **Evening Analysis** (8:30 PM IST): Market review and global outlook
- **Market Monitoring**: Continuous monitoring during market hours
- **Custom Scheduling**: One-time and recurring analysis jobs

### 📱 Telegram Integration
- Morning and evening briefings
- Real-time price alerts
- Portfolio performance updates
- Trade recommendations
- Custom notifications

### 🗃️ Memory System
- **SQLite Database**: Persistent storage for all analysis and decisions
- **Learning Capability**: Agent learns from past decisions and outcomes
- **Historical Tracking**: Portfolio snapshots and performance over time
- **Explainability**: Complete audit trail of agent decisions

### 📈 Enhanced Dashboard
- **Market Overview**: Nifty, Sensex, USD/INR, key metrics
- **Stock Analysis**: Quick and AI-powered comprehensive analysis
- **Portfolio Management**: Multi-asset portfolio tracking and optimization
- **Agent Activity**: Monitor agent decisions, alerts, and tool usage
- **Agent Settings**: Configure watchlist, risk tolerance, scheduling
- **AI Assistant**: Interactive chat with multi-step reasoning

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd "AI Agent - Stock Market Dashboard"

# Install core dependencies (minimal)
pip install anthropic yfinance pandas numpy pytz requests streamlit plotly SQLAlchemy APScheduler aiohttp python-telegram-bot

# Or install all dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set up environment variables:

```bash
# Required
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional (for Telegram notifications)
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# Optional (customize behavior)
export RISK_TOLERANCE="medium"  # low, medium, high
export ALERT_THRESHOLD="3.0"    # percentage move to trigger alert
export DEFAULT_WATCHLIST="RELIANCE,TCS,HDFCBANK,INFY,ITC"
export MORNING_RUN_TIME="08:30"
export EVENING_RUN_TIME="20:30"
```

### 3. Running the Application

#### Option A: Streamlit Dashboard (Recommended)
```bash
streamlit run indian_dashboard.py
```

#### Option B: Agent CLI Demo
```bash
python indian_stock_market_agent.py
```

#### Option C: Autonomous Scheduler
```bash
python scheduler.py
```

### 4. Setting up Telegram Notifications (Optional)

1. Create a bot with [@BotFather](https://t.me/botfather) on Telegram
2. Get your bot token
3. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
4. Set environment variables and restart

## 📁 Project Structure

```
AI Agent - Stock Market Dashboard/
├── indian_stock_market_agent.py   # Enhanced agent with tool use
├── indian_dashboard.py            # Streamlit dashboard
├── tools.py                       # Agent tool definitions
├── scheduler.py                   # Autonomous scheduling
├── memory.py                      # SQLite memory system
├── notifications.py               # Telegram integration
├── config.py                      # Configuration management
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── agent_memory.db               # SQLite database (auto-created)
└── agent_jobs.db                 # Scheduler database (auto-created)
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Required | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5-20250929` | AI model to use |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID |
| `DEFAULT_WATCHLIST` | `RELIANCE,TCS,HDFCBANK,INFY,ITC` | Stocks to monitor |
| `RISK_TOLERANCE` | `medium` | Risk level: low/medium/high |
| `ALERT_THRESHOLD` | `3.0` | Price move % to trigger alerts |
| `MORNING_RUN_TIME` | `08:30` | Morning analysis time (IST) |
| `EVENING_RUN_TIME` | `20:30` | Evening analysis time (IST) |
| `SCHEDULER_ENABLED` | `True` | Enable autonomous scheduling |
| `DATABASE_PATH` | `agent_memory.db` | Database file path |
| `LOG_LEVEL` | `INFO` | Logging level |

### Risk Tolerance Settings

- **Low**: Conservative approach, frequent alerts, smaller positions
- **Medium**: Balanced approach, standard thresholds
- **High**: Aggressive approach, fewer alerts, larger positions

## 📊 Usage Examples

### 1. Comprehensive Stock Analysis
```python
from indian_stock_market_agent import EnhancedIndianStockMarketAgent
from config import load_config

# Initialize agent
config = load_config()
agent = EnhancedIndianStockMarketAgent(config)

# Analyze a stock (agent will use multiple tools automatically)
response = await agent.analyze_stock_comprehensive('RELIANCE')
print(response.final_answer)
print(f"Tools used: {response.tools_used}")
```

### 2. Portfolio Health Check
```python
# Check portfolio performance
portfolio = {
    'RELIANCE': {'qty': 100, 'avg_price': 2400.0},
    'TCS': {'qty': 50, 'avg_price': 3200.0}
}

response = await agent.portfolio_health_check(portfolio)
print(response.final_answer)
```

### 3. Autonomous Morning Analysis
```python
# Run morning pre-market analysis
response = await agent.autonomous_morning_analysis()
print(response.final_answer)
```

### 4. Custom Agent Task
```python
# Ask the agent to perform any task
task = "Analyze the impact of USD/INR movement on IT sector stocks and recommend top 3 picks"
response = await agent.run_agent_loop(task)
print(response.final_answer)
```

## 📱 Dashboard Features

### Market Overview
- Real-time Nifty 50 and Sensex data
- USD/INR exchange rates
- Watchlist stock performance
- Agent status and health

### Stock Analysis
- Quick traditional analysis with charts
- AI-powered comprehensive analysis using agent reasoning
- Technical and fundamental metrics
- Investment recommendations

### Portfolio Management
- Multi-asset portfolio tracking
- Performance analysis and P&L calculation
- AI-powered portfolio optimization suggestions
- Sector allocation and risk analysis

### Agent Activity Monitor
- Recent agent decisions and reasoning
- Alerts sent and notification history
- Analysis history and patterns
- Tool usage statistics

### Agent Settings
- Watchlist management
- Risk tolerance configuration
- Notification preferences
- Scheduling settings
- Manual trigger controls

### AI Assistant
- Interactive chat with the agent
- Multi-step reasoning for complex queries
- Tool integration for comprehensive answers
- Conversation history

## 🛡️ Safety and Disclaimers

### Agent Safety
- **No Automatic Trading**: Agent only provides recommendations, never executes trades
- **Risk Assessment**: Always includes risk analysis in recommendations
- **Explainable Decisions**: Complete reasoning trace for all decisions
- **Configurable Limits**: User-defined risk tolerance and position limits

### Investment Disclaimer
⚠️ **Important**: This tool is for educational and informational purposes only. It does not constitute financial advice. Always:
- Consult with qualified financial advisors
- Conduct your own research
- Consider your risk tolerance
- Never invest more than you can afford to lose
- Understand that past performance doesn't guarantee future results

## 🔧 Troubleshooting

### Common Issues

1. **Agent not available**: Check `ANTHROPIC_API_KEY` is set correctly
2. **Telegram not working**: Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
3. **Database errors**: Ensure write permissions in project directory
4. **Scheduler not running**: Check `SCHEDULER_ENABLED=True` and restart
5. **Stock data issues**: yfinance API limitations, try again later

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python indian_stock_market_agent.py
```

### Testing Components

```bash
# Test configuration
python config.py

# Test memory system
python memory.py

# Test notifications
python notifications.py

# Test individual tools
python tools.py
```

## 🚀 Deployment

### Local Development
- Run with `streamlit run indian_dashboard.py`
- Use `python scheduler.py` for autonomous operation

### Production Deployment
1. Set up on cloud VM (Google Cloud, AWS, Azure)
2. Configure environment variables
3. Set up systemd service for scheduler
4. Use reverse proxy (nginx) for web access
5. Set up monitoring and logging

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "indian_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is for educational purposes. Please review the license file for terms and conditions.

## 🙋‍♂️ Support

For questions, issues, or suggestions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed description

---

**Built with ❤️ for the Indian investor community**

*Powered by Anthropic Claude API • Real-time market data • Autonomous reasoning*
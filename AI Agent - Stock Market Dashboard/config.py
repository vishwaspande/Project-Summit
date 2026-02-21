#!/usr/bin/env python3
"""
Configuration management for Indian Stock Market AI Agent
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class AgentConfig:
    """Configuration class for the AI Agent."""

    # API Configuration
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4000
    temperature: float = 0.3

    # Telegram Configuration
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False

    # Default Watchlist (NSE symbols)
    default_watchlist: List[str] = None

    # Risk Management
    risk_tolerance: str = "medium"  # low, medium, high
    alert_threshold: float = 3.0  # Percentage move to trigger alert
    max_position_size: float = 10.0  # Maximum position size as % of portfolio
    stop_loss_threshold: float = 5.0  # Stop loss threshold in %

    # Scheduling Configuration
    morning_run_time: str = "08:30"  # IST
    evening_run_time: str = "20:30"  # IST
    scheduler_enabled: bool = True

    # Database Configuration
    database_path: str = "agent_memory.db"

    # Market Data Configuration
    cache_duration: int = 300  # seconds (5 minutes)
    max_retries: int = 3
    request_timeout: int = 30

    # Portfolio Configuration
    default_portfolio: Dict[str, Dict[str, float]] = None
    portfolio_rebalance_threshold: float = 5.0  # % deviation to trigger rebalancing

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "agent.log"

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.default_watchlist is None:
            self.default_watchlist = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC',
                'HINDUNILVR', 'BHARTIARTL', 'KOTAKBANK',
                'ASIANPAINT', 'MARUTI'
            ]

        if self.default_portfolio is None:
            self.default_portfolio = {
                'RELIANCE': {'qty': 100, 'avg_price': 2400.0},
                'HDFCBANK': {'qty': 75, 'avg_price': 1500.0},
                'TCS': {'qty': 50, 'avg_price': 3200.0},
                'INFY': {'qty': 200, 'avg_price': 1400.0},
                'ITC': {'qty': 500, 'avg_price': 350.0}
            }

def load_config() -> AgentConfig:
    """Load configuration from environment variables and defaults."""

    config = AgentConfig()

    # Load from environment variables
    config.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
    config.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    config.telegram_enabled = bool(config.telegram_bot_token and config.telegram_chat_id)

    # Risk and alert settings
    config.risk_tolerance = os.getenv('RISK_TOLERANCE', config.risk_tolerance).lower()
    config.alert_threshold = float(os.getenv('ALERT_THRESHOLD', str(config.alert_threshold)))

    # Scheduling settings
    config.morning_run_time = os.getenv('MORNING_RUN_TIME', config.morning_run_time)
    config.evening_run_time = os.getenv('EVENING_RUN_TIME', config.evening_run_time)
    config.scheduler_enabled = os.getenv('SCHEDULER_ENABLED', 'True').lower() == 'true'

    # Database settings
    config.database_path = os.getenv('DATABASE_PATH', config.database_path)

    # Logging settings
    config.log_level = os.getenv('LOG_LEVEL', config.log_level).upper()
    config.log_file = os.getenv('LOG_FILE', config.log_file)

    # Model settings
    config.model_name = os.getenv('ANTHROPIC_MODEL', config.model_name)
    config.max_tokens = int(os.getenv('MAX_TOKENS', str(config.max_tokens)))
    config.temperature = float(os.getenv('TEMPERATURE', str(config.temperature)))

    # Parse watchlist from environment if provided
    watchlist_env = os.getenv('DEFAULT_WATCHLIST')
    if watchlist_env:
        config.default_watchlist = [stock.strip() for stock in watchlist_env.split(',')]

    return config

def setup_logging(config: AgentConfig) -> logging.Logger:
    """Setup logging configuration."""

    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Set up file and console logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {config.log_level}, File: {config.log_file}")

    return logger

def validate_config(config: AgentConfig) -> bool:
    """Validate the configuration."""

    errors = []

    # Check required API key
    if not config.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is required")

    # Validate risk tolerance
    if config.risk_tolerance not in ['low', 'medium', 'high']:
        errors.append("RISK_TOLERANCE must be one of: low, medium, high")

    # Validate thresholds
    if not (0 < config.alert_threshold <= 20):
        errors.append("ALERT_THRESHOLD must be between 0 and 20")

    if not (1 < config.max_position_size <= 50):
        errors.append("MAX_POSITION_SIZE must be between 1 and 50")

    # Validate time formats
    try:
        from datetime import datetime
        datetime.strptime(config.morning_run_time, "%H:%M")
        datetime.strptime(config.evening_run_time, "%H:%M")
    except ValueError:
        errors.append("Time format must be HH:MM (24-hour format)")

    # Validate model parameters
    if not (100 <= config.max_tokens <= 8000):
        errors.append("MAX_TOKENS must be between 100 and 8000")

    if not (0 <= config.temperature <= 1):
        errors.append("TEMPERATURE must be between 0 and 1")

    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Configuration validation passed")
    return True

def get_risk_multipliers(risk_tolerance: str) -> Dict[str, float]:
    """Get risk-based multipliers for different parameters."""

    multipliers = {
        'low': {
            'alert_threshold': 0.7,  # More sensitive alerts
            'position_size': 0.5,    # Smaller positions
            'rebalance_threshold': 0.8  # More frequent rebalancing
        },
        'medium': {
            'alert_threshold': 1.0,  # Normal sensitivity
            'position_size': 1.0,    # Normal positions
            'rebalance_threshold': 1.0  # Normal rebalancing
        },
        'high': {
            'alert_threshold': 1.5,  # Less sensitive alerts
            'position_size': 1.5,    # Larger positions
            'rebalance_threshold': 1.5  # Less frequent rebalancing
        }
    }

    return multipliers.get(risk_tolerance, multipliers['medium'])

# Example usage
if __name__ == "__main__":
    # Load and validate configuration
    config = load_config()

    print("🔧 Indian Stock Market AI Agent Configuration")
    print("=" * 50)
    print(f"API Key: {'✅ Set' if config.anthropic_api_key else '❌ Missing'}")
    print(f"Model: {config.model_name}")
    print(f"Risk Tolerance: {config.risk_tolerance}")
    print(f"Alert Threshold: {config.alert_threshold}%")
    print(f"Telegram: {'✅ Enabled' if config.telegram_enabled else '❌ Disabled'}")
    print(f"Scheduler: {'✅ Enabled' if config.scheduler_enabled else '❌ Disabled'}")
    print(f"Morning Run: {config.morning_run_time} IST")
    print(f"Evening Run: {config.evening_run_time} IST")
    print(f"Watchlist: {', '.join(config.default_watchlist[:5])}...")
    print("=" * 50)

    # Validate configuration
    is_valid = validate_config(config)

    if is_valid:
        print("🚀 Configuration ready for agent startup!")
    else:
        print("❌ Please fix configuration errors before starting agent")
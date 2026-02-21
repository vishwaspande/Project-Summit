#!/usr/bin/env python3
"""
Telegram bot notification system for the Indian Stock Market AI Agent

This module handles sending notifications via Telegram bot API for:
- Morning and evening briefings
- Urgent alerts (price moves, news)
- Trade recommendations
- Portfolio updates
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pytz
import json
from dataclasses import dataclass
import asyncio
import aiohttp
from config import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class NotificationResult:
    """Result of notification attempt."""
    success: bool
    message_id: Optional[int] = None
    error: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

class NotificationManager:
    """Manages Telegram bot notifications for the AI agent."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.enabled = config.telegram_enabled
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.indian_timezone = pytz.timezone('Asia/Kolkata')

        # Message formatting templates
        self.templates = {
            'morning_briefing': """🌅 **Indian Markets - Morning Briefing**

📊 **Market Overview:**
{market_data}

📈 **Key Highlights:**
{highlights}

⚠️ **Today's Focus:**
{focus_points}

🕘 Generated at {timestamp} IST""",

            'evening_briefing': """🌆 **Indian Markets - Evening Summary**

📊 **Today's Performance:**
{performance_data}

🌍 **Global Impact:**
{global_impact}

📋 **Tomorrow's Outlook:**
{outlook}

🕕 Generated at {timestamp} IST""",

            'price_alert': """🚨 **Price Alert**

📊 **{symbol}**: ₹{price} ({change:+.1f}%)

🔍 **Trigger:** {trigger}

📝 **Analysis:** {analysis}

🕐 {timestamp} IST""",

            'trade_recommendation': """💡 **Trade Recommendation**

📊 **Stock:** {symbol}
💰 **Action:** {action}
🎯 **Target:** ₹{target}
🛡️ **Stop Loss:** ₹{stop_loss}

📋 **Rationale:**
{rationale}

⚖️ **Risk Level:** {risk_level}

🕐 {timestamp} IST""",

            'portfolio_update': """💼 **Portfolio Update**

📈 **Current Value:** ₹{current_value:,.0f}
📊 **P&L:** ₹{pnl:,.0f} ({pnl_percent:+.1f}%)
📅 **Day Change:** ₹{day_change:,.0f} ({day_change_percent:+.1f}%)

🎯 **Top Performers:**
{top_performers}

📉 **Attention Needed:**
{attention_needed}

🕐 {timestamp} IST"""
        }

        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram notifications disabled - check bot token and chat ID")

    def send_notification(self, message: str, alert_type: str = "info",
                         parse_mode: str = "Markdown", disable_preview: bool = True) -> bool:
        """
        Send a notification via Telegram bot.

        Args:
            message (str): Message to send
            alert_type (str): Type of alert (info, warning, critical)
            parse_mode (str): Telegram parse mode
            disable_preview (bool): Disable web page preview

        Returns:
            bool: Success status
        """
        if not self.enabled:
            logger.warning("Telegram notifications disabled")
            return False

        try:
            # Add emoji based on alert type
            emoji_map = {
                'info': '📢',
                'warning': '⚠️',
                'critical': '🚨',
                'success': '✅',
                'error': '❌'
            }

            emoji = emoji_map.get(alert_type, '📢')
            formatted_message = f"{emoji} {message}"

            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': disable_preview
            }

            response = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info(f"Notification sent successfully: {alert_type}")
                    return True
                else:
                    logger.error(f"Telegram API error: {result.get('description', 'Unknown error')}")
                    return False
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    def send_morning_briefing(self, market_data: Dict[str, Any],
                            highlights: List[str], focus_points: List[str]) -> bool:
        """
        Send morning briefing before market opens.

        Args:
            market_data (Dict): Market overview data
            highlights (List[str]): Key highlights
            focus_points (List[str]): Points to focus on today

        Returns:
            bool: Success status
        """
        try:
            # Format market data
            market_summary = []
            if 'nifty' in market_data:
                nifty = market_data['nifty']
                market_summary.append(f"Nifty 50: {nifty['value']:.0f} ({nifty['change']:+.1f}%)")

            if 'sensex' in market_data:
                sensex = market_data['sensex']
                market_summary.append(f"Sensex: {sensex['value']:.0f} ({sensex['change']:+.1f}%)")

            if 'usd_inr' in market_data:
                usd_inr = market_data['usd_inr']
                market_summary.append(f"USD/INR: ₹{usd_inr['rate']:.2f} ({usd_inr['change']:+.2f})")

            # Format highlights and focus points
            highlights_text = "\n".join([f"• {highlight}" for highlight in highlights])
            focus_text = "\n".join([f"• {point}" for point in focus_points])

            message = self.templates['morning_briefing'].format(
                market_data="\n".join(market_summary),
                highlights=highlights_text,
                focus_points=focus_text,
                timestamp=datetime.now(self.indian_timezone).strftime("%H:%M")
            )

            return self.send_notification(message, "info")

        except Exception as e:
            logger.error(f"Error sending morning briefing: {e}")
            return False

    def send_evening_briefing(self, performance_data: Dict[str, Any],
                            global_impact: str, outlook: str) -> bool:
        """
        Send evening briefing after market closes.

        Args:
            performance_data (Dict): Today's performance data
            global_impact (str): Global market impact summary
            outlook (str): Tomorrow's outlook

        Returns:
            bool: Success status
        """
        try:
            # Format performance data
            performance_summary = []
            if 'indices' in performance_data:
                for index, data in performance_data['indices'].items():
                    performance_summary.append(f"{index}: {data['close']:.0f} ({data['change']:+.1f}%)")

            if 'sectors' in performance_data:
                performance_summary.append("\n**Sector Performance:**")
                for sector, change in performance_data['sectors'].items():
                    performance_summary.append(f"• {sector}: {change:+.1f}%")

            message = self.templates['evening_briefing'].format(
                performance_data="\n".join(performance_summary),
                global_impact=global_impact,
                outlook=outlook,
                timestamp=datetime.now(self.indian_timezone).strftime("%H:%M")
            )

            return self.send_notification(message, "info")

        except Exception as e:
            logger.error(f"Error sending evening briefing: {e}")
            return False

    def send_price_alert(self, symbol: str, price: float, change: float,
                        trigger: str, analysis: str) -> bool:
        """
        Send price movement alert.

        Args:
            symbol (str): Stock symbol
            price (float): Current price
            change (float): Price change percentage
            trigger (str): What triggered the alert
            analysis (str): Brief analysis

        Returns:
            bool: Success status
        """
        try:
            alert_type = "critical" if abs(change) >= 5 else "warning"

            message = self.templates['price_alert'].format(
                symbol=symbol,
                price=price,
                change=change,
                trigger=trigger,
                analysis=analysis,
                timestamp=datetime.now(self.indian_timezone).strftime("%H:%M")
            )

            return self.send_notification(message, alert_type)

        except Exception as e:
            logger.error(f"Error sending price alert: {e}")
            return False

    def send_trade_recommendation(self, symbol: str, action: str, target: float,
                                stop_loss: float, rationale: str, risk_level: str) -> bool:
        """
        Send trade recommendation.

        Args:
            symbol (str): Stock symbol
            action (str): BUY/SELL/HOLD
            target (float): Target price
            stop_loss (float): Stop loss price
            rationale (str): Reasoning for the recommendation
            risk_level (str): Risk level assessment

        Returns:
            bool: Success status
        """
        try:
            message = self.templates['trade_recommendation'].format(
                symbol=symbol,
                action=action,
                target=target,
                stop_loss=stop_loss,
                rationale=rationale,
                risk_level=risk_level,
                timestamp=datetime.now(self.indian_timezone).strftime("%H:%M")
            )

            return self.send_notification(message, "info")

        except Exception as e:
            logger.error(f"Error sending trade recommendation: {e}")
            return False

    def send_portfolio_update(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Send portfolio performance update.

        Args:
            portfolio_data (Dict): Portfolio performance data

        Returns:
            bool: Success status
        """
        try:
            # Format top performers and attention needed
            top_performers = []
            attention_needed = []

            if 'positions' in portfolio_data:
                # Sort positions by performance
                positions = sorted(
                    portfolio_data['positions'],
                    key=lambda x: x.get('pnl_percent', 0),
                    reverse=True
                )

                # Top 3 performers
                for pos in positions[:3]:
                    if pos.get('pnl_percent', 0) > 0:
                        top_performers.append(f"• {pos['symbol']}: {pos['pnl_percent']:+.1f}%")

                # Bottom 3 or negative performers
                for pos in positions[-3:]:
                    if pos.get('pnl_percent', 0) < -2:  # More than 2% loss
                        attention_needed.append(f"• {pos['symbol']}: {pos['pnl_percent']:+.1f}%")

            top_performers_text = "\n".join(top_performers) if top_performers else "None today"
            attention_text = "\n".join(attention_needed) if attention_needed else "All positions healthy"

            message = self.templates['portfolio_update'].format(
                current_value=portfolio_data.get('current_value', 0),
                pnl=portfolio_data.get('total_pnl', 0),
                pnl_percent=portfolio_data.get('total_pnl_percent', 0),
                day_change=portfolio_data.get('day_change', 0),
                day_change_percent=portfolio_data.get('day_change_percent', 0),
                top_performers=top_performers_text,
                attention_needed=attention_text,
                timestamp=datetime.now(self.indian_timezone).strftime("%H:%M")
            )

            return self.send_notification(message, "info")

        except Exception as e:
            logger.error(f"Error sending portfolio update: {e}")
            return False

    def send_custom_alert(self, title: str, message: str, alert_type: str = "info") -> bool:
        """
        Send custom formatted alert.

        Args:
            title (str): Alert title
            message (str): Alert message
            alert_type (str): Alert type

        Returns:
            bool: Success status
        """
        try:
            formatted_message = f"**{title}**\n\n{message}\n\n🕐 {datetime.now(self.indian_timezone).strftime('%H:%M')} IST"
            return self.send_notification(formatted_message, alert_type)

        except Exception as e:
            logger.error(f"Error sending custom alert: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test Telegram bot connection.

        Returns:
            bool: Connection status
        """
        if not self.enabled:
            logger.warning("Telegram notifications disabled")
            return False

        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    bot_info = result.get('result', {})
                    logger.info(f"Telegram bot connected: {bot_info.get('first_name', 'Unknown')}")
                    return True
                else:
                    logger.error(f"Telegram bot test failed: {result.get('description')}")
                    return False
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error testing Telegram connection: {e}")
            return False

    async def send_notification_async(self, message: str, alert_type: str = "info") -> bool:
        """
        Send notification asynchronously.

        Args:
            message (str): Message to send
            alert_type (str): Alert type

        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False

        try:
            emoji_map = {
                'info': '📢',
                'warning': '⚠️',
                'critical': '🚨',
                'success': '✅',
                'error': '❌'
            }

            emoji = emoji_map.get(alert_type, '📢')
            formatted_message = f"{emoji} {message}"

            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('ok'):
                            logger.info(f"Async notification sent: {alert_type}")
                            return True
                        else:
                            logger.error(f"Telegram API error: {result.get('description')}")
                            return False
                    else:
                        logger.error(f"HTTP error {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending async notification: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    from config import load_config

    print("📱 Telegram Notification Manager Test")
    print("=" * 45)

    # Load configuration
    config = load_config()

    if not config.telegram_enabled:
        print("❌ Telegram notifications not configured")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        exit(1)

    # Initialize notification manager
    notifier = NotificationManager(config)

    # Test connection
    connection_ok = notifier.test_connection()
    print(f"Connection test: {'✅' if connection_ok else '❌'}")

    if connection_ok:
        # Test basic notification
        success = notifier.send_notification(
            "🧪 Test notification from Indian Stock Market AI Agent",
            "info"
        )
        print(f"Test notification: {'✅' if success else '❌'}")

        # Test morning briefing
        test_market_data = {
            'nifty': {'value': 19500, 'change': 1.2},
            'sensex': {'value': 65000, 'change': 0.8},
            'usd_inr': {'rate': 83.15, 'change': 0.05}
        }

        test_highlights = [
            "IT stocks showing strong momentum",
            "Banking sector under pressure",
            "FII buying continues in pharma"
        ]

        test_focus = [
            "RBI policy announcement expected",
            "Q3 earnings season begins",
            "Watch for USD/INR levels"
        ]

        success = notifier.send_morning_briefing(
            test_market_data,
            test_highlights,
            test_focus
        )
        print(f"Morning briefing: {'✅' if success else '❌'}")

        # Test price alert
        success = notifier.send_price_alert(
            symbol="RELIANCE",
            price=2450.75,
            change=3.5,
            trigger="Volume surge + breakout above ₹2400",
            analysis="Strong momentum with good volume support. Consider partial booking above ₹2500."
        )
        print(f"Price alert: {'✅' if success else '❌'}")

    print("\n✅ Notification Manager test completed!")
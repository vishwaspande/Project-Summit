#!/usr/bin/env python3
"""
SQLite memory system for the Indian Stock Market AI Agent

This module handles all database operations for storing and retrieving
agent analyses, decisions, alerts, and portfolio snapshots.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pytz
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Analysis:
    """Structure for storing analysis records."""
    id: Optional[int] = None
    analysis_type: str = ""
    symbol: str = ""
    data: Dict[str, Any] = None
    reasoning: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    timestamp: datetime = None
    source: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

@dataclass
class Alert:
    """Structure for storing alert records."""
    id: Optional[int] = None
    message: str = ""
    alert_type: str = ""
    symbol: str = ""
    trigger_condition: str = ""
    sent: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

@dataclass
class PortfolioSnapshot:
    """Structure for storing portfolio snapshots."""
    id: Optional[int] = None
    portfolio_data: Dict[str, Any] = None
    total_value: float = 0.0
    total_pnl: float = 0.0
    pnl_percent: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

@dataclass
class AgentDecision:
    """Structure for storing agent decisions."""
    id: Optional[int] = None
    decision_type: str = ""
    context: str = ""
    reasoning: str = ""
    action_taken: str = ""
    outcome: str = ""
    confidence: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

@dataclass
class MarketCondition:
    """Structure for storing market conditions."""
    id: Optional[int] = None
    nifty_value: float = 0.0
    sensex_value: float = 0.0
    usd_inr_rate: float = 0.0
    fii_flow: float = 0.0
    dii_flow: float = 0.0
    market_sentiment: str = ""
    volatility_index: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

class MemoryManager:
    """Manages SQLite database operations for the AI agent."""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Analyses table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    symbol TEXT,
                    data TEXT,
                    reasoning TEXT,
                    recommendation TEXT,
                    confidence REAL,
                    timestamp DATETIME,
                    source TEXT
                )
                ''')

                # Alerts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts_sent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    alert_type TEXT,
                    symbol TEXT,
                    trigger_condition TEXT,
                    sent BOOLEAN,
                    timestamp DATETIME
                )
                ''')

                # Portfolio snapshots table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_data TEXT,
                    total_value REAL,
                    total_pnl REAL,
                    pnl_percent REAL,
                    timestamp DATETIME
                )
                ''')

                # Agent decisions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_type TEXT,
                    context TEXT,
                    reasoning TEXT,
                    action_taken TEXT,
                    outcome TEXT,
                    confidence REAL,
                    timestamp DATETIME
                )
                ''')

                # Market conditions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nifty_value REAL,
                    sensex_value REAL,
                    usd_inr_rate REAL,
                    fii_flow REAL,
                    dii_flow REAL,
                    market_sentiment TEXT,
                    volatility_index REAL,
                    timestamp DATETIME
                )
                ''')

                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_type ON analyses(analysis_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_symbol ON analyses(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts_sent(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_snapshots(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON agent_decisions(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_conditions(timestamp)')

                conn.commit()
                logger.info(f"Database initialized successfully: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def store_analysis(self, analysis_type: str, data: Dict[str, Any],
                      reasoning: str = "", symbol: str = "",
                      recommendation: str = "", confidence: float = 0.0,
                      source: str = "") -> bool:
        """
        Store analysis results in database.

        Args:
            analysis_type (str): Type of analysis
            data (Dict): Analysis data
            reasoning (str): AI reasoning
            symbol (str): Stock symbol if applicable
            recommendation (str): AI recommendation
            confidence (float): Confidence level (0-1)
            source (str): Data source

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO analyses
                (analysis_type, symbol, data, reasoning, recommendation, confidence, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_type,
                    symbol,
                    json.dumps(data, default=str),
                    reasoning,
                    recommendation,
                    confidence,
                    datetime.now(self.indian_timezone),
                    source
                ))

                conn.commit()
                logger.info(f"Stored analysis: {analysis_type} for {symbol or 'general'}")
                return True

        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            return False

    def store_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Store alert information in database.

        Args:
            alert_data (Dict): Alert information

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO alerts_sent
                (message, alert_type, symbol, trigger_condition, sent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_data.get('message', ''),
                    alert_data.get('alert_type', 'info'),
                    alert_data.get('symbol', ''),
                    alert_data.get('trigger_condition', ''),
                    alert_data.get('sent', False),
                    alert_data.get('timestamp', datetime.now(self.indian_timezone))
                ))

                conn.commit()
                logger.info(f"Stored alert: {alert_data.get('alert_type', 'info')}")
                return True

        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            return False

    def store_portfolio_snapshot(self, portfolio_data: Dict[str, Any],
                                total_value: float, total_pnl: float,
                                pnl_percent: float) -> bool:
        """
        Store portfolio snapshot for tracking P&L over time.

        Args:
            portfolio_data (Dict): Portfolio positions data
            total_value (float): Total portfolio value
            total_pnl (float): Total P&L
            pnl_percent (float): P&L percentage

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO portfolio_snapshots
                (portfolio_data, total_value, total_pnl, pnl_percent, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    json.dumps(portfolio_data, default=str),
                    total_value,
                    total_pnl,
                    pnl_percent,
                    datetime.now(self.indian_timezone)
                ))

                conn.commit()
                logger.info(f"Stored portfolio snapshot: ₹{total_value:,.0f} ({pnl_percent:+.1f}%)")
                return True

        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {e}")
            return False

    def store_agent_decision(self, decision_type: str, context: str,
                            reasoning: str, action_taken: str,
                            outcome: str = "", confidence: float = 0.0) -> bool:
        """
        Store agent decision for explainability.

        Args:
            decision_type (str): Type of decision made
            context (str): Context of the decision
            reasoning (str): Agent's reasoning
            action_taken (str): Action that was taken
            outcome (str): Outcome of the action
            confidence (float): Confidence in the decision

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO agent_decisions
                (decision_type, context, reasoning, action_taken, outcome, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    decision_type,
                    context,
                    reasoning,
                    action_taken,
                    outcome,
                    confidence,
                    datetime.now(self.indian_timezone)
                ))

                conn.commit()
                logger.info(f"Stored agent decision: {decision_type}")
                return True

        except Exception as e:
            logger.error(f"Error storing agent decision: {e}")
            return False

    def store_market_condition(self, nifty_value: float, sensex_value: float,
                              usd_inr_rate: float, fii_flow: float = 0.0,
                              dii_flow: float = 0.0, market_sentiment: str = "",
                              volatility_index: float = 0.0) -> bool:
        """
        Store daily market condition summary.

        Args:
            nifty_value (float): Nifty 50 value
            sensex_value (float): Sensex value
            usd_inr_rate (float): USD/INR rate
            fii_flow (float): FII flow in crores
            dii_flow (float): DII flow in crores
            market_sentiment (str): Overall market sentiment
            volatility_index (float): Volatility measure

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO market_conditions
                (nifty_value, sensex_value, usd_inr_rate, fii_flow, dii_flow,
                 market_sentiment, volatility_index, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    nifty_value,
                    sensex_value,
                    usd_inr_rate,
                    fii_flow,
                    dii_flow,
                    market_sentiment,
                    volatility_index,
                    datetime.now(self.indian_timezone)
                ))

                conn.commit()
                logger.info(f"Stored market condition: Nifty {nifty_value:.0f}, Sensex {sensex_value:.0f}")
                return True

        except Exception as e:
            logger.error(f"Error storing market condition: {e}")
            return False

    def get_recent_analyses(self, analysis_type: str = None, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Retrieve recent analyses from database.

        Args:
            analysis_type (str): Filter by analysis type
            days_back (int): Number of days to look back

        Returns:
            List[Dict]: List of analysis records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_back)

                if analysis_type:
                    cursor.execute('''
                    SELECT * FROM analyses
                    WHERE analysis_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    ''', (analysis_type, cutoff_date))
                else:
                    cursor.execute('''
                    SELECT * FROM analyses
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    ''', (cutoff_date,))

                rows = cursor.fetchall()

                analyses = []
                for row in rows:
                    analysis = dict(row)
                    # Parse JSON data
                    if analysis['data']:
                        analysis['data'] = json.loads(analysis['data'])
                    analyses.append(analysis)

                logger.info(f"Retrieved {len(analyses)} analyses")
                return analyses

        except Exception as e:
            logger.error(f"Error retrieving analyses: {e}")
            return []

    def get_recent_alerts(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Retrieve recent alerts from database.

        Args:
            days_back (int): Number of days to look back

        Returns:
            List[Dict]: List of alert records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_back)

                cursor.execute('''
                SELECT * FROM alerts_sent
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                ''', (cutoff_date,))

                rows = cursor.fetchall()
                alerts = [dict(row) for row in rows]

                logger.info(f"Retrieved {len(alerts)} alerts")
                return alerts

        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []

    def get_portfolio_history(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve portfolio snapshots for performance tracking.

        Args:
            days_back (int): Number of days to look back

        Returns:
            List[Dict]: List of portfolio snapshots
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_back)

                cursor.execute('''
                SELECT * FROM portfolio_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                ''', (cutoff_date,))

                rows = cursor.fetchall()

                snapshots = []
                for row in rows:
                    snapshot = dict(row)
                    # Parse JSON data
                    if snapshot['portfolio_data']:
                        snapshot['portfolio_data'] = json.loads(snapshot['portfolio_data'])
                    snapshots.append(snapshot)

                logger.info(f"Retrieved {len(snapshots)} portfolio snapshots")
                return snapshots

        except Exception as e:
            logger.error(f"Error retrieving portfolio history: {e}")
            return []

    def get_agent_decisions(self, decision_type: str = None, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Retrieve agent decisions for explainability.

        Args:
            decision_type (str): Filter by decision type
            days_back (int): Number of days to look back

        Returns:
            List[Dict]: List of decision records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_back)

                if decision_type:
                    cursor.execute('''
                    SELECT * FROM agent_decisions
                    WHERE decision_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    ''', (decision_type, cutoff_date))
                else:
                    cursor.execute('''
                    SELECT * FROM agent_decisions
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    ''', (cutoff_date,))

                rows = cursor.fetchall()
                decisions = [dict(row) for row in rows]

                logger.info(f"Retrieved {len(decisions)} agent decisions")
                return decisions

        except Exception as e:
            logger.error(f"Error retrieving agent decisions: {e}")
            return []

    def get_market_conditions(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve market conditions history.

        Args:
            days_back (int): Number of days to look back

        Returns:
            List[Dict]: List of market condition records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_back)

                cursor.execute('''
                SELECT * FROM market_conditions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                ''', (cutoff_date,))

                rows = cursor.fetchall()
                conditions = [dict(row) for row in rows]

                logger.info(f"Retrieved {len(conditions)} market conditions")
                return conditions

        except Exception as e:
            logger.error(f"Error retrieving market conditions: {e}")
            return []

    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """
        Clean up old data to maintain database size.

        Args:
            days_to_keep (int): Number of days of data to keep

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_date = datetime.now(self.indian_timezone) - timedelta(days=days_to_keep)

                # Clean up old records from all tables
                tables = ['analyses', 'alerts_sent', 'portfolio_snapshots',
                         'agent_decisions', 'market_conditions']

                total_deleted = 0
                for table in tables:
                    cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_date,))
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    logger.info(f"Deleted {deleted} old records from {table}")

                # Vacuum to reclaim space
                cursor.execute('VACUUM')

                conn.commit()
                logger.info(f"Database cleanup completed. Deleted {total_deleted} old records.")
                return True

        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict: Database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}
                tables = {
                    'analyses': 'Analysis records',
                    'alerts_sent': 'Alerts sent',
                    'portfolio_snapshots': 'Portfolio snapshots',
                    'agent_decisions': 'Agent decisions',
                    'market_conditions': 'Market conditions'
                }

                for table, description in tables.items():
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats[description] = count

                # Get database file size
                import os
                if os.path.exists(self.db_path):
                    size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
                    stats['Database size (MB)'] = round(size_mb, 2)

                return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize memory manager
    memory = MemoryManager("test_memory.db")

    print("🗃️  Memory Manager Test")
    print("=" * 40)

    # Test storing analysis
    test_data = {
        "symbol": "RELIANCE",
        "price": 2450.75,
        "recommendation": "BUY",
        "target": 2600.0
    }

    success = memory.store_analysis(
        analysis_type="stock_analysis",
        data=test_data,
        reasoning="Strong fundamentals and technical breakout",
        symbol="RELIANCE",
        recommendation="BUY",
        confidence=0.85
    )

    print(f"Store analysis: {'✅' if success else '❌'}")

    # Test retrieving analysis
    analyses = memory.get_recent_analyses("stock_analysis", 1)
    print(f"Retrieved analyses: {len(analyses)}")

    # Test portfolio snapshot
    portfolio_data = {
        "RELIANCE": {"qty": 100, "avg_price": 2400, "current_price": 2450},
        "TCS": {"qty": 50, "avg_price": 3200, "current_price": 3250}
    }

    success = memory.store_portfolio_snapshot(
        portfolio_data=portfolio_data,
        total_value=365000,
        total_pnl=7500,
        pnl_percent=2.1
    )

    print(f"Store portfolio: {'✅' if success else '❌'}")

    # Get database stats
    stats = memory.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Memory Manager test completed!")
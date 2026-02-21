#!/usr/bin/env python3
"""
APScheduler setup for autonomous agent runs

This module handles scheduling of the AI agent to run twice daily:
- Morning run (8:30 AM IST): Before Indian market opens at 9:15 AM
- Evening run (8:30 PM IST): Before US market opens
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Any
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import signal
import sys

from config import AgentConfig, load_config
from memory import MemoryManager
from notifications import NotificationManager
from tools import StockMarketTools

logger = logging.getLogger(__name__)

class AgentScheduler:
    """Handles scheduling of autonomous agent runs."""

    def __init__(self, config: AgentConfig, memory_manager: MemoryManager,
                 notification_manager: NotificationManager, tools: StockMarketTools,
                 agent_runner=None):
        self.config = config
        self.memory = memory_manager
        self.notifications = notification_manager
        self.tools = tools
        self.agent_runner = agent_runner
        self.indian_timezone = pytz.timezone('Asia/Kolkata')

        # Initialize scheduler
        self.scheduler = None
        self.setup_scheduler()

        # Track running jobs
        self.running_jobs = set()
        self.job_history = []

    def setup_scheduler(self):
        """Setup APScheduler with appropriate configuration."""
        try:
            # Configure job stores and executors
            jobstores = {
                'default': SQLAlchemyJobStore(url='sqlite:///agent_jobs.db')
            }

            executors = {
                'default': AsyncIOExecutor()
            }

            job_defaults = {
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 300  # 5 minutes
            }

            # Create scheduler
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=self.indian_timezone
            )

            # Add event listeners
            self.scheduler.add_listener(
                self._job_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )

            logger.info("Scheduler configured successfully")

        except Exception as e:
            logger.error(f"Error setting up scheduler: {e}")
            raise

    def _job_listener(self, event):
        """Handle scheduler job events."""
        job_id = event.job_id

        if event.exception:
            logger.error(f"Job {job_id} failed: {event.exception}")
            # Send error notification
            asyncio.create_task(self.notifications.send_notification(
                f"❌ Scheduled job {job_id} failed: {str(event.exception)}",
                "error"
            ))
        else:
            logger.info(f"Job {job_id} completed successfully")

        # Remove from running jobs
        self.running_jobs.discard(job_id)

    def schedule_morning_run(self):
        """Schedule daily morning run before Indian market opens."""
        if not self.config.scheduler_enabled:
            logger.info("Scheduler disabled in configuration")
            return

        try:
            # Parse morning run time
            hour, minute = map(int, self.config.morning_run_time.split(':'))

            # Schedule daily morning run
            self.scheduler.add_job(
                func=self.morning_analysis_job,
                trigger=CronTrigger(hour=hour, minute=minute),
                id='morning_run',
                name='Morning Market Analysis',
                replace_existing=True,
                max_instances=1
            )

            logger.info(f"Morning run scheduled for {self.config.morning_run_time} IST daily")

        except Exception as e:
            logger.error(f"Error scheduling morning run: {e}")

    def schedule_evening_run(self):
        """Schedule daily evening run before US market opens."""
        if not self.config.scheduler_enabled:
            logger.info("Scheduler disabled in configuration")
            return

        try:
            # Parse evening run time
            hour, minute = map(int, self.config.evening_run_time.split(':'))

            # Schedule daily evening run
            self.scheduler.add_job(
                func=self.evening_analysis_job,
                trigger=CronTrigger(hour=hour, minute=minute),
                id='evening_run',
                name='Evening Market Analysis',
                replace_existing=True,
                max_instances=1
            )

            logger.info(f"Evening run scheduled for {self.config.evening_run_time} IST daily")

        except Exception as e:
            logger.error(f"Error scheduling evening run: {e}")

    def schedule_periodic_monitoring(self):
        """Schedule periodic monitoring during market hours."""
        if not self.config.scheduler_enabled:
            return

        try:
            # Monitor every 30 minutes during market hours (9:15 AM - 3:30 PM IST)
            self.scheduler.add_job(
                func=self.periodic_monitoring_job,
                trigger=CronTrigger(
                    hour='9-15',  # Market hours
                    minute='15,45',  # Every 30 minutes
                    day_of_week='mon-fri'  # Weekdays only
                ),
                id='periodic_monitoring',
                name='Market Hours Monitoring',
                replace_existing=True,
                max_instances=1
            )

            logger.info("Periodic monitoring scheduled during market hours")

        except Exception as e:
            logger.error(f"Error scheduling periodic monitoring: {e}")

    def schedule_one_time_analysis(self, run_time: datetime, analysis_type: str,
                                  symbols: List[str] = None) -> str:
        """
        Schedule a one-time analysis at specific time.

        Args:
            run_time (datetime): When to run the analysis
            analysis_type (str): Type of analysis to perform
            symbols (List[str]): Optional list of symbols to analyze

        Returns:
            str: Job ID
        """
        try:
            job_id = f"onetime_{analysis_type}_{int(run_time.timestamp())}"

            self.scheduler.add_job(
                func=self.one_time_analysis_job,
                trigger=DateTrigger(run_date=run_time),
                args=[analysis_type, symbols or []],
                id=job_id,
                name=f'One-time {analysis_type} Analysis',
                max_instances=1
            )

            logger.info(f"One-time analysis scheduled: {job_id} at {run_time}")
            return job_id

        except Exception as e:
            logger.error(f"Error scheduling one-time analysis: {e}")
            return ""

    async def morning_analysis_job(self):
        """
        Morning analysis job - runs before Indian market opens.

        Analyzes:
        - Overnight global market impact
        - Watchlist stock screening
        - FII/DII flows
        - Currency movements
        - Generate morning briefing
        """
        job_id = 'morning_run'
        self.running_jobs.add(job_id)

        try:
            logger.info("🌅 Starting morning analysis job")

            # Initialize analysis results
            analysis_results = {
                'job_type': 'morning_analysis',
                'timestamp': datetime.now(self.indian_timezone),
                'results': {},
                'alerts': [],
                'briefing_sent': False
            }

            # 1. Get USD/INR rate and global impact
            logger.info("Fetching currency rates...")
            usd_inr_result = self.tools.get_usd_inr_rate()
            if usd_inr_result.success:
                analysis_results['results']['usd_inr'] = usd_inr_result.data

                # Check for significant currency moves
                change_pct = usd_inr_result.data.get('change_percent', 0)
                if abs(change_pct) > 0.5:  # More than 0.5% move
                    alert_msg = f"USD/INR moved {change_pct:+.2f}% to ₹{usd_inr_result.data['current_rate']:.2f}"
                    analysis_results['alerts'].append(alert_msg)

            # 2. Get FII/DII flows
            logger.info("Fetching FII/DII data...")
            fii_dii_result = self.tools.get_fii_dii_data()
            if fii_dii_result.success:
                analysis_results['results']['fii_dii'] = fii_dii_result.data

            # 3. Analyze watchlist stocks
            logger.info("Analyzing watchlist stocks...")
            watchlist_result = self.tools.get_multiple_stocks(self.config.default_watchlist)
            if watchlist_result.success:
                analysis_results['results']['watchlist'] = watchlist_result.data

                # Check for significant overnight moves
                for symbol, data in watchlist_result.data.items():
                    change_pct = data.get('change_percent', 0)
                    if abs(change_pct) > self.config.alert_threshold:
                        alert_msg = f"{symbol}: {change_pct:+.1f}% to ₹{data['current_price']}"
                        analysis_results['alerts'].append(alert_msg)

            # 4. Get market news
            logger.info("Fetching market news...")
            news_result = self.tools.get_market_news("Indian stock market morning", 5)
            if news_result.success:
                analysis_results['results']['news'] = news_result.data

            # 5. Use agent to generate comprehensive analysis
            if self.agent_runner:
                logger.info("Running agent analysis...")
                prompt = f"""
                Analyze the morning market data and generate insights for Indian market opening:

                USD/INR: {analysis_results['results'].get('usd_inr', {})}
                FII/DII Flows: {analysis_results['results'].get('fii_dii', {})}
                Key Watchlist Moves: {analysis_results.get('alerts', [])}

                Provide:
                1. Key overnight developments
                2. Stocks to watch today
                3. Sector outlook
                4. Risk factors
                5. Trading strategy for the day
                """

                agent_analysis = await self.agent_runner.run_analysis(prompt)
                analysis_results['agent_analysis'] = agent_analysis

            # 6. Send morning briefing
            logger.info("Sending morning briefing...")
            market_data = {
                'usd_inr': analysis_results['results'].get('usd_inr', {}),
                'fii_dii': analysis_results['results'].get('fii_dii', {})
            }

            highlights = analysis_results.get('alerts', [])[:5]  # Top 5 alerts
            focus_points = [
                "Monitor opening sentiment",
                "Watch for FII flow impact",
                "Track currency levels"
            ]

            briefing_sent = self.notifications.send_morning_briefing(
                market_data, highlights, focus_points
            )
            analysis_results['briefing_sent'] = briefing_sent

            # 7. Store results in memory
            self.memory.store_analysis(
                analysis_type="morning_analysis",
                data=analysis_results,
                reasoning="Automated morning market analysis before market open",
                source="scheduler"
            )

            # 8. Store agent decision
            self.memory.store_agent_decision(
                decision_type="morning_analysis",
                context="Pre-market analysis and briefing",
                reasoning="Analyzed overnight developments and market positioning",
                action_taken=f"Generated briefing with {len(highlights)} key points",
                outcome="Briefing sent successfully" if briefing_sent else "Briefing failed",
                confidence=0.8
            )

            logger.info(f"Morning analysis completed. Briefing sent: {briefing_sent}")

        except Exception as e:
            logger.error(f"Error in morning analysis job: {e}")
            await self.notifications.send_notification(
                f"❌ Morning analysis job failed: {str(e)}",
                "error"
            )
        finally:
            self.running_jobs.discard(job_id)

    async def evening_analysis_job(self):
        """
        Evening analysis job - runs before US market opens.

        Analyzes:
        - Indian market performance today
        - US market pre-open data
        - Currency impact assessment
        - Portfolio performance review
        - Generate evening briefing
        """
        job_id = 'evening_run'
        self.running_jobs.add(job_id)

        try:
            logger.info("🌆 Starting evening analysis job")

            analysis_results = {
                'job_type': 'evening_analysis',
                'timestamp': datetime.now(self.indian_timezone),
                'results': {},
                'portfolio_update': {},
                'briefing_sent': False
            }

            # 1. Analyze portfolio performance
            logger.info("Analyzing portfolio performance...")
            if hasattr(self.config, 'default_portfolio') and self.config.default_portfolio:
                portfolio_result = self.tools.check_portfolio(self.config.default_portfolio)
                if portfolio_result.success:
                    analysis_results['portfolio_update'] = portfolio_result.data

                    # Store portfolio snapshot
                    self.memory.store_portfolio_snapshot(
                        portfolio_data=portfolio_result.data,
                        total_value=portfolio_result.data.get('current_value', 0),
                        total_pnl=portfolio_result.data.get('total_pnl', 0),
                        pnl_percent=portfolio_result.data.get('total_pnl_percent', 0)
                    )

            # 2. Get market performance summary
            logger.info("Getting market performance...")
            indices_symbols = ['NIFTY', 'SENSEX', 'NIFTYBANK']  # Using index names

            # For demo, we'll use individual stocks as proxy
            market_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
            market_result = self.tools.get_multiple_stocks(market_stocks)
            if market_result.success:
                analysis_results['results']['market_performance'] = market_result.data

            # 3. Currency update
            logger.info("Updating currency rates...")
            usd_inr_result = self.tools.get_usd_inr_rate()
            if usd_inr_result.success:
                analysis_results['results']['usd_inr'] = usd_inr_result.data

            # 4. Get evening news
            logger.info("Fetching evening market news...")
            news_result = self.tools.get_market_news("Indian stock market closing", 5)
            if news_result.success:
                analysis_results['results']['news'] = news_result.data

            # 5. Use agent for comprehensive evening analysis
            if self.agent_runner:
                logger.info("Running evening agent analysis...")
                prompt = f"""
                Analyze today's Indian market performance and provide evening summary:

                Portfolio P&L: {analysis_results.get('portfolio_update', {})}
                Market Performance: {analysis_results['results'].get('market_performance', {})}
                USD/INR: {analysis_results['results'].get('usd_inr', {})}

                Provide:
                1. Today's key market themes
                2. Sector performance analysis
                3. Impact of global factors
                4. Tomorrow's outlook
                5. Overnight risks to monitor
                """

                agent_analysis = await self.agent_runner.run_analysis(prompt)
                analysis_results['agent_analysis'] = agent_analysis

            # 6. Send evening briefing
            logger.info("Sending evening briefing...")

            # Prepare performance data
            performance_data = {
                'portfolio': analysis_results.get('portfolio_update', {}),
                'market': analysis_results['results'].get('market_performance', {})
            }

            global_impact = f"USD/INR: {analysis_results['results'].get('usd_inr', {}).get('current_rate', 'N/A')}"
            outlook = "Monitor overnight global cues and US market opening"

            briefing_sent = self.notifications.send_evening_briefing(
                performance_data, global_impact, outlook
            )
            analysis_results['briefing_sent'] = briefing_sent

            # 7. Send portfolio update if configured
            if analysis_results.get('portfolio_update'):
                portfolio_sent = self.notifications.send_portfolio_update(
                    analysis_results['portfolio_update']
                )
                analysis_results['portfolio_notification_sent'] = portfolio_sent

            # 8. Store results
            self.memory.store_analysis(
                analysis_type="evening_analysis",
                data=analysis_results,
                reasoning="Automated evening market analysis and portfolio review",
                source="scheduler"
            )

            # 9. Store market condition summary
            if analysis_results['results'].get('market_performance'):
                # Use representative stocks for indices
                rep_data = analysis_results['results']['market_performance']
                nifty_proxy = rep_data.get('RELIANCE', {}).get('current_price', 0)
                sensex_proxy = rep_data.get('TCS', {}).get('current_price', 0)

                self.memory.store_market_condition(
                    nifty_value=nifty_proxy,
                    sensex_value=sensex_proxy,
                    usd_inr_rate=analysis_results['results'].get('usd_inr', {}).get('current_rate', 83.0),
                    market_sentiment="neutral"  # Could be enhanced with sentiment analysis
                )

            logger.info(f"Evening analysis completed. Briefing sent: {briefing_sent}")

        except Exception as e:
            logger.error(f"Error in evening analysis job: {e}")
            await self.notifications.send_notification(
                f"❌ Evening analysis job failed: {str(e)}",
                "error"
            )
        finally:
            self.running_jobs.discard(job_id)

    async def periodic_monitoring_job(self):
        """Periodic monitoring during market hours."""
        job_id = f"monitoring_{int(datetime.now().timestamp())}"
        self.running_jobs.add(job_id)

        try:
            logger.info("📊 Running periodic monitoring")

            # Quick check of key stocks for significant moves
            key_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
            result = self.tools.get_multiple_stocks(key_stocks)

            if result.success:
                alerts = []
                for symbol, data in result.data.items():
                    change_pct = data.get('change_percent', 0)
                    volume_ratio = data.get('volume', 0) / max(data.get('avg_volume', 1), 1)

                    # Check for significant price moves
                    if abs(change_pct) > self.config.alert_threshold:
                        alerts.append(f"{symbol}: {change_pct:+.1f}%")

                    # Check for volume spikes
                    if volume_ratio > 2.0:  # 2x normal volume
                        alerts.append(f"{symbol}: Volume spike {volume_ratio:.1f}x")

                # Send alerts if any
                if alerts:
                    alert_message = "🚨 Market Monitoring Alert:\n" + "\n".join(alerts)
                    await self.notifications.send_notification(alert_message, "warning")

        except Exception as e:
            logger.error(f"Error in periodic monitoring: {e}")
        finally:
            self.running_jobs.discard(job_id)

    async def one_time_analysis_job(self, analysis_type: str, symbols: List[str]):
        """Run a one-time analysis job."""
        job_id = f"onetime_{analysis_type}_{int(datetime.now().timestamp())}"

        try:
            logger.info(f"Running one-time analysis: {analysis_type}")

            if analysis_type == "stock_screening":
                # Screen provided symbols
                result = self.tools.get_multiple_stocks(symbols)
                if result.success:
                    # Send results
                    message = f"📊 Stock Screening Results:\n"
                    for symbol, data in result.data.items():
                        message += f"{symbol}: ₹{data['current_price']:.2f} ({data['change_percent']:+.1f}%)\n"

                    await self.notifications.send_notification(message, "info")

        except Exception as e:
            logger.error(f"Error in one-time analysis {analysis_type}: {e}")

    def start_scheduler(self):
        """Start the scheduler."""
        if not self.config.scheduler_enabled:
            logger.info("Scheduler is disabled in configuration")
            return

        try:
            # Schedule jobs
            self.schedule_morning_run()
            self.schedule_evening_run()
            self.schedule_periodic_monitoring()

            # Start scheduler
            self.scheduler.start()
            logger.info("🚀 Scheduler started successfully")

            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down gracefully...")
                self.stop_scheduler()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise

    def stop_scheduler(self):
        """Stop the scheduler gracefully."""
        try:
            if self.scheduler and self.scheduler.running:
                logger.info("Stopping scheduler...")
                self.scheduler.shutdown(wait=True)
                logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """Get list of scheduled jobs."""
        if not self.scheduler:
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time,
                'trigger': str(job.trigger),
                'function': job.func.__name__
            })

        return jobs

    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job execution history."""
        return self.job_history[-limit:] if self.job_history else []

# Async runner for integration with the main agent
async def run_scheduled_agent():
    """Main async function to run the scheduled agent."""
    try:
        # Load configuration
        config = load_config()

        # Initialize components
        memory_manager = MemoryManager(config.database_path)
        notification_manager = NotificationManager(config)
        tools = StockMarketTools(config, memory_manager, notification_manager)

        # Initialize scheduler (agent_runner will be set separately if needed)
        scheduler = AgentScheduler(
            config=config,
            memory_manager=memory_manager,
            notification_manager=notification_manager,
            tools=tools
        )

        # Start scheduler
        scheduler.start_scheduler()

        # Send startup notification
        if notification_manager.enabled:
            await notification_manager.send_notification(
                "🚀 Indian Stock Market AI Agent started!\n\n"
                f"Morning briefings: {config.morning_run_time} IST\n"
                f"Evening briefings: {config.evening_run_time} IST\n"
                f"Monitoring: Market hours (Mon-Fri)",
                "success"
            )

        # Keep the scheduler running
        logger.info("Agent scheduler is running. Press Ctrl+C to stop.")

        # Run forever
        while True:
            await asyncio.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Error in scheduled agent: {e}")
        raise
    finally:
        if 'scheduler' in locals():
            scheduler.stop_scheduler()

if __name__ == "__main__":
    # Run the scheduled agent
    print("🕐 Indian Stock Market AI Agent - Scheduler")
    print("=" * 50)

    try:
        asyncio.run(run_scheduled_agent())
    except KeyboardInterrupt:
        print("\n👋 Scheduler stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
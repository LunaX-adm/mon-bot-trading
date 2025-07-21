#!/usr/bin/env python3
"""
🤖 LunaX - AI Trading Bot Hedge Fund Level
Main orchestrator for all AI agents and trading logic
"""

import asyncio
import logging
import os
import sys
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from bot.agents.social_scanner import SocialScannerAgent
from bot.agents.technical_analyzer import TechnicalAnalyzerAgent
from bot.agents.fundamental_analyzer import FundamentalAnalyzerAgent
from bot.agents.anti_manipulation import AntiManipulationAgent
from bot.agents.meta_learning import MetaLearningAgent
from bot.agents.signal_filter import SignalFilterAgent
from bot.agents.trade_executor import TradeExecutorAgent
from bot.agents.telegram_notifier import TelegramNotifierAgent
from bot.utils.logger import setup_logger
from bot.utils.config

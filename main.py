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
from bot.agents.arbitrage_hunter import ArbitrageHunterAgent
from bot.agents.whale_tracker import WhaleTrackerAgent
from bot.agents.correlation_analyzer import CorrelationAnalyzerAgent
from bot.agents.liquidation_hunter import LiquidationHunterAgent
from bot.agents.market_maker_mimic import MarketMakerMimicAgent
from bot.agents.sentiment_manipulation_detector import SentimentManipulationDetectorAgent
from bot.agents.cross_chain_arbitrage import CrossChainArbitrageAgent
from bot.agents.macro_predictor import MacroPredictorAgent
from bot.utils.logger import setup_logger
from bot.utils.config import load_config

@dataclass
class TradingSignal:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    confidence: float
    risk_reward: float
    strategy: str
    reasoning: str
    timestamp: datetime

class LunaXBot:
    """
    🧠 LunaX AI Trading Bot - Orchestrateur Principal
    
    Coordonne tous les agents IA pour générer des signaux de trading
    ultra-filtrés avec un taux de réussite maximal.
    """
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger()
        self.is_running = False
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        self._initialize_agents()
        
        self.logger.info("🤖 LunaX Bot initialisé - Prêt pour le trading niveau hedge fund")
    
    def _initialize_agents(self):
        """Initialise tous les agents IA spécialisés"""
        try:
            self.social_scanner = SocialScannerAgent(self.config)
            self.technical_analyzer = TechnicalAnalyzerAgent(self.config)
            self.fundamental_analyzer = FundamentalAnalyzerAgent(self.config)
            self.anti_manipulation = AntiManipulationAgent(self.config)
            self.meta_learning = MetaLearningAgent(self.config)
            self.signal_filter = SignalFilterAgent(self.config)
            self.trade_executor = TradeExecutorAgent(self.config)
            self.telegram_notifier = TelegramNotifierAgent(self.config)
            
            self.arbitrage_hunter = ArbitrageHunterAgent(self.config)
            self.whale_tracker = WhaleTrackerAgent(self.config)
            self.correlation_analyzer = CorrelationAnalyzerAgent(self.config)
            self.liquidation_hunter = LiquidationHunterAgent(self.config)
            self.market_maker_mimic = MarketMakerMimicAgent(self.config)
            self.sentiment_manipulation_detector = SentimentManipulationDetectorAgent(self.config)
            self.cross_chain_arbitrage = CrossChainArbitrageAgent(self.config)
            self.macro_predictor = MacroPredictorAgent(self.config)
            
            self.logger.info("✅ Tous les agents IA initialisés avec succès (16 agents total)")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation agents: {e}")
            raise
    
    async def start(self):
        """Démarre le bot en mode autonome 24h/24"""
        self.logger.info("🚀 Démarrage LunaX Bot - Mode Autonome Activé")
        self.is_running = True
        
        while self.is_running:
            try:
                await self._main_trading_cycle()
                
                scan_interval = self.config['bot']['scan_interval_minutes']
                await asyncio.sleep(scan_interval * 60)
                
            except Exception as e:
                self.logger.error(f"❌ Erreur dans le cycle principal: {e}")
                await asyncio.sleep(300)  # Attendre 5 min avant retry
    
    async def _main_trading_cycle(self):
        """Cycle principal d'analyse et de trading"""
        self.logger.info("🔍 Début du cycle d'analyse de marché")
        
        if not self._can_trade_today():
            self.logger.info("📊 Limite quotidienne de trades atteinte")
            return
        
        market_data = await self._scan_markets()
        
        potential_signals = await self._generate_signals(market_data)
        
        filtered_signals = await self._filter_signals(potential_signals)
        
        best_signal = await self._select_best_signal(filtered_signals)
        
        if best_signal:
            await self._execute_trade(best_signal)
        
        await self._daily_optimization()
    
    async def _scan_markets(self) -> Dict:
        """Scanner tous les marchés avec les agents IA"""
        self.logger.info("📡 Scan des marchés en cours...")
        
        tasks = []
        
        if self.config['ai_agents']['social_scanner']['enabled']:
            tasks.append(self.social_scanner.scan())
        
        if self.config['ai_agents']['technical_analyzer']['enabled']:
            tasks.append(self.technical_analyzer.analyze())
        
        if self.config['ai_agents']['fundamental_analyzer']['enabled']:
            tasks.append(self.fundamental_analyzer.analyze())
        
        if self.config['ai_agents']['anti_manipulation']['enabled']:
            tasks.append(self.anti_manipulation.scan())
        
        if self.config['ai_agents'].get('arbitrage_hunter', {}).get('enabled', True):
            tasks.append(self.arbitrage_hunter.analyze())
        
        if self.config['ai_agents'].get('whale_tracker', {}).get('enabled', True):
            tasks.append(self.whale_tracker.analyze())
        
        if self.config['ai_agents'].get('correlation_analyzer', {}).get('enabled', True):
            tasks.append(self.correlation_analyzer.analyze())
        
        if self.config['ai_agents'].get('liquidation_hunter', {}).get('enabled', True):
            tasks.append(self.liquidation_hunter.analyze())
        
        if self.config['ai_agents'].get('market_maker_mimic', {}).get('enabled', True):
            tasks.append(self.market_maker_mimic.analyze())
        
        if self.config['ai_agents'].get('sentiment_manipulation_detector', {}).get('enabled', True):
            tasks.append(self.sentiment_manipulation_detector.analyze())
        
        if self.config['ai_agents'].get('cross_chain_arbitrage', {}).get('enabled', True):
            tasks.append(self.cross_chain_arbitrage.analyze())
        
        if self.config['ai_agents'].get('macro_predictor', {}).get('enabled', True):
            tasks.append(self.macro_predictor.analyze())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {
            'social': {},
            'technical': {},
            'fundamental': {},
            'manipulation': {},
            'arbitrage': {},
            'whale_tracking': {},
            'correlations': {},
            'liquidations': {},
            'market_making': {},
            'manipulation_detection': {},
            'cross_chain': {},
            'macro': {},
            'timestamp': datetime.now()
        }
        
        result_index = 0
        
        if self.config['ai_agents']['social_scanner']['enabled']:
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['social'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents']['technical_analyzer']['enabled']:
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['technical'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents']['fundamental_analyzer']['enabled']:
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['fundamental'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents']['anti_manipulation']['enabled']:
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['manipulation'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('arbitrage_hunter', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['arbitrage'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('whale_tracker', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['whale_tracking'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('correlation_analyzer', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['correlations'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('liquidation_hunter', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['liquidations'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('market_maker_mimic', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['market_making'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('sentiment_manipulation_detector', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['manipulation_detection'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('cross_chain_arbitrage', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['cross_chain'] = results[result_index]
            result_index += 1
        
        if self.config['ai_agents'].get('macro_predictor', {}).get('enabled', True):
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                market_data['macro'] = results[result_index]
            result_index += 1
        
        self.logger.info("✅ Scan des marchés terminé")
        return market_data
    
    async def _generate_signals(self, market_data: Dict) -> List[TradingSignal]:
        """Génère des signaux de trading basés sur les données"""
        self.logger.info("🧠 Génération des signaux IA...")
        
        signals = []
        
        for symbol in self.config['trading']['markets']:
            try:
                signal = await self._analyze_symbol(symbol, market_data)
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"❌ Erreur analyse {symbol}: {e}")
        
        self.logger.info(f"📊 {len(signals)} signaux générés")
        return signals
    
    async def _analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[TradingSignal]:
        """Analyse complète d'un symbole avec intégration des 31 stratégies"""
        
        technical_data = market_data.get('technical', {}).get(symbol, {})
        if not technical_data:
            return None
        
        strategy_weights = self._load_strategy_weights()
        
        strategy_scores = {}
        total_weighted_score = 0
        
        for strategy_name, config in strategy_weights.items():
            if config.get('enabled', False):
                strategy_score = technical_data.get('strategies', {}).get(strategy_name, {}).get('score', 0)
                weight = config.get('weight', 0)
                weighted_score = strategy_score * weight
                strategy_scores[strategy_name] = weighted_score
                total_weighted_score += weighted_score
        
        scores = {
            'technical': total_weighted_score,
            'social': market_data.get('social', {}).get(symbol, {}).get('score', 0),
            'fundamental': market_data.get('fundamental', {}).get(symbol, {}).get('score', 0),
            'manipulation': market_data.get('manipulation', {}).get(symbol, {}).get('score', 0),
            'arbitrage': market_data.get('arbitrage', {}).get(symbol, {}).get('score', 0),
            'whale_tracking': market_data.get('whale_tracking', {}).get(symbol, {}).get('score', 0),
            'correlations': market_data.get('correlations', {}).get(symbol, {}).get('score', 0),
            'liquidations': market_data.get('liquidations', {}).get(symbol, {}).get('score', 0)
        }
        
        total_score = (
            scores['technical'] * 0.35 +
            scores['social'] * 0.15 +
            scores['fundamental'] * 0.20 +
            scores['manipulation'] * 0.05 +
            scores['arbitrage'] * 0.10 +
            scores['whale_tracking'] * 0.08 +
            scores['correlations'] * 0.04 +
            scores['liquidations'] * 0.03
        )
        
        if total_score < self.config['bot']['min_confidence_score']:
            return None
        
        direction = technical_data.get('direction', 'LONG')
        entry_price = technical_data.get('entry_price', 0)
        stop_loss = technical_data.get('stop_loss', 0)
        take_profit = technical_data.get('take_profit', 0)
        
        if direction == "LONG":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < self.config['bot']['min_risk_reward_ratio']:
            return None
        
        leverage = self._calculate_optimal_leverage(symbol, technical_data)
        
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0] if strategy_scores else 'composite'
        
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            confidence=total_score,
            risk_reward=risk_reward,
            strategy=best_strategy,
            reasoning=technical_data.get('reasoning', ''),
            timestamp=datetime.now()
        )
    
    def _calculate_optimal_leverage(self, symbol: str, technical_data: Dict) -> int:
        """Calcule le levier optimal basé sur la volatilité et le setup"""
        base_leverage = 10
        max_leverage = self.config['bot']['max_leverage']
        
        volatility = technical_data.get('volatility', 0.02)
        if volatility < 0.01:
            leverage_multiplier = 2.0
        elif volatility < 0.03:
            leverage_multiplier = 1.5
        else:
            leverage_multiplier = 1.0
        
        setup_quality = technical_data.get('setup_quality', 0.5)
        quality_multiplier = 1 + setup_quality
        
        optimal_leverage = int(base_leverage * leverage_multiplier * quality_multiplier)
        
        return min(optimal_leverage, max_leverage)
    
    async def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filtrage ultra-strict des signaux"""
        if not signals:
            return []
        
        self.logger.info(f"🔍 Filtrage de {len(signals)} signaux...")
        
        filtered = []
        
        for signal in signals:
            is_valid = await self.signal_filter.validate_signal(signal)
            
            if is_valid:
                filtered.append(signal)
        
        self.logger.info(f"✅ {len(filtered)} signaux passent les filtres")
        return filtered
    
    async def _select_best_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Sélectionne le meilleur signal parmi les candidats"""
        if not signals:
            return None
        
        signals.sort(key=lambda s: s.confidence * s.risk_reward, reverse=True)
        
        best_signal = signals[0]
        
        self.logger.info(f"🎯 Meilleur signal sélectionné: {best_signal.symbol} {best_signal.direction}")
        self.logger.info(f"   Confiance: {best_signal.confidence:.1f}% | R:R: 1:{best_signal.risk_reward:.1f}")
        
        return best_signal
    
    async def _execute_trade(self, signal: TradingSignal):
        """Exécute le trade sélectionné"""
        try:
            self.logger.info(f"🚀 Exécution du trade: {signal.symbol} {signal.direction}")
            
            success = await self.trade_executor.execute_trade(signal)
            
            if success:
                await self.telegram_notifier.send_signal(signal)
                
                self._update_daily_count()
                
                self.logger.info("✅ Trade exécuté avec succès")
            else:
                self.logger.error("❌ Échec de l'exécution du trade")
                
        except Exception as e:
            self.logger.error(f"❌ Erreur exécution trade: {e}")
    
    async def _daily_optimization(self):
        """Optimisation quotidienne via l'agent meta-learning"""
        if self.config['ai_agents']['meta_learning']['enabled']:
            try:
                await self.meta_learning.daily_optimization()
                self.logger.info("🧠 Optimisation quotidienne terminée")
            except Exception as e:
                self.logger.error(f"❌ Erreur optimisation: {e}")
    
    def _can_trade_today(self) -> bool:
        """Vérifie si on peut encore trader aujourd'hui"""
        today = datetime.now().date()
        
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
        
        max_trades = self.config['bot']['max_trades_per_day']
        return self.daily_trades_count < max_trades
    
    def _update_daily_count(self):
        """Met à jour le compteur de trades quotidiens"""
        self.daily_trades_count += 1
        self.last_trade_date = datetime.now().date()
    
    def _load_strategy_weights(self) -> Dict:
        """Charge les poids des stratégies depuis strategies.json"""
        try:
            import json
            with open('config/strategies.json', 'r') as f:
                config = json.load(f)
                return config.get('strategies', {})
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement strategies.json: {e}")
            return {}
    
    def stop(self):
        """Arrête le bot proprement"""
        self.logger.info("🛑 Arrêt de LunaX Bot")
        self.is_running = False

async def main():
    """Point d'entrée principal"""
    bot = LunaXBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\n🛑 LunaX Bot arrêté par l'utilisateur")
    except Exception as e:
        logging.error(f"❌ Erreur fatale: {e}")
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..strategies.trend_volatility_breakout import TrendVolatilityBreakoutStrategy
from ..strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from ..strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from ..strategies.hybrid_trend_reversion import HybridTrendReversionStrategy
from ..strategies.vwap_mean_reversion_scalper import VWAPMeanReversionScalperStrategy
from ..strategies.trend_atr_breakout import TrendATRBreakoutStrategy
from ..strategies.opening_range_breakout_orb import OpeningRangeBreakoutORBStrategy
from ..strategies.bollinger_squeeze_expansion import BollingerSqueezeExpansionStrategy
from ..strategies.rsi_pullback_trend import RSIPullbackTrendStrategy
from ..strategies.donchian_channel_breakout import DonchianChannelBreakoutStrategy
from ..strategies.macd_adx_filter import MACDADXFilterStrategy
from ..strategies.breakout_pullback_continuation import BreakoutPullbackContinuationStrategy
from ..strategies.heikin_ashi_trend_ride import HeikinAshiTrendRideStrategy
from ..strategies.volume_spike_reversal import VolumeSpikeReversalStrategy

logger = logging.getLogger(__name__)

class StrategyEngine:
    """Coordinates and executes all trading strategies."""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
        
    def initialize_strategies(self, strategy_params: Dict[str, Dict[str, Any]] = None):
        """Initialize all strategies with optional custom parameters."""
        default_params = strategy_params or {}
        
        self.strategies = {
            # Original strategies
            'trend_volatility_breakout': TrendVolatilityBreakoutStrategy(
                default_params.get('trend_volatility_breakout', {})
            ),
            'vwap_mean_reversion': VWAPMeanReversionStrategy(
                default_params.get('vwap_mean_reversion', {})
            ),
            'opening_range_breakout': OpeningRangeBreakoutStrategy(
                default_params.get('opening_range_breakout', {})
            ),
            'hybrid_trend_reversion': HybridTrendReversionStrategy(
                default_params.get('hybrid_trend_reversion', {})
            ),
            
            # New strategies
            'vwap_mean_reversion_scalper': VWAPMeanReversionScalperStrategy(
                default_params.get('vwap_mean_reversion_scalper', {})
            ),
            'trend_atr_breakout': TrendATRBreakoutStrategy(
                default_params.get('trend_atr_breakout', {})
            ),
            'opening_range_breakout_orb': OpeningRangeBreakoutORBStrategy(
                default_params.get('opening_range_breakout_orb', {})
            ),
            'bollinger_squeeze_expansion': BollingerSqueezeExpansionStrategy(
                default_params.get('bollinger_squeeze_expansion', {})
            ),
            'rsi_pullback_trend': RSIPullbackTrendStrategy(
                default_params.get('rsi_pullback_trend', {})
            ),
            'donchian_channel_breakout': DonchianChannelBreakoutStrategy(
                default_params.get('donchian_channel_breakout', {})
            ),
            'macd_adx_filter': MACDADXFilterStrategy(
                default_params.get('macd_adx_filter', {})
            ),
            'breakout_pullback_continuation': BreakoutPullbackContinuationStrategy(
                default_params.get('breakout_pullback_continuation', {})
            ),
            'heikin_ashi_trend_ride': HeikinAshiTrendRideStrategy(
                default_params.get('heikin_ashi_trend_ride', {})
            ),
            'volume_spike_reversal': VolumeSpikeReversalStrategy(
                default_params.get('volume_spike_reversal', {})
            )
        }
    
    def run_strategy(self, strategy_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a single strategy and return results."""
        try:
            strategy = self.strategies[strategy_name]
            
            # Execute trades directly (no need for generate_signals)
            trades = strategy.execute_trades(df)
            
            # Calculate performance metrics
            metrics = strategy.get_performance_metrics(trades)
            
            return {
                'strategy_name': strategy_name,
                'trades': trades,
                'metrics': metrics,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_name}: {e}")
            return {
                'strategy_name': strategy_name,
                'trades': [],
                'metrics': {},
                'error': str(e)
            }
    
    def run_all_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all strategies in parallel and return combined results."""
        results = {}
        
        # Run strategies in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_strategy = {
                executor.submit(self.run_strategy, name, df): name 
                for name in self.strategies.keys()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} failed: {e}")
                    results[strategy_name] = {
                        'strategy_name': strategy_name,
                        'trades': [],
                        'metrics': {},
                        'error': str(e)
                    }
        
        # Calculate comparative analysis
        comparative_analysis = self._calculate_comparative_analysis(results, df)
        
        return {
            'individual_results': results,
            'comparative_analysis': comparative_analysis,
            'summary': self._generate_summary(results)
        }
    
    def _calculate_comparative_analysis(self, results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comparative analysis across all strategies."""
        analysis = {
            'performance_ranking': [],
            'best_strategy': None,
            'worst_strategy': None,
            'total_trades': 0,
            'overall_win_rate': 0,
            'total_initial_capital': 0,
            'total_closing_capital': 0,
            'overall_return_pct': 0
        }
        
        # Collect all trades and strategy performance
        all_trades = []
        strategy_performance = {}
        total_initial = 0
        total_closing = 0
        
        for strategy_name, result in results.items():
            # Include all strategies, even those with no trades
            metrics = result.get('metrics', {})
            initial_capital = metrics.get('initial_capital', 100.0)
            closing_capital = metrics.get('closing_capital', 100.0)
            
            total_initial += initial_capital
            total_closing += closing_capital
            
            if 'trades' in result and result['trades']:
                trades = result['trades']
                all_trades.extend(trades)
                
                # Calculate strategy performance
                df_trades = pd.DataFrame(trades)
                strategy_performance[strategy_name] = {
                    'total_trades': len(trades),
                    'win_rate': len(df_trades[df_trades['profit'] > 0]) / len(trades),
                    'total_profit': df_trades['profit'].sum(),
                    'avg_profit': df_trades['profit'].mean(),
                    'avg_r_multiple': df_trades['r_multiple'].mean(),
                    'initial_capital': initial_capital,
                    'closing_capital': closing_capital,
                    'return_pct': metrics.get('total_return_pct', 0)
                }
            else:
                # Include strategies with no trades
                strategy_performance[strategy_name] = {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit': 0,
                    'avg_r_multiple': 0,
                    'initial_capital': initial_capital,
                    'closing_capital': closing_capital,
                    'return_pct': metrics.get('total_return_pct', 0)
                }
        
        # Calculate overall metrics
        analysis['total_trades'] = len(all_trades)
        analysis['total_initial_capital'] = total_initial
        analysis['total_closing_capital'] = total_closing
        analysis['overall_return_pct'] = ((total_closing - total_initial) / total_initial) * 100 if total_initial > 0 else 0
        
        if all_trades:
            df_all_trades = pd.DataFrame(all_trades)
            winning_trades = len(df_all_trades[df_all_trades['profit'] > 0])
            analysis['overall_win_rate'] = winning_trades / len(all_trades)
        
        # Rank strategies by return percentage
        strategy_ranking = sorted(
            strategy_performance.items(),
            key=lambda x: x[1]['return_pct'],
            reverse=True
        )
        
        analysis['performance_ranking'] = [
            {
                'strategy': name,
                'return_pct': perf['return_pct'],
                'total_trades': perf['total_trades'],
                'win_rate': perf['win_rate']
            }
            for name, perf in strategy_ranking
        ]
        
        if strategy_ranking:
            analysis['best_strategy'] = strategy_ranking[0][0]
            analysis['worst_strategy'] = strategy_ranking[-1][0]
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_strategies': len(results),
            'successful_strategies': 0,
            'failed_strategies': 0,
            'total_trades': 0,
            'best_performer': None,
            'worst_performer': None
        }
        
        successful_strategies = []
        failed_strategies = []
        
        for strategy_name, result in results.items():
            if 'error' in result:
                failed_strategies.append(strategy_name)
                summary['failed_strategies'] += 1
            else:
                successful_strategies.append(strategy_name)
                summary['successful_strategies'] += 1
                
                trades = result.get('trades', [])
                summary['total_trades'] += len(trades)
        
        # Find best and worst performers
        if successful_strategies:
            best_return = -float('inf')
            worst_return = float('inf')
            
            for strategy_name in successful_strategies:
                result = results[strategy_name]
                metrics = result.get('metrics', {})
                return_pct = metrics.get('total_return_pct', 0)
                
                if return_pct > best_return:
                    best_return = return_pct
                    summary['best_performer'] = strategy_name
                
                if return_pct < worst_return:
                    worst_return = return_pct
                    summary['worst_performer'] = strategy_name
        
        return summary
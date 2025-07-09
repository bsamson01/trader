import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..strategies.trend_volatility_breakout import TrendVolatilityBreakoutStrategy
from ..strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from ..strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from ..strategies.hybrid_trend_reversion import HybridTrendReversionStrategy

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
            )
        }
    
    def run_strategy(self, strategy_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a single strategy and return results."""
        try:
            strategy = self.strategies[strategy_name]
            
            # Generate signals
            df_with_signals = strategy.generate_signals(df)
            
            # Execute trades
            trades = strategy.execute_trades(df_with_signals)
            
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
        with ThreadPoolExecutor(max_workers=4) as executor:
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
                    'return_pct': 0
                }
        
        # Rank strategies by total return percentage
        if strategy_performance:
            sorted_strategies = sorted(
                strategy_performance.items(),
                key=lambda x: x[1]['return_pct'],
                reverse=True
            )
            
            analysis['performance_ranking'] = [name for name, _ in sorted_strategies]
            analysis['best_strategy'] = sorted_strategies[0][0] if sorted_strategies else None
            analysis['worst_strategy'] = sorted_strategies[-1][0] if sorted_strategies else None
        
        # Calculate overall statistics
        analysis['total_initial_capital'] = total_initial
        analysis['total_closing_capital'] = total_closing
        analysis['overall_return_pct'] = ((total_closing - total_initial) / total_initial * 100) if total_initial > 0 else 0
        
        if all_trades:
            df_all_trades = pd.DataFrame(all_trades)
            analysis['total_trades'] = len(all_trades)
            analysis['overall_win_rate'] = len(df_all_trades[df_all_trades['profit'] > 0]) / len(all_trades)
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all strategy results."""
        summary = {
            'total_strategies': len(results),
            'strategies_with_trades': 0,
            'total_trades': 0,
            'total_initial_capital': 0,
            'total_closing_capital': 0,
            'best_performing': None,
            'best_return_pct': float('-inf')
        }
        
        for strategy_name, result in results.items():
            metrics = result.get('metrics', {})
            initial_capital = metrics.get('initial_capital', 100.0)
            closing_capital = metrics.get('closing_capital', 100.0)
            return_pct = metrics.get('total_return_pct', 0)
            
            summary['total_initial_capital'] += initial_capital
            summary['total_closing_capital'] += closing_capital
            
            if 'trades' in result and result['trades']:
                summary['strategies_with_trades'] += 1
                summary['total_trades'] += len(result['trades'])
                
                if return_pct > summary['best_return_pct']:
                    summary['best_return_pct'] = return_pct
                    summary['best_performing'] = strategy_name
        
        return summary
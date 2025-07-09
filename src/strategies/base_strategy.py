from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.trades = []
        self.current_position = None
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals for the strategy.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with signal columns added
        """
        pass
    
    @abstractmethod
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Check if we should enter a position.
        
        Args:
            row: Current data row
            
        Returns:
            Entry signal dict or None
        """
        pass
    
    @abstractmethod
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """
        Check if we should exit a position.
        
        Args:
            row: Current data row
            position: Current position information
            
        Returns:
            True if should exit, False otherwise
        """
        pass
    
    def execute_trades(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Execute the strategy and return trade list.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        current_position = None
        
        for i, row in df.iterrows():
            # Check for exit if we have a position
            if current_position and self.should_exit(row, current_position):
                # Close position
                exit_price = row['close']
                profit = (exit_price - current_position['entry_price']) * current_position['side']
                
                trade = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': row['time'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'side': current_position['side'],
                    'profit': profit,
                    'r_multiple': profit / (current_position['atr'] * 2),  # Assuming 2 ATR stop
                    'duration': (row['time'] - current_position['entry_time']).total_seconds() / 60,  # minutes
                    'strategy': self.name,
                    'volatility_at_entry': current_position['atr'],
                    'signal_confidence': current_position.get('confidence', 0.5)
                }
                trades.append(trade)
                current_position = None
            
            # Check for entry if we don't have a position
            elif not current_position:
                entry_signal = self.should_entry(row)
                if entry_signal:
                    current_position = {
                        'entry_time': row['time'],
                        'entry_price': row['close'],
                        'side': entry_signal['side'],
                        'atr': row.get('atr_14', 0.001),
                        'confidence': entry_signal.get('confidence', 0.5)
                    }
        
        return trades
    
    def get_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_r_multiple': 0,
                'max_drawdown': 0,
                'avg_duration': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        avg_profit = df_trades['profit'].mean()
        total_profit = df_trades['profit'].sum()
        
        # R-multiple metrics
        avg_r_multiple = df_trades['r_multiple'].mean()
        
        # Duration metrics
        avg_duration = df_trades['duration'].mean()
        
        # Drawdown calculation
        cumulative_profit = df_trades['profit'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = cumulative_profit - running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'avg_r_multiple': avg_r_multiple,
            'max_drawdown': max_drawdown,
            'avg_duration': avg_duration,
            'profit_factor': abs(df_trades[df_trades['profit'] > 0]['profit'].sum() / 
                               df_trades[df_trades['profit'] < 0]['profit'].sum()) if len(df_trades[df_trades['profit'] < 0]) > 0 else float('inf')
        }
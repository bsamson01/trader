import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class BreakoutPullbackContinuationStrategy(BaseStrategy):
    """
    Breakout-Pullback Continuation Strategy
    
    Edge: Most breakouts retest; entering on the pullback improves R/R and win-rate.
    
    Implementation:
    - Detect breakout candle (range > 1.5× ATR, closes beyond resistance)
    - Place limit order at 38–62 % Fibonacci retrace of breakout candle with stop beyond 78 %
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("breakout_pullback_continuation", params or {})
        self.description = "Breakout-Pullback Continuation Strategy"
        
        # Strategy parameters
        self.atr_period = self.params.get('atr_period', 14)
        self.breakout_multiplier = self.params.get('breakout_multiplier', 1.5)
        self.fib_38 = self.params.get('fib_38', 0.382)
        self.fib_62 = self.params.get('fib_62', 0.618)
        self.fib_78 = self.params.get('fib_78', 0.786)
        self.max_bars_to_fill = self.params.get('max_bars_to_fill', 10)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on breakout pullback continuation."""
        df = df.copy()
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate candle range
        df['candle_range'] = df['high'] - df['low']
        
        # Detect breakout candles
        df['is_breakout'] = (
            (df['candle_range'] > self.breakout_multiplier * df['atr']) &
            (df['close'] > df['high'].shift(1))  # Breakout above previous high
        )
        
        # Calculate Fibonacci levels for breakout candles
        df['breakout_high'] = np.nan
        df['breakout_low'] = np.nan
        df['fib_38_level'] = np.nan
        df['fib_62_level'] = np.nan
        df['fib_78_level'] = np.nan
        
        for i in range(len(df)):
            if df.iloc[i]['is_breakout']:
                breakout_high = df.iloc[i]['high']
                breakout_low = df.iloc[i]['low']
                breakout_range = breakout_high - breakout_low
                
                # Set Fibonacci levels
                df.iloc[i, df.columns.get_loc('breakout_high')] = breakout_high
                df.iloc[i, df.columns.get_loc('breakout_low')] = breakout_low
                df.iloc[i, df.columns.get_loc('fib_38_level')] = breakout_high - self.fib_38 * breakout_range
                df.iloc[i, df.columns.get_loc('fib_62_level')] = breakout_high - self.fib_62 * breakout_range
                df.iloc[i, df.columns.get_loc('fib_78_level')] = breakout_high - self.fib_78 * breakout_range
        
        # Forward fill Fibonacci levels
        df['fib_38_level'] = df['fib_38_level'].fillna(method='ffill')
        df['fib_62_level'] = df['fib_62_level'].fillna(method='ffill')
        df['fib_78_level'] = df['fib_78_level'].fillna(method='ffill')
        
        # Generate signals (entry at Fibonacci retracement)
        df['long_signal'] = (
            (df['close'] >= df['fib_62_level']) &
            (df['close'] <= df['fib_38_level']) &
            (df['high'].shift(1) > df['fib_78_level'])  # Previous high above 78% level
        )
        
        # Calculate stop loss and take profit
        df['stop_loss'] = df['fib_78_level']
        df['take_profit'] = df['breakout_high'] + (df['breakout_high'] - df['fib_38_level'])
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.8,
                'reason': 'breakout_pullback',
                'stop_loss': row['stop_loss'],
                'take_profit': row['take_profit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price hits stop loss or take profit
            return (row['close'] <= position.get('stop_loss', 0) or 
                   row['close'] >= position.get('take_profit', float('inf')))
        return False
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
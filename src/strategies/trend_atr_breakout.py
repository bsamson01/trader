import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class TrendATRBreakoutStrategy(BaseStrategy):
    """
    Trend + ATR Breakout Strategy
    
    Edge: Trade only when volatility expands with the prevailing trend.
    
    Implementation:
    - Trend filter: price above EMA200 (long bias) or below (short)
    - Entry on close > previous high + 0.5 × ATR (inverse for shorts)
    - ATR-based trailing stop
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("trend_atr_breakout", params)
        self.description = "Trend + ATR Breakout Strategy"
        
        # Strategy parameters
        self.ema_period = self.params.get('ema_period', 200)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 0.5)
        self.trailing_stop_multiplier = self.params.get('trailing_stop_multiplier', 2.0)
        self.max_consecutive_losses = self.params.get('max_consecutive_losses', 3)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on trend and ATR breakout."""
        df = df.copy()
        
        # Calculate EMA200
        df['ema_200'] = df['close'].ewm(span=self.ema_period).mean()
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate previous highs and lows
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        
        # Determine trend bias
        df['trend_bias'] = np.where(df['close'] > df['ema_200'], 'bullish', 'bearish')
        
        # Calculate breakout levels
        df['long_breakout_level'] = df['prev_high'] + self.atr_multiplier * df['atr']
        df['short_breakout_level'] = df['prev_low'] - self.atr_multiplier * df['atr']
        
        # Generate signals
        df['long_signal'] = (
            (df['trend_bias'] == 'bullish') &
            (df['close'] > df['long_breakout_level'])
        )
        
        df['short_signal'] = (
            (df['trend_bias'] == 'bearish') &
            (df['close'] < df['short_breakout_level'])
        )
        
        # Calculate trailing stops
        df['long_trailing_stop'] = df['close'] - self.trailing_stop_multiplier * df['atr']
        df['short_trailing_stop'] = df['close'] + self.trailing_stop_multiplier * df['atr']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.7,
                'reason': 'trend_breakout',
                'trailing_stop': row['long_trailing_stop']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.7,
                'reason': 'trend_breakout',
                'trailing_stop': row['short_trailing_stop']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price hits trailing stop
            return row['close'] <= position.get('trailing_stop', 0)
        elif position['side'] == -1:  # Short position
            # Exit if price hits trailing stop
            return row['close'] >= position.get('trailing_stop', float('inf'))
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
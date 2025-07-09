import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class DonchianChannelBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel 20-Bar Breakout Strategy
    
    Edge: Classic Turtle logic—ride medium-term breakouts.
    
    Implementation:
    - Buy break of 20-bar high; initial SL at 10-bar low; scale in at +0.5 × ATR
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("donchian_channel_breakout", params or {})
        self.description = "Donchian Channel 20-Bar Breakout Strategy"
        
        # Strategy parameters
        self.breakout_period = self.params.get('breakout_period', 20)
        self.stop_period = self.params.get('stop_period', 10)
        self.atr_period = self.params.get('atr_period', 14)
        self.scale_multiplier = self.params.get('scale_multiplier', 0.5)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Donchian Channel breakout."""
        df = df.copy()
        
        # Calculate Donchian Channels
        df['donchian_high'] = df['high'].rolling(window=self.breakout_period).max()
        df['donchian_low'] = df['low'].rolling(window=self.breakout_period).min()
        df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate stop loss levels
        df['stop_loss'] = df['low'].rolling(window=self.stop_period).min()
        
        # Generate signals
        df['long_signal'] = df['close'] > df['donchian_high'].shift(1)
        df['short_signal'] = df['close'] < df['donchian_low'].shift(1)
        
        # Calculate scale-in levels
        df['scale_in_level'] = df['donchian_high'] + self.scale_multiplier * df['atr']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.7,
                'reason': 'donchian_breakout',
                'stop_loss': row['stop_loss'],
                'scale_in': row['scale_in_level']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.7,
                'reason': 'donchian_breakout',
                'stop_loss': row['donchian_high'],
                'scale_in': row['donchian_low'] - self.scale_multiplier * row['atr']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price hits stop loss
            return row['close'] <= position.get('stop_loss', 0)
        elif position['side'] == -1:  # Short position
            # Exit if price hits stop loss
            return row['close'] >= position.get('stop_loss', float('inf'))
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
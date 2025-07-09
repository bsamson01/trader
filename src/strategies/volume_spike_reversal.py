import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class VolumeSpikeReversalStrategy(BaseStrategy):
    """
    Volume Spike Reversal Strategy
    
    Edge: Extreme volume often marks capitulation; price snaps back.
    
    Implementation:
    - Identify volume >2× 20-bar average and candle range >1.5× ATR
    - Enter opposite direction on confirmation Doji/hammer; TP at midpoint of spike bar
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("volume_spike_reversal", params or {})
        self.description = "Volume Spike Reversal Strategy"
        
        # Strategy parameters
        self.volume_multiplier = self.params.get('volume_multiplier', 2.0)
        self.volume_avg_period = self.params.get('volume_avg_period', 20)
        self.atr_period = self.params.get('atr_period', 14)
        self.range_multiplier = self.params.get('range_multiplier', 1.5)
        self.doji_threshold = self.params.get('doji_threshold', 0.1)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volume spike reversal."""
        df = df.copy()
        
        # Calculate volume average
        df['volume_avg'] = df['volume'].rolling(self.volume_avg_period).mean()
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate candle range
        df['candle_range'] = df['high'] - df['low']
        
        # Detect volume spikes
        df['is_volume_spike'] = (
            (df['volume'] > self.volume_multiplier * df['volume_avg']) &
            (df['candle_range'] > self.range_multiplier * df['atr'])
        )
        
        # Detect Doji/hammer patterns
        df['is_doji'] = self._detect_doji(df)
        
        # Calculate spike bar midpoint
        df['spike_midpoint'] = (df['high'] + df['low']) / 2
        
        # Generate signals
        df['long_signal'] = (
            df['is_volume_spike'] &
            df['is_doji'] &
            (df['close'] < df['open'])  # Bearish spike, go long
        )
        
        df['short_signal'] = (
            df['is_volume_spike'] &
            df['is_doji'] &
            (df['close'] > df['open'])  # Bullish spike, go short
        )
        
        # Calculate take profit levels
        df['long_take_profit'] = df['spike_midpoint']
        df['short_take_profit'] = df['spike_midpoint']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.7,
                'reason': 'volume_spike_reversal',
                'take_profit': row['long_take_profit']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.7,
                'reason': 'volume_spike_reversal',
                'take_profit': row['short_take_profit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price reaches spike midpoint
            return row['close'] >= position.get('take_profit', float('inf'))
        elif position['side'] == -1:  # Short position
            # Exit if price reaches spike midpoint
            return row['close'] <= position.get('take_profit', 0)
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
    
    def _detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect Doji/hammer candle patterns."""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        # Doji: small body relative to total range
        is_doji = body_size <= self.doji_threshold * total_range
        
        return is_doji
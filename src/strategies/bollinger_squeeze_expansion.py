import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class BollingerSqueezeExpansionStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Expansion Strategy
    
    Edge: Volatility contraction signals an impending trend burst.
    
    Implementation:
    - Monitor 20-period BB width; fire only when width < 10-bar percentile 15
    - Enter on candle close beyond band with MACD histogram above zero (long) / below zero (short)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("bollinger_squeeze_expansion", params or {})
        self.description = "Bollinger Band Squeeze Expansion Strategy"
        
        # Strategy parameters
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2)
        self.squeeze_percentile = self.params.get('squeeze_percentile', 15)
        self.squeeze_lookback = self.params.get('squeeze_lookback', 10)
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Band squeeze expansion."""
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_middle'] - self.bb_std * bb_std
        
        # Calculate BB width
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Calculate BB width percentile
        df['bb_width_percentile'] = df['bb_width'].rolling(
            window=self.squeeze_lookback
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        
        # Calculate MACD
        df['macd'] = self._calculate_macd(df['close'], self.macd_fast, self.macd_slow)
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Detect squeeze condition
        df['is_squeeze'] = df['bb_width_percentile'] < self.squeeze_percentile
        
        # Generate signals
        df['long_signal'] = (
            df['is_squeeze'] &
            (df['close'] > df['bb_upper']) &
            (df['macd_histogram'] > 0)
        )
        
        df['short_signal'] = (
            df['is_squeeze'] &
            (df['close'] < df['bb_lower']) &
            (df['macd_histogram'] < 0)
        )
        
        # Calculate exit levels (opposite band)
        df['long_exit'] = df['bb_middle']
        df['short_exit'] = df['bb_middle']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.75,
                'reason': 'bb_expansion',
                'exit_level': row['long_exit']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.75,
                'reason': 'bb_expansion',
                'exit_level': row['short_exit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price returns to middle band
            return row['close'] <= position.get('exit_level', 0)
        elif position['side'] == -1:  # Short position
            # Exit if price returns to middle band
            return row['close'] >= position.get('exit_level', float('inf'))
        return False
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int) -> pd.Series:
        """Calculate MACD line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
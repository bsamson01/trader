import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class MACDADXFilterStrategy(BaseStrategy):
    """
    MACD Cross with ADX Filter Strategy
    
    Edge: Momentum crosses are more reliable when trend strength (ADX) is elevated.
    
    Implementation:
    - MACD(12,26,9) bullish cross and ADX(14) > 25 = long; bearish cross + ADX>25 = short
    - Exit on opposite MACD cross or +2 × ATR
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("macd_adx_filter", params or {})
        self.description = "MACD Cross with ADX Filter Strategy"
        
        # Strategy parameters
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_threshold = self.params.get('adx_threshold', 25)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)
        self.throttle_bars = self.params.get('throttle_bars', 3)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD cross with ADX filter."""
        df = df.copy()
        
        # Calculate MACD
        df['macd'] = self._calculate_macd(df['close'], self.macd_fast, self.macd_slow)
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate ADX
        df['adx'] = self._calculate_adx(df, self.adx_period)
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Detect MACD crosses
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # Generate signals
        df['long_signal'] = (
            df['macd_cross_up'] &
            (df['adx'] > self.adx_threshold)
        )
        
        df['short_signal'] = (
            df['macd_cross_down'] &
            (df['adx'] > self.adx_threshold)
        )
        
        # Calculate exit levels
        df['long_exit'] = df['close'] + self.atr_multiplier * df['atr']
        df['short_exit'] = df['close'] - self.atr_multiplier * df['atr']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.75,
                'reason': 'macd_adx_bullish',
                'take_profit': row['long_exit']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.75,
                'reason': 'macd_adx_bearish',
                'take_profit': row['short_exit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit on bearish cross or take profit
            return (row['macd_cross_down'] or 
                   row['close'] >= position.get('take_profit', float('inf')))
        elif position['side'] == -1:  # Short position
            # Exit on bullish cross or take profit
            return (row['macd_cross_up'] or 
                   row['close'] <= position.get('take_profit', 0))
        return False
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int) -> pd.Series:
        """Calculate MACD line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        plus_di = pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean() * 100
        minus_di = pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean() * 100
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
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
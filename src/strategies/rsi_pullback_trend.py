import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class RSIPullbackTrendStrategy(BaseStrategy):
    """
    RSI-2 Pullback in Trend Strategy
    
    Edge: Buy the dip inside a strong up-trend; quick snap-backs produce high hit rate.
    
    Implementation:
    - Up-trend: price > EMA200; enter when RSI(2)≤10; exit when RSI crosses 50 or +0.6 × ATR
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("rsi_pullback_trend", params or {})
        self.description = "RSI-2 Pullback in Trend Strategy"
        
        # Strategy parameters
        self.ema_period = self.params.get('ema_period', 200)
        self.rsi_period = self.params.get('rsi_period', 2)
        self.rsi_oversold = self.params.get('rsi_oversold', 10)
        self.rsi_exit = self.params.get('rsi_exit', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 0.6)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI pullback in trend."""
        df = df.copy()
        
        # Calculate EMA200
        df['ema_200'] = df['close'].ewm(span=self.ema_period).mean()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Determine trend
        df['is_uptrend'] = df['close'] > df['ema_200']
        
        # Generate signals
        df['long_signal'] = (
            df['is_uptrend'] &
            (df['rsi'] <= self.rsi_oversold)
        )
        
        # Exit signals
        df['exit_long'] = df['rsi'] >= self.rsi_exit
        
        # Calculate take profit levels
        df['take_profit'] = df['close'] + self.atr_multiplier * df['atr']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.8,
                'reason': 'rsi_pullback',
                'take_profit': row['take_profit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if RSI crosses above exit level or hits take profit
            return (row['exit_long'] or 
                   row['close'] >= position.get('take_profit', float('inf')))
        return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
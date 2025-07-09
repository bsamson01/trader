import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class VWAPMeanReversionScalperStrategy(BaseStrategy):
    """
    VWAP Mean-Reversion Scalper Strategy
    
    Edge: Price rarely stays far from VWAP intraday; exploit snap-backs.
    
    Implementation:
    - Long if price ≤ VWAP – 1.5 × ATR and RSI(2)<10
    - Short if ≥ VWAP + 1.5 × ATR and RSI(2)>90
    - Exit at VWAP touch or after N bars
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("vwap_mean_reversion_scalper", params)
        self.description = "VWAP Mean-Reversion Scalper Strategy"
        
        # Strategy parameters
        self.vwap_multiplier = self.params.get('vwap_multiplier', 1.5)
        self.rsi_period = self.params.get('rsi_period', 2)
        self.rsi_oversold = self.params.get('rsi_oversold', 10)
        self.rsi_overbought = self.params.get('rsi_overbought', 90)
        self.max_bars_in_trade = self.params.get('max_bars_in_trade', 20)
        self.atr_period = self.params.get('atr_period', 14)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on VWAP mean reversion."""
        df = df.copy()
        
        # Calculate VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate VWAP bands
        df['vwap_upper'] = df['vwap'] + self.vwap_multiplier * df['atr']
        df['vwap_lower'] = df['vwap'] - self.vwap_multiplier * df['atr']
        
        # Generate signals
        df['long_signal'] = (
            (df['close'] <= df['vwap_lower']) & 
            (df['rsi'] < self.rsi_oversold)
        )
        
        df['short_signal'] = (
            (df['close'] >= df['vwap_upper']) & 
            (df['rsi'] > self.rsi_overbought)
        )
        
        # Exit signals
        df['exit_long'] = df['close'] >= df['vwap']
        df['exit_short'] = df['close'] <= df['vwap']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.8,
                'reason': 'vwap_oversold'
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.8,
                'reason': 'vwap_overbought'
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            return row['exit_long']
        elif position['side'] == -1:  # Short position
            return row['exit_short']
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
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
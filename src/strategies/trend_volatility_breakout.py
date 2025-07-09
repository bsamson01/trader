import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class TrendVolatilityBreakoutStrategy(BaseStrategy):
    """Trend + Volatility Breakout Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'atr_period': 14,
            'adx_period': 14,
            'lookback_period': 20,
            'atr_multiplier': 1.5,
            'adx_threshold': 25,
            'stop_loss_pct': 0.8,          # 0.8% stop loss
            'take_profit_pct': 1.8         # 1.8% take profit
        }
        super().__init__("Trend Volatility Breakout", {**default_params, **(params or {})})
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals based on trend and volatility."""
        df = df.copy()
        
        # Calculate rolling highs and lows
        df['rolling_high'] = df['high'].rolling(window=self.params['lookback_period']).max()
        df['rolling_low'] = df['low'].rolling(window=self.params['lookback_period']).min()
        
        # Calculate breakout signals
        df['breakout_high'] = (df['close'] > df['rolling_high'].shift(1)) & \
                             (df['adx_14'] > self.params['adx_threshold']) & \
                             (df['atr_14'] > df['atr_14'].rolling(window=20).mean() * self.params['atr_multiplier'])
        
        df['breakout_low'] = (df['close'] < df['rolling_low'].shift(1)) & \
                            (df['adx_14'] > self.params['adx_threshold']) & \
                            (df['atr_14'] > df['atr_14'].rolling(window=20).mean() * self.params['atr_multiplier'])
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for breakout entry signals."""
        if pd.isna(row.get('atr_14')) or pd.isna(row.get('adx_14')):
            return None
        
        # Long entry
        if row.get('breakout_high', False):
            confidence = min(0.9, (row['adx_14'] / 50) * (row['atr_14'] / row['atr_14'].rolling(window=20).mean()))
            return {
                'side': 1,  # Long
                'confidence': confidence,
                'signal_type': 'breakout_high'
            }
        
        # Short entry
        elif row.get('breakout_low', False):
            confidence = min(0.9, (row['adx_14'] / 50) * (row['atr_14'] / row['atr_14'].rolling(window=20).mean()))
            return {
                'side': -1,  # Short
                'confidence': confidence,
                'signal_type': 'breakout_low'
            }
        
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check for exit conditions."""
        entry_price = position['entry_price']
        side = position['side']
        
        # Stop loss (percentage-based)
        stop_loss = entry_price * (1 - side * self.params['stop_loss_pct'] / 100)
        
        # Take profit (percentage-based)
        take_profit = entry_price * (1 + side * self.params['take_profit_pct'] / 100)
        
        # Exit if price hits stop loss or take profit
        if side == 1:  # Long position
            return row['low'] <= stop_loss or row['high'] >= take_profit
        else:  # Short position
            return row['high'] >= stop_loss or row['low'] <= take_profit
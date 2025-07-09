import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class HybridTrendReversionStrategy(BaseStrategy):
    """Hybrid Trend-Reversion Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'ema_short': 50,
            'ema_long': 200,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_period': 14,
            'stop_loss_pct': 0.8,          # 0.8% stop loss for trend following
            'take_profit_pct': 1.6         # 1.6% take profit for hybrid approach
        }
        super().__init__("Hybrid Trend Reversion", {**default_params, **(params or {})})
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate hybrid trend-reversion signals."""
        df = df.copy()
        
        # Calculate trend direction
        df['trend_up'] = df[f'ema_{self.params["ema_short"]}'] > df[f'ema_{self.params["ema_long"]}']
        df['trend_down'] = df[f'ema_{self.params["ema_short"]}'] < df[f'ema_{self.params["ema_long"]}']
        
        # Calculate trend strength
        df['trend_strength'] = abs(df[f'ema_{self.params["ema_short"]}'] - df[f'ema_{self.params["ema_long"]}']) / df['atr_14']
        
        # Calculate RSI conditions
        df['rsi_oversold'] = df[f'rsi_{self.params["rsi_period"]}'] < self.params['rsi_oversold']
        df['rsi_overbought'] = df[f'rsi_{self.params["rsi_period"]}'] > self.params['rsi_overbought']
        
        # Generate signals
        df['long_signal'] = df['trend_up'] & df['rsi_oversold'] & (df['trend_strength'] > 0.5)
        df['short_signal'] = df['trend_down'] & df['rsi_overbought'] & (df['trend_strength'] > 0.5)
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for hybrid trend-reversion entry signals."""
        if pd.isna(row.get('atr_14')) or pd.isna(row.get(f'ema_{self.params["ema_short"]}')):
            return None
        
        # Long entry (uptrend + oversold RSI)
        if row.get('long_signal', False):
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            trend_strength = row['trend_strength']
            confidence = min(0.9, ((100 - rsi) / 70) * min(trend_strength / 2, 1.0))
            
            return {
                'side': 1,  # Long
                'confidence': confidence,
                'signal_type': 'hybrid_long',
                'trend_strength': trend_strength,
                'rsi_value': rsi
            }
        
        # Short entry (downtrend + overbought RSI)
        elif row.get('short_signal', False):
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            trend_strength = row['trend_strength']
            confidence = min(0.9, (rsi / 70) * min(trend_strength / 2, 1.0))
            
            return {
                'side': -1,  # Short
                'confidence': confidence,
                'signal_type': 'hybrid_short',
                'trend_strength': trend_strength,
                'rsi_value': rsi
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
        
        # Exit if price hits stop loss, take profit, or RSI normalizes
        rsi = row[f'rsi_{self.params["rsi_period"]}']
        rsi_normalized = (side == 1 and rsi > 50) or (side == -1 and rsi < 50)
        
        if side == 1:  # Long position
            return (row['low'] <= stop_loss or 
                   row['high'] >= take_profit or 
                   rsi_normalized)
        else:  # Short position
            return (row['high'] >= stop_loss or 
                   row['low'] <= take_profit or 
                   rsi_normalized)
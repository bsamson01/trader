import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class VWAPMeanReversionStrategy(BaseStrategy):
    """VWAP Mean Reversion Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'atr_period': 14,
            'rsi_period': 2,
            'deviation_multiplier': 1.5,
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.0
        }
        super().__init__("VWAP Mean Reversion", {**default_params, **(params or {})})
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals based on VWAP deviation."""
        df = df.copy()
        
        # Calculate VWAP deviation
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['atr_14']
        
        # Calculate RSI conditions
        df['rsi_oversold'] = df[f'rsi_{self.params["rsi_period"]}'] < self.params['rsi_oversold']
        df['rsi_overbought'] = df[f'rsi_{self.params["rsi_period"]}'] > self.params['rsi_overbought']
        
        # Generate signals
        df['long_signal'] = (df['vwap_deviation'] < -self.params['deviation_multiplier']) & \
                           df['rsi_oversold']
        
        df['short_signal'] = (df['vwap_deviation'] > self.params['deviation_multiplier']) & \
                            df['rsi_overbought']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for mean reversion entry signals."""
        if pd.isna(row.get('atr_14')) or pd.isna(row.get('vwap')):
            return None
        
        # Long entry (price below VWAP, oversold RSI)
        if row.get('long_signal', False):
            deviation = abs(row['vwap_deviation'])
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            confidence = min(0.9, (deviation / 3) * ((100 - rsi) / 80))
            
            return {
                'side': 1,  # Long
                'confidence': confidence,
                'signal_type': 'vwap_long',
                'deviation': row['vwap_deviation']
            }
        
        # Short entry (price above VWAP, overbought RSI)
        elif row.get('short_signal', False):
            deviation = abs(row['vwap_deviation'])
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            confidence = min(0.9, (deviation / 3) * (rsi / 80))
            
            return {
                'side': -1,  # Short
                'confidence': confidence,
                'signal_type': 'vwap_short',
                'deviation': row['vwap_deviation']
            }
        
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check for exit conditions."""
        if pd.isna(row.get('atr_14')) or pd.isna(row.get('vwap')):
            return False
        
        entry_price = position['entry_price']
        side = position['side']
        atr = row['atr_14']
        
        # Stop loss (1.5 ATR)
        stop_loss = entry_price - (side * atr * self.params['stop_loss_atr'])
        
        # Take profit (2 ATR)
        take_profit = entry_price + (side * atr * self.params['take_profit_atr'])
        
        # Exit if price hits stop loss, take profit, or returns to VWAP
        vwap_return = abs(row['close'] - row['vwap']) < (atr * 0.5)
        
        if side == 1:  # Long position
            return (row['low'] <= stop_loss or 
                   row['high'] >= take_profit or 
                   (row['close'] >= row['vwap'] and vwap_return))
        else:  # Short position
            return (row['high'] >= stop_loss or 
                   row['low'] <= take_profit or 
                   (row['close'] <= row['vwap'] and vwap_return))
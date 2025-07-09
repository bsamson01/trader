import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Opening Range Breakout Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'opening_minutes': 30,
            'volume_threshold': 1.5,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        super().__init__("Opening Range Breakout", {**default_params, **(params or {})})
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate opening range breakout signals."""
        df = df.copy()
        
        # Add date and time components
        df['date'] = df['time'].dt.date
        df['time_of_day'] = df['time'].dt.time
        
        # Calculate opening range for each day
        opening_range = df.groupby('date').apply(
            lambda x: x[x['time_of_day'] <= pd.Timestamp(f'00:{self.params["opening_minutes"]}:00').time()]
        )
        
        # Calculate opening high and low
        opening_high = opening_range.groupby('date')['high'].max()
        opening_low = opening_range.groupby('date')['low'].min()
        
        # Map back to original dataframe
        df['opening_high'] = df['date'].map(opening_high)
        df['opening_low'] = df['date'].map(opening_low)
        
        # Calculate volume spike
        volume_ma = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] / volume_ma
        
        # Generate breakout signals
        df['breakout_high'] = (df['close'] > df['opening_high']) & \
                             (df['volume_spike'] > self.params['volume_threshold'])
        
        df['breakout_low'] = (df['close'] < df['opening_low']) & \
                            (df['volume_spike'] > self.params['volume_threshold'])
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for opening range breakout entry signals."""
        if pd.isna(row.get('opening_high')) or pd.isna(row.get('opening_low')):
            return None
        
        # Long entry (break above opening high)
        if row.get('breakout_high', False):
            volume_confidence = min(1.0, row['volume_spike'] / 3)
            price_confidence = min(1.0, (row['close'] - row['opening_high']) / (row.get('atr_14', 0.001) * 2))
            confidence = (volume_confidence + price_confidence) / 2
            
            return {
                'side': 1,  # Long
                'confidence': confidence,
                'signal_type': 'opening_breakout_high',
                'breakout_strength': (row['close'] - row['opening_high']) / row.get('atr_14', 0.001)
            }
        
        # Short entry (break below opening low)
        elif row.get('breakout_low', False):
            volume_confidence = min(1.0, row['volume_spike'] / 3)
            price_confidence = min(1.0, (row['opening_low'] - row['close']) / (row.get('atr_14', 0.001) * 2))
            confidence = (volume_confidence + price_confidence) / 2
            
            return {
                'side': -1,  # Short
                'confidence': confidence,
                'signal_type': 'opening_breakout_low',
                'breakout_strength': (row['opening_low'] - row['close']) / row.get('atr_14', 0.001)
            }
        
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check for exit conditions."""
        if pd.isna(row.get('atr_14')):
            return False
        
        entry_price = position['entry_price']
        side = position['side']
        atr = row['atr_14']
        
        # Stop loss (1.5 ATR)
        stop_loss = entry_price - (side * atr * self.params['stop_loss_atr'])
        
        # Take profit (2.5 ATR)
        take_profit = entry_price + (side * atr * self.params['take_profit_atr'])
        
        # Exit if price hits stop loss or take profit
        if side == 1:  # Long position
            return row['low'] <= stop_loss or row['high'] >= take_profit
        else:  # Short position
            return row['high'] >= stop_loss or row['low'] <= take_profit
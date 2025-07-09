import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class OpeningRangeBreakoutORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy
    
    Edge: First 15-min high/low often sets the day's bias.
    
    Implementation:
    - Define range from 00:00–00:15 (crypto) or market open (indices/gold)
    - Buy break of range high with volume>50% of 20-bar avg; SL = ½ range, TP = 2× range
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("opening_range_breakout_orb", params or {})
        self.description = "Opening Range Breakout (ORB) Strategy"
        
        # Strategy parameters
        self.range_minutes = self.params.get('range_minutes', 15)
        self.volume_threshold = self.params.get('volume_threshold', 0.5)
        self.volume_avg_period = self.params.get('volume_avg_period', 20)
        self.stop_loss_multiplier = self.params.get('stop_loss_multiplier', 0.5)
        self.take_profit_multiplier = self.params.get('take_profit_multiplier', 2.0)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on opening range breakout."""
        df = df.copy()
        
        # Add time-based columns
        df['time'] = pd.to_datetime(df.index)
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        
        # Calculate opening range (first 15 minutes)
        df['is_opening_range'] = (
            (df['hour'] == 0) & 
            (df['minute'] < self.range_minutes)
        )
        
        # Calculate opening range high and low
        opening_range_mask = df['is_opening_range']
        if opening_range_mask.any():
            opening_high = df.loc[opening_range_mask, 'high'].max()
            opening_low = df.loc[opening_range_mask, 'low'].min()
            opening_range = opening_high - opening_low
            
            # Apply to all rows
            df['opening_high'] = opening_high
            df['opening_low'] = opening_low
            df['opening_range'] = opening_range
        else:
            # Fallback if no opening range data
            df['opening_high'] = df['high'].rolling(15).max()
            df['opening_low'] = df['low'].rolling(15).min()
            df['opening_range'] = df['opening_high'] - df['opening_low']
        
        # Calculate volume average
        df['volume_avg'] = df['volume'].rolling(self.volume_avg_period).mean()
        
        # Generate signals (only after opening range period)
        df['after_opening_range'] = ~df['is_opening_range']
        
        df['long_signal'] = (
            df['after_opening_range'] &
            (df['close'] > df['opening_high']) &
            (df['volume'] > self.volume_threshold * df['volume_avg'])
        )
        
        df['short_signal'] = (
            df['after_opening_range'] &
            (df['close'] < df['opening_low']) &
            (df['volume'] > self.volume_threshold * df['volume_avg'])
        )
        
        # Calculate stop loss and take profit levels
        df['long_stop_loss'] = df['opening_high'] - self.stop_loss_multiplier * df['opening_range']
        df['long_take_profit'] = df['opening_high'] + self.take_profit_multiplier * df['opening_range']
        
        df['short_stop_loss'] = df['opening_low'] + self.stop_loss_multiplier * df['opening_range']
        df['short_take_profit'] = df['opening_low'] - self.take_profit_multiplier * df['opening_range']
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.8,
                'reason': 'orb_breakout',
                'stop_loss': row['long_stop_loss'],
                'take_profit': row['long_take_profit']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.8,
                'reason': 'orb_breakout',
                'stop_loss': row['short_stop_loss'],
                'take_profit': row['short_take_profit']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit if price hits stop loss or take profit
            return (row['close'] <= position.get('stop_loss', 0) or 
                   row['close'] >= position.get('take_profit', float('inf')))
        elif position['side'] == -1:  # Short position
            # Exit if price hits stop loss or take profit
            return (row['close'] >= position.get('stop_loss', float('inf')) or 
                   row['close'] <= position.get('take_profit', 0))
        return False
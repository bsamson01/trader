import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

class HeikinAshiTrendRideStrategy(BaseStrategy):
    """
    Heikin-Ashi Trend Ride Strategy
    
    Edge: HA candles filter noise, letting you stay in trending waves longer.
    
    Implementation:
    - Enter after 3 consecutive HA candles in trend direction and true-range slope rising
    - Exit on first HA color flip or PSAR hit
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("heikin_ashi_trend_ride", params or {})
        self.description = "Heikin-Ashi Trend Ride Strategy"
        
        # Strategy parameters
        self.ha_consecutive_bars = self.params.get('ha_consecutive_bars', 3)
        self.psar_acceleration = self.params.get('psar_acceleration', 0.02)
        self.psar_maximum = self.params.get('psar_maximum', 0.2)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Heikin-Ashi trend riding."""
        df = df.copy()
        
        # Calculate Heikin-Ashi candles
        df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Determine HA candle color
        df['ha_color'] = np.where(df['ha_close'] > df['ha_open'], 'green', 'red')
        
        # Calculate true range slope
        df['true_range'] = df['ha_high'] - df['ha_low']
        df['tr_slope'] = df['true_range'].diff()
        
        # Detect consecutive HA candles
        df['ha_green_consecutive'] = 0
        df['ha_red_consecutive'] = 0
        
        for i in range(len(df)):
            if i < self.ha_consecutive_bars - 1:
                continue
                
            # Check for consecutive green candles
            if all(df.iloc[j]['ha_color'] == 'green' 
                   for j in range(i - self.ha_consecutive_bars + 1, i + 1)):
                df.iloc[i, df.columns.get_loc('ha_green_consecutive')] = 1
                
            # Check for consecutive red candles
            if all(df.iloc[j]['ha_color'] == 'red' 
                   for j in range(i - self.ha_consecutive_bars + 1, i + 1)):
                df.iloc[i, df.columns.get_loc('ha_red_consecutive')] = 1
        
        # Calculate PSAR
        df['psar'] = self._calculate_psar(df, self.psar_acceleration, self.psar_maximum)
        
        # Generate signals
        df['long_signal'] = (
            (df['ha_green_consecutive'] == 1) &
            (df['tr_slope'] > 0) &
            (df['ha_close'] > df['psar'])
        )
        
        df['short_signal'] = (
            (df['ha_red_consecutive'] == 1) &
            (df['tr_slope'] > 0) &
            (df['ha_close'] < df['psar'])
        )
        
        # Exit signals (HA color flip)
        df['exit_long'] = df['ha_color'] == 'red'
        df['exit_short'] = df['ha_color'] == 'green'
        
        return df
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        if row['long_signal']:
            return {
                'side': 1,  # Long
                'confidence': 0.8,
                'reason': 'ha_trend_ride',
                'psar': row['psar']
            }
        elif row['short_signal']:
            return {
                'side': -1,  # Short
                'confidence': 0.8,
                'reason': 'ha_trend_ride',
                'psar': row['psar']
            }
        return None
    
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        if position['side'] == 1:  # Long position
            # Exit on HA color flip or PSAR hit
            return (row['exit_long'] or row['ha_close'] <= row['psar'])
        elif position['side'] == -1:  # Short position
            # Exit on HA color flip or PSAR hit
            return (row['exit_short'] or row['ha_close'] >= row['psar'])
        return False
    
    def _calculate_psar(self, df: pd.DataFrame, acceleration: float, maximum: float) -> pd.Series:
        """Calculate Parabolic SAR."""
        psar = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        ep = pd.Series(index=df.index, dtype=float)
        
        # Initialize
        psar.iloc[0] = df['ha_low'].iloc[0]
        af.iloc[0] = acceleration
        ep.iloc[0] = df['ha_high'].iloc[0]
        
        long_position = True
        
        for i in range(1, len(df)):
            if long_position:
                psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                if df['ha_high'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = df['ha_high'].iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
                
                if df['ha_low'].iloc[i] < psar.iloc[i]:
                    long_position = False
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['ha_low'].iloc[i]
                    af.iloc[i] = acceleration
            else:
                psar.iloc[i] = psar.iloc[i-1] - af.iloc[i-1] * (psar.iloc[i-1] - ep.iloc[i-1])
                
                if df['ha_low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = df['ha_low'].iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
                
                if df['ha_high'].iloc[i] > psar.iloc[i]:
                    long_position = True
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['ha_high'].iloc[i]
                    af.iloc[i] = acceleration
        
        return psar
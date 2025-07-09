import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, StrategyError

class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy
    
    Trades breakouts from the opening range with volume confirmation.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # Range parameters
            'range_minutes': 30,      # Opening range duration (15/30/60)
            'volume_threshold': 1.5,  # Relative volume threshold
            
            # Day-type filter
            'atr_multiplier': 1.2,   # Skip if pre-market range > 1.2 × yesterday ATR
            
            # Time stop
            'time_stop_multiplier': 10,  # Auto-close at 10× range minutes
            
            # Risk management
            'stop_loss_pct': 0.7,
            'take_profit_pct': 1.5,
            'position_size_pct': 0.02,
            'min_notional': 0.0,
            
            # Daily loss limits
            'max_daily_loss_r': 3.0,
            'max_daily_loss_currency': None,
        }
        super().__init__("Opening Range Breakout", {**default_params, **(params or {})})
        
        # Initialize state
        self.open_position = False
        self.entry_bar = 0
        self.opening_range_high = None
        self.opening_range_low = None
        self.range_start_time = None
        self.range_end_time = None
    
    def _calculate_opening_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate opening range with look-ahead guard."""
        df_copy = df.copy()
        df_copy['date'] = df_copy['time'].dt.date
        df_copy['time_of_day'] = df_copy['time'].dt.time
        
        # Calculate range start and end times
        range_minutes = self.params['range_minutes']
        range_start = pd.Timestamp(f'00:{range_minutes:02d}:00').time()
        
        # Get opening range data
        opening_range = df_copy[df_copy['time_of_day'] <= range_start]
        
        # Calculate opening high and low for each day
        daily_ranges = {}
        for date, day_data in opening_range.groupby('date'):
            if len(day_data) > 0:
                daily_ranges[date] = {
                    'high': day_data['high'].max(),
                    'low': day_data['low'].min(),
                    'start_time': day_data['time'].min(),
                    'end_time': day_data['time'].max()
                }
        
        return daily_ranges
    
    def _check_day_type_filter(self, row: pd.Series, daily_ranges: Dict[str, Any]) -> bool:
        """Check day-type filter."""
        current_date = row['time'].date()
        if current_date not in daily_ranges:
            return False
        
        # Get yesterday's ATR
        yesterday_date = current_date - pd.Timedelta(days=1)
        if yesterday_date in daily_ranges:
            yesterday_atr = row.get('atr_14_yesterday', row.get('atr_14', 0.001))
            
            # Calculate pre-market range
            today_range = daily_ranges[current_date]
            pre_market_range = today_range['high'] - today_range['low']
            
            # Check if range is too large
            if pre_market_range > self.params['atr_multiplier'] * yesterday_atr:
                self.logger.debug(f"Day-type filter: pre-market range too large")
                return False
        
        return True
    
    def _check_volume_confirmation(self, row: pd.Series) -> bool:
        """Check volume confirmation."""
        if 'volume' not in row or pd.isna(row['volume']):
            return True  # Skip if no volume data
        
        # Calculate relative volume
        volume_ma = row.get('volume_ma_20', row.get('volume', 1))
        if volume_ma > 0:
            relative_volume = row['volume'] / volume_ma
            if relative_volume >= self.params['volume_threshold']:
                return True
        
        self.logger.debug(f"Volume confirmation failed: rel_vol={relative_volume:.2f}")
        return False
    
    def _validate_indicators(self, row: pd.Series) -> bool:
        """Validate that required indicators are available."""
        required_indicators = ['time', 'open', 'high', 'low', 'close']
        
        for indicator in required_indicators:
            if pd.isna(row.get(indicator)):
                self.logger.debug(f"Missing indicator: {indicator}")
                return False
        
        return True
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for opening range breakout entry signals."""
        try:
            # Validate indicators
            if not self._validate_indicators(row):
                return None
            
            # Check if we're in the opening range period
            current_time = row['time']
            current_date = current_time.date()
            
            # Calculate opening range if not already done
            if not hasattr(self, '_daily_ranges'):
                df_sample = pd.DataFrame([row])
                self._daily_ranges = self._calculate_opening_range(df_sample)
            
            # Check day-type filter
            if not self._check_day_type_filter(row, self._daily_ranges):
                return None
            
            # Check if we're past the opening range period
            if current_date in self._daily_ranges:
                range_end = self._daily_ranges[current_date]['end_time']
                if current_time <= range_end:
                    return None  # Still in opening range
            
            # Get opening range levels
            if current_date not in self._daily_ranges:
                return None
            
            opening_high = self._daily_ranges[current_date]['high']
            opening_low = self._daily_ranges[current_date]['low']
            
            # Check volume confirmation
            if not self._check_volume_confirmation(row):
                return None
            
            # Long entry (break above opening high)
            if row['close'] > opening_high:
                breakout_strength = (row['close'] - opening_high) / row.get('atr_14', 0.001)
                confidence = min(0.9, breakout_strength / 2)
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 + self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Long breakout: entry={entry_price:.5f}, breakout_strength={breakout_strength:.2f}")
                
                return {
                    'side': 1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'opening_breakout_high',
                        'breakout_strength': breakout_strength,
                        'opening_high': opening_high,
                        'opening_low': opening_low,
                        'confidence': confidence
                    }
                }
            
            # Short entry (break below opening low)
            elif row['close'] < opening_low:
                breakout_strength = (opening_low - row['close']) / row.get('atr_14', 0.001)
                confidence = min(0.9, breakout_strength / 2)
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 - self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Short breakout: entry={entry_price:.5f}, breakout_strength={breakout_strength:.2f}")
                
                return {
                    'side': -1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'opening_breakout_low',
                        'breakout_strength': breakout_strength,
                        'opening_high': opening_high,
                        'opening_low': opening_low,
                        'confidence': confidence
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in should_entry: {e}")
            return None
    
    def should_exit(self, row: pd.Series) -> bool:
        """Check for exit conditions."""
        if not self.open_position or self.current_position is None:
            return False
        
        try:
            position = self.current_position
            current_time = row['time']
            current_bar = row.name
            
            # Time stop
            if self.range_end_time:
                time_stop_minutes = self.params['range_minutes'] * self.params['time_stop_multiplier']
                time_since_range = (current_time - self.range_end_time).total_seconds() / 60
                if time_since_range >= time_stop_minutes:
                    self.logger.debug(f"Time stop exit: {time_since_range:.0f} minutes since range")
                    return True
            
            # Session end exit
            session_end = pd.Timestamp('16:00:00').time()
            if current_time.time() >= session_end:
                self.logger.debug("Session end exit")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in should_exit: {e}")
            return False
    
    def enter_trade(self, row: pd.Series, **kwargs) -> Optional[Any]:
        """Override enter_trade to track position state."""
        position = super().enter_trade(row, **kwargs)
        if position:
            self.open_position = True
            self.entry_bar = row.name
            
            # Store opening range info
            extra = kwargs.get('extra', {})
            self.opening_range_high = extra.get('opening_high')
            self.opening_range_low = extra.get('opening_low')
            self.range_end_time = row['time']
        
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Override exit_trade to track position state."""
        trade = super().exit_trade(row, reason)
        if trade:
            self.open_position = False
            self.opening_range_high = None
            self.opening_range_low = None
            self.range_end_time = None
        return trade
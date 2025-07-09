import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, StrategyError

class VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy
    
    Trades mean reversion opportunities when price deviates significantly from VWAP.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # VWAP parameters
            'atr_period': 14,
            'rsi_period': 2,
            'deviation_multiplier': 1.5,
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            
            # Session VWAP parameters
            'session_start': "09:30",  # Session start time
            'session_end': "16:00",    # Session end time
            'session_timezone': "US/Eastern",
            
            # Exit parameters
            'time_stop_bars': 10,      # Maximum bars to hold position
            'vwap_touch_threshold': 0.003,  # 0.3% threshold for VWAP touch
            
            # Filters
            'spread_multiplier': 2.0,  # Spread filter (2x median)
            'macro_event_blackout': True,  # Enable macro event blackout
            
            # Risk management
            'stop_loss_pct': 0.6,
            'take_profit_pct': 1.2,
            'position_size_pct': 0.02,
            'min_notional': 0.0,
            
            # Daily loss limits
            'max_daily_loss_r': 3.0,
            'max_daily_loss_currency': None,
        }
        super().__init__("VWAP Mean Reversion", {**default_params, **(params or {})})
        
        # Initialize state
        self.open_position = False
        self.entry_bar = 0
        self.entry_vwap_dev = 0.0
    
    def _calc_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR internally."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.params['atr_period']).mean()
        
        return atr
    
    def _calc_session_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate session VWAP with parameterized start/stop times."""
        # Create session mask
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['time'])
        df_copy['hour'] = df_copy['time'].dt.hour
        df_copy['minute'] = df_copy['time'].dt.minute
        
        # Parse session times
        start_hour, start_minute = map(int, self.params['session_start'].split(':'))
        end_hour, end_minute = map(int, self.params['session_end'].split(':'))
        
        # Create session mask
        session_mask = (
            (df_copy['hour'] > start_hour) | 
            ((df_copy['hour'] == start_hour) & (df_copy['minute'] >= start_minute))
        ) & (
            (df_copy['hour'] < end_hour) | 
            ((df_copy['hour'] == end_hour) & (df_copy['minute'] <= end_minute))
        )
        
        # Calculate VWAP for session periods
        typical_price = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        
        # Use volume if available, otherwise use typical price
        if 'volume' in df_copy.columns and not df_copy['volume'].isna().all():
            vwap = (typical_price * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()
        else:
            vwap = typical_price.rolling(window=20).mean()
        
        # Reset VWAP at session boundaries
        session_changes = session_mask.astype(int).diff().fillna(0)
        session_starts = session_changes == 1
        
        # Create session VWAP
        session_vwap = vwap.copy()
        for i in range(1, len(df_copy)):
            if session_starts.iloc[i]:
                # Reset VWAP calculation for new session
                session_vwap.iloc[i:] = typical_price.iloc[i:].rolling(window=20).mean()
        
        return session_vwap
    
    def _check_spread_filter(self, row: pd.Series) -> bool:
        """Check spread filter."""
        if 'spread' not in row or pd.isna(row['spread']):
            return True  # Skip if no spread data
        
        # Calculate median spread (would need historical data)
        # For now, use a simple threshold
        max_spread = row['close'] * 0.0001  # 1 pip equivalent
        return row['spread'] <= max_spread * self.params['spread_multiplier']
    
    def _check_macro_events(self, row: pd.Series) -> bool:
        """Check for macro event blackout periods."""
        if not self.params['macro_event_blackout']:
            return True
        
        # This would integrate with an economic calendar
        # For now, skip during major market hours
        if 'time' in row:
            hour = pd.to_datetime(row['time']).hour
            # Skip during major news times (8:30, 10:00, 14:00)
            news_hours = [8, 10, 14]
            for news_hour in news_hours:
                if abs(hour - news_hour) <= 1:  # 1 hour before/after
                    self.logger.debug(f"Skipping trade during macro event window: {hour}:00")
                    return False
        
        return True
    
    def _validate_indicators(self, row: pd.Series) -> bool:
        """Validate that required indicators are available."""
        required_indicators = ['vwap', 'atr_14', f'rsi_{self.params["rsi_period"]}']
        
        for indicator in required_indicators:
            if pd.isna(row.get(indicator)):
                self.logger.debug(f"Missing indicator: {indicator}")
                return False
        
        return True
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for mean reversion entry signals."""
        try:
            # Validate indicators
            if not self._validate_indicators(row):
                return None
            
            # Check filters
            if not self._check_spread_filter(row):
                self.logger.debug("Spread filter failed")
                return None
            
            if not self._check_macro_events(row):
                return None
            
            # Calculate VWAP deviation
            vwap_deviation = (row['close'] - row['vwap']) / row['atr_14']
            
            # Calculate RSI conditions
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            rsi_oversold = rsi < self.params['rsi_oversold']
            rsi_overbought = rsi > self.params['rsi_overbought']
            
            # Long entry (price below VWAP, oversold RSI)
            if (vwap_deviation < -self.params['deviation_multiplier'] and rsi_oversold):
                deviation = abs(vwap_deviation)
                confidence = min(0.9, (deviation / 3) * ((100 - rsi) / 80))
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 + self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Long signal: vwap_dev={vwap_deviation:.2f}, RSI={rsi:.1f}, confidence={confidence:.2f}")
                
                return {
                    'side': 1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'vwap_long',
                        'deviation': vwap_deviation,
                        'vwap_dev_at_entry': vwap_deviation,
                        'rsi_value': rsi,
                        'confidence': confidence
                    }
                }
            
            # Short entry (price above VWAP, overbought RSI)
            elif (vwap_deviation > self.params['deviation_multiplier'] and rsi_overbought):
                deviation = abs(vwap_deviation)
                confidence = min(0.9, (deviation / 3) * (rsi / 80))
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 - self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Short signal: vwap_dev={vwap_deviation:.2f}, RSI={rsi:.1f}, confidence={confidence:.2f}")
                
                return {
                    'side': -1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'vwap_short',
                        'deviation': vwap_deviation,
                        'vwap_dev_at_entry': vwap_deviation,
                        'rsi_value': rsi,
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
            current_price = row['close']
            current_bar = row.name
            
            # Time stop
            if current_bar - self.entry_bar >= self.params['time_stop_bars']:
                self.logger.debug(f"Time stop exit: {current_bar - self.entry_bar} bars")
                return True
            
            # VWAP touch exit
            if 'vwap' in row and not pd.isna(row['vwap']):
                vwap_distance = abs(current_price - row['vwap']) / current_price
                if vwap_distance <= self.params['vwap_touch_threshold']:
                    self.logger.debug(f"VWAP touch exit: distance={vwap_distance:.4f}")
                    return True
            
            # RSI normalization exit
            rsi = row.get(f'rsi_{self.params["rsi_period"]}')
            if rsi is not None:
                rsi_normalized = (position.side == 1 and rsi > 50) or (position.side == -1 and rsi < 50)
                if rsi_normalized:
                    self.logger.debug(f"RSI normalized exit: RSI={rsi:.1f}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in should_exit: {e}")
            return False
    
    def enter_trade(self, row: pd.Series, **kwargs) -> Optional[Any]:
        """Override enter_trade to track position state and log telemetry."""
        position = super().enter_trade(row, **kwargs)
        if position:
            self.open_position = True
            self.entry_bar = row.name
            self.entry_vwap_dev = kwargs.get('extra', {}).get('vwap_dev_at_entry', 0.0)
            
            # Log telemetry
            self.logger.info(f"Entered VWAP trade: dev_at_entry={self.entry_vwap_dev:.3f}")
        
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Override exit_trade to track position state and log telemetry."""
        trade = super().exit_trade(row, reason)
        if trade:
            self.open_position = False
            
            # Calculate and log telemetry
            if 'vwap' in row and not pd.isna(row['vwap']):
                exit_vwap_dev = (row['close'] - row['vwap']) / row.get('atr_14', 1.0)
                seconds_to_exit = (row['time'] - trade['entry_time']).total_seconds()
                
                self.logger.info(f"Exited VWAP trade: dev_at_entry={self.entry_vwap_dev:.3f}, "
                               f"dev_at_exit={exit_vwap_dev:.3f}, seconds_to_exit={seconds_to_exit:.0f}")
        
        return trade
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy, StrategyError

class TrendATRBreakoutStrategy(BaseStrategy):
    """
    Trend + ATR Breakout Strategy
    
    Edge: Trade only when volatility expands with the prevailing trend.
    
    Implementation:
    - Trend filter: price above EMA200 (long bias) or below (short)
    - Entry on close > previous high + 0.5 × ATR (inverse for shorts)
    - ATR-based trailing stop
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # Trend parameters
            'ema_period': 200,
            'htf_ema_period': 200,  # Higher timeframe EMA
            'htf_timeframe': 15,     # Higher timeframe in minutes
            
            # ATR parameters
            'atr_period': 14,
            'atr_multiplier_entry': 0.5,
            'atr_multiplier_stop': 2.0,
            'atr_multiplier_trail': 3.0,
            
            # Signal filters
            'adx_min': 25,           # Minimum ADX for trend strength
            'adx_rising': True,      # Require ADX to be rising
            'doji_threshold': 0.4,   # Maximum body size for doji (40%)
            
            # Entry parameters
            'stop_limit_offset': 0.1,  # ATR offset for stop-limit entry
            'limit_offset': 0.05,      # ATR offset for limit order
            
            # Risk management
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'position_size_pct': 0.02,
            'min_notional': 0.0,
            
            # Daily loss limits
            'max_daily_loss_r': 3.0,
            'max_daily_loss_currency': None,
        }
        super().__init__("Trend ATR Breakout", {**default_params, **(params or {})})
        
        # Initialize state
        self.open_position = False
        self.chandelier_stop = None
    
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
    
    def _check_signal_filters(self, row: pd.Series) -> bool:
        """Check signal filters."""
        # ADX filter
        if 'adx_14' in row and not pd.isna(row['adx_14']):
            if row['adx_14'] < self.params['adx_min']:
                self.logger.debug(f"ADX too low: {row['adx_14']:.1f} < {self.params['adx_min']}")
                return False
            
            # Check if ADX is rising
            if self.params['adx_rising'] and 'adx_14' in row.index:
                adx_prev = row.get('adx_14_prev', row['adx_14'])
                if row['adx_14'] <= adx_prev:
                    self.logger.debug("ADX not rising")
                    return False
        
        # Doji filter
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        if total_range > 0:
            body_ratio = body_size / total_range
            if body_ratio < self.params['doji_threshold']:
                self.logger.debug(f"Doji filter: body_ratio={body_ratio:.2f} < {self.params['doji_threshold']}")
                return False
        
        return True
    
    def _check_htf_trend_filter(self, row: pd.Series) -> bool:
        """Check higher timeframe trend filter."""
        # This would require HTF data
        # For now, use current timeframe EMA as proxy
        if f'ema_{self.params["htf_ema_period"]}' in row:
            htf_ema = row[f'ema_{self.params["htf_ema_period"]}']
            current_price = row['close']
            
            # Long bias when price above HTF EMA
            if self.current_position and self.current_position.side == 1:
                return current_price > htf_ema
            # Short bias when price below HTF EMA
            elif self.current_position and self.current_position.side == -1:
                return current_price < htf_ema
        
        return True
    
    def _validate_indicators(self, row: pd.Series) -> bool:
        """Validate that required indicators are available."""
        required_indicators = [
            f'ema_{self.params["ema_period"]}',
            'atr_14',
            'high',
            'low',
            'open',
            'close'
        ]
        
        for indicator in required_indicators:
            if pd.isna(row.get(indicator)):
                self.logger.debug(f"Missing indicator: {indicator}")
                return False
        
        return True
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if we should enter a position."""
        try:
            # Validate indicators
            if not self._validate_indicators(row):
                return None
            
            # Check signal filters
            if not self._check_signal_filters(row):
                return None
            
            # Calculate trend bias
            ema = row[f'ema_{self.params["ema_period"]}']
            trend_bullish = row['close'] > ema
            trend_bearish = row['close'] < ema
            
            # Get ATR
            atr = row['atr_14']
            if pd.isna(atr) or atr <= 0:
                return None
            
            # Calculate breakout levels
            prev_high = row.get('high_prev', row['high'])
            prev_low = row.get('low_prev', row['low'])
            
            long_breakout_level = prev_high + self.params['atr_multiplier_entry'] * atr
            short_breakout_level = prev_low - self.params['atr_multiplier_entry'] * atr
            
            # Long entry
            if trend_bullish and row['close'] > long_breakout_level:
                # Calculate stop-limit entry
                entry_price = row['high'] + self.params['stop_limit_offset'] * atr
                stop_loss = entry_price - self.params['atr_multiplier_stop'] * atr
                take_profit = entry_price + self.params['take_profit_pct'] / 100 * entry_price
                
                # Calculate Chandelier stop
                chandelier_stop = entry_price - self.params['atr_multiplier_trail'] * atr
                
                self.logger.debug(f"Long breakout: entry={entry_price:.5f}, stop={stop_loss:.5f}, "
                               f"chandelier={chandelier_stop:.5f}")
                
                return {
                    'side': 1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'trail': chandelier_stop,
                    'extra': {
                        'signal_type': 'trend_breakout_long',
                        'breakout_level': long_breakout_level,
                        'atr_value': atr,
                        'trend_strength': (row['close'] - ema) / atr
                    }
                }
            
            # Short entry
            elif trend_bearish and row['close'] < short_breakout_level:
                # Calculate stop-limit entry
                entry_price = row['low'] - self.params['stop_limit_offset'] * atr
                stop_loss = entry_price + self.params['atr_multiplier_stop'] * atr
                take_profit = entry_price - self.params['take_profit_pct'] / 100 * entry_price
                
                # Calculate Chandelier stop
                chandelier_stop = entry_price + self.params['atr_multiplier_trail'] * atr
                
                self.logger.debug(f"Short breakout: entry={entry_price:.5f}, stop={stop_loss:.5f}, "
                               f"chandelier={chandelier_stop:.5f}")
                
                return {
                    'side': -1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'trail': chandelier_stop,
                    'extra': {
                        'signal_type': 'trend_breakout_short',
                        'breakout_level': short_breakout_level,
                        'atr_value': atr,
                        'trend_strength': (ema - row['close']) / atr
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in should_entry: {e}")
            return None
    
    def should_exit(self, row: pd.Series) -> bool:
        """Check if we should exit a position."""
        if not self.open_position or self.current_position is None:
            return False
        
        try:
            position = self.current_position
            current_price = row['close']
            
            # Check HTF trend filter
            if not self._check_htf_trend_filter(row):
                self.logger.debug("HTF trend filter exit")
                return True
            
            # Update Chandelier stop if trailing
            if position.trail is not None:
                atr = row.get('atr_14', 0)
                if atr > 0:
                    if position.side == 1:  # Long
                        new_chandelier = current_price - self.params['atr_multiplier_trail'] * atr
                        if new_chandelier > position.trail:
                            position.trail = new_chandelier
                            self.logger.debug(f"Updated Chandelier stop: {position.trail:.5f}")
                    else:  # Short
                        new_chandelier = current_price + self.params['atr_multiplier_trail'] * atr
                        if new_chandelier < position.trail:
                            position.trail = new_chandelier
                            self.logger.debug(f"Updated Chandelier stop: {position.trail:.5f}")
            
            # Check Chandelier stop
            if position.trail is not None:
                if position.side == 1 and current_price <= position.trail:
                    self.logger.debug(f"Chandelier stop hit: {current_price:.5f} <= {position.trail:.5f}")
                    return True
                elif position.side == -1 and current_price >= position.trail:
                    self.logger.debug(f"Chandelier stop hit: {current_price:.5f} >= {position.trail:.5f}")
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
            self.chandelier_stop = position.trail
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Override exit_trade to track position state."""
        trade = super().exit_trade(row, reason)
        if trade:
            self.open_position = False
            self.chandelier_stop = None
        return trade
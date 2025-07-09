import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, StrategyError

class HybridTrendReversionStrategy(BaseStrategy):
    """
    Hybrid Trend-Reversion Strategy
    
    Combines trend following with mean reversion using EMA crossovers and RSI.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # Trend parameters
            'ema_short': 50,
            'ema_long': 200,
            
            # RSI parameters
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # ATR parameters
            'atr_period': 14,
            
            # Entry filters
            'bb_width_pct': 0.5,  # Minimum Bollinger Band width percentage
            'cooldown_bars': 5,    # Minimum bars between trades
            'atr_slope_min': 0.0,  # Minimum ATR slope for trend confirmation
            
            # Risk management
            'stop_loss_pct': 0.8,
            'take_profit_pct': 1.6,
            'position_size_pct': 0.02,
            'min_notional': 0.0,
            
            # Daily loss limits
            'max_daily_loss_r': 3.0,
            'max_daily_loss_currency': None,
        }
        super().__init__("Hybrid Trend Reversion", {**default_params, **(params or {})})
        
        # Initialize state
        self.open_position = False
        self.last_trade_bar = -999
    
    def _calc_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with fallback if volume column missing."""
        if 'volume' in df.columns and not df['volume'].isna().all():
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap
        else:
            # Fallback to simple moving average of typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            return typical_price.rolling(window=20).mean()
    
    def _validate_indicators(self, row: pd.Series) -> bool:
        """Validate that required indicators are available."""
        required_indicators = [
            f'ema_{self.params["ema_short"]}',
            f'ema_{self.params["ema_long"]}',
            f'rsi_{self.params["rsi_period"]}',
            'atr_14'
        ]
        
        for indicator in required_indicators:
            if pd.isna(row.get(indicator)):
                self.logger.debug(f"Missing indicator: {indicator}")
                return False
        
        return True
    
    def _check_entry_filters(self, row: pd.Series) -> bool:
        """Check entry filters."""
        # Cooldown filter
        if row.name - self.last_trade_bar < self.params['cooldown_bars']:
            return False
        
        # ATR slope filter
        if 'atr_14' in row and 'atr_14' in row.index:
            atr_slope = row['atr_14'] - row.get('atr_14_prev', row['atr_14'])
            if atr_slope < self.params['atr_slope_min']:
                return False
        
        # Bollinger Band width filter (if available)
        if 'bb_width' in row:
            bb_width_pct = row['bb_width'] / row['close'] * 100
            if bb_width_pct < self.params['bb_width_pct']:
                return False
        
        return True
    
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check for hybrid trend-reversion entry signals."""
        try:
            # Validate indicators
            if not self._validate_indicators(row):
                return None
            
            # Check entry filters
            if not self._check_entry_filters(row):
                return None
            
            # Calculate trend direction
            ema_short = row[f'ema_{self.params["ema_short"]}']
            ema_long = row[f'ema_{self.params["ema_long"]}']
            trend_up = ema_short > ema_long
            trend_down = ema_short < ema_long
            
            # Calculate trend strength
            atr = row['atr_14']
            trend_strength = abs(ema_short - ema_long) / atr if atr > 0 else 0
            
            # Calculate RSI conditions
            rsi = row[f'rsi_{self.params["rsi_period"]}']
            rsi_oversold = rsi < self.params['rsi_oversold']
            rsi_overbought = rsi > self.params['rsi_overbought']
            
            # Long entry (uptrend + oversold RSI)
            if trend_up and rsi_oversold and trend_strength > 0.5:
                confidence = min(0.9, ((100 - rsi) / 70) * min(trend_strength / 2, 1.0))
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 + self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Long signal: RSI={rsi:.1f}, trend_strength={trend_strength:.2f}, confidence={confidence:.2f}")
                
                return {
                    'side': 1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'hybrid_long',
                        'trend_strength': trend_strength,
                        'rsi_value': rsi,
                        'confidence': confidence
                    }
                }
            
            # Short entry (downtrend + overbought RSI)
            elif trend_down and rsi_overbought and trend_strength > 0.5:
                confidence = min(0.9, (rsi / 70) * min(trend_strength / 2, 1.0))
                
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 - self.params['take_profit_pct'] / 100)
                
                self.logger.debug(f"Short signal: RSI={rsi:.1f}, trend_strength={trend_strength:.2f}, confidence={confidence:.2f}")
                
                return {
                    'side': -1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'hybrid_short',
                        'trend_strength': trend_strength,
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
            
            # RSI normalization exit
            rsi = row.get(f'rsi_{self.params["rsi_period"]}')
            if rsi is not None:
                rsi_normalized = (position.side == 1 and rsi > 50) or (position.side == -1 and rsi < 50)
                if rsi_normalized:
                    self.logger.debug(f"RSI normalized exit: RSI={rsi:.1f}")
                    return True
            
            # Trend reversal exit
            ema_short = row.get(f'ema_{self.params["ema_short"]}')
            ema_long = row.get(f'ema_{self.params["ema_long"]}')
            if ema_short is not None and ema_long is not None:
                trend_reversal = (position.side == 1 and ema_short < ema_long) or (position.side == -1 and ema_short > ema_long)
                if trend_reversal:
                    self.logger.debug("Trend reversal exit")
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
            self.last_trade_bar = row.name
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Override exit_trade to track position state."""
        trade = super().exit_trade(row, reason)
        if trade:
            self.open_position = False
        return trade
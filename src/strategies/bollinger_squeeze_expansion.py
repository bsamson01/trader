import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy, StrategyError

class BollingerSqueezeExpansionStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Expansion Strategy
    
    Edge: Volatility contraction signals an impending trend burst.
    
    Implementation:
    - Monitor 20-period BB width; fire only when width < 10-bar percentile 15
    - Enter on candle close beyond band with MACD histogram above zero (long) / below zero (short)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # Bollinger Band parameters
            'bb_period': 20,
            'bb_std': 2,
            'squeeze_percentile': 15,
            'squeeze_lookback': 10,
            
            # MACD parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Pre-filters
            'rsi_neutral_min': 30,    # Neutral RSI range
            'rsi_neutral_max': 70,
            'inside_bar_threshold': 0.8,  # Inside bar threshold
            
            # Break candle requirements
            'break_atr_multiplier': 0.1,  # Close > band + 0.1 ATR
            'volume_multiplier': 1.3,     # Volume > 1.3 × avg20
            
            # Risk management
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'position_size_pct': 0.02,
            'min_notional': 0.0,
            'scale_out_r': 2.0,       # Scale out at +2R
            'break_even_r': 1.0,      # Move to BE at +1R
            
            # Daily loss limits
            'max_daily_loss_r': 3.0,
            'max_daily_loss_currency': None,
        }
        super().__init__("Bollinger Squeeze Expansion", {**default_params, **(params or {})})
        
        # Initialize state
        self.open_position = False
        self.entry_bar = 0
        self._bb_width_cache = None
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int) -> pd.Series:
        """Calculate MACD line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bb_width_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate BB width percentile with caching."""
        if self._bb_width_cache is not None:
            return self._bb_width_cache
        
        # Calculate Bollinger Bands
        bb_middle = df['close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['close'].rolling(window=self.params['bb_period']).std()
        bb_upper = bb_middle + self.params['bb_std'] * bb_std
        bb_lower = bb_middle - self.params['bb_std'] * bb_std
        
        # Calculate BB width
        bb_width = bb_upper - bb_lower
        
        # Calculate width percentile
        width_percentile = bb_width.rolling(
            window=self.params['squeeze_lookback']
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        
        self._bb_width_cache = width_percentile
        return width_percentile
    
    def _check_pre_filters(self, row: pd.Series) -> bool:
        """Check pre-filters."""
        # RSI neutral filter
        if 'rsi_14' in row and not pd.isna(row['rsi_14']):
            rsi = row['rsi_14']
            if rsi < self.params['rsi_neutral_min'] or rsi > self.params['rsi_neutral_max']:
                self.logger.debug(f"RSI not neutral: {rsi:.1f}")
                return False
        
        # Inside bar filter
        if 'high_prev' in row and 'low_prev' in row:
            prev_range = row['high_prev'] - row['low_prev']
            current_range = row['high'] - row['low']
            if prev_range > 0:
                range_ratio = current_range / prev_range
                if range_ratio < self.params['inside_bar_threshold']:
                    self.logger.debug(f"Inside bar filter: ratio={range_ratio:.2f}")
                    return False
        
        return True
    
    def _check_break_candle(self, row: pd.Series, bb_upper: float, bb_lower: float) -> bool:
        """Check break candle requirements."""
        # Check close beyond band + ATR
        atr = row.get('atr_14', 0.001)
        break_threshold = self.params['break_atr_multiplier'] * atr
        
        # Long break
        if row['close'] > bb_upper + break_threshold:
            # Check volume confirmation
            if 'volume' in row and 'volume_ma_20' in row:
                volume_ma = row['volume_ma_20']
                if volume_ma > 0:
                    relative_volume = row['volume'] / volume_ma
                    if relative_volume >= self.params['volume_multiplier']:
                        return True
            else:
                # Skip volume check if data not available
                return True
        
        # Short break
        elif row['close'] < bb_lower - break_threshold:
            # Check volume confirmation
            if 'volume' in row and 'volume_ma_20' in row:
                volume_ma = row['volume_ma_20']
                if volume_ma > 0:
                    relative_volume = row['volume'] / volume_ma
                    if relative_volume >= self.params['volume_multiplier']:
                        return True
            else:
                # Skip volume check if data not available
                return True
        
        return False
    
    def _validate_indicators(self, row: pd.Series) -> bool:
        """Validate that required indicators are available."""
        required_indicators = ['close', 'high', 'low', 'open']
        
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
            
            # Check pre-filters
            if not self._check_pre_filters(row):
                return None
            
            # Calculate BB width percentile
            df_sample = pd.DataFrame([row])
            width_percentile = self._calculate_bb_width_percentile(df_sample).iloc[0]
            
            # Check squeeze condition
            if width_percentile >= self.params['squeeze_percentile']:
                return None
            
            # Calculate Bollinger Bands
            bb_middle = row.get('bb_middle', row['close'])
            bb_upper = row.get('bb_upper', bb_middle + 2 * row.get('bb_std', 0.001))
            bb_lower = row.get('bb_lower', bb_middle - 2 * row.get('bb_std', 0.001))
            
            # Check break candle requirements
            if not self._check_break_candle(row, bb_upper, bb_lower):
                return None
            
            # Check MACD histogram
            macd_histogram = row.get('macd_histogram', 0)
            
            # Long entry
            if row['close'] > bb_upper and macd_histogram > 0:
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = bb_lower  # Stop at opposite band
                take_profit = entry_price + self.params['take_profit_pct'] / 100 * entry_price
                
                self.logger.debug(f"Long squeeze expansion: entry={entry_price:.5f}, "
                               f"width_percentile={width_percentile:.1f}")
                
                return {
                    'side': 1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'bb_expansion_long',
                        'width_percentile': width_percentile,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'macd_histogram': macd_histogram
                    }
                }
            
            # Short entry
            elif row['close'] < bb_lower and macd_histogram < 0:
                # Calculate stop and target
                entry_price = row['close']
                stop_loss = bb_upper  # Stop at opposite band
                take_profit = entry_price - self.params['take_profit_pct'] / 100 * entry_price
                
                self.logger.debug(f"Short squeeze expansion: entry={entry_price:.5f}, "
                               f"width_percentile={width_percentile:.1f}")
                
                return {
                    'side': -1,
                    'stop': stop_loss,
                    'tp': take_profit,
                    'extra': {
                        'signal_type': 'bb_expansion_short',
                        'width_percentile': width_percentile,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'macd_histogram': macd_histogram
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
            
            # Check PSAR trailing stop
            if 'psar' in row and not pd.isna(row['psar']):
                psar = row['psar']
                if position.side == 1 and current_price <= psar:
                    self.logger.debug(f"PSAR trailing stop hit: {current_price:.5f} <= {psar:.5f}")
                    return True
                elif position.side == -1 and current_price >= psar:
                    self.logger.debug(f"PSAR trailing stop hit: {current_price:.5f} >= {psar:.5f}")
                    return True
            
            # Check scale-out levels
            if hasattr(self, 'entry_price'):
                price_change = (current_price - self.entry_price) / self.entry_price * position.side
                stop_loss_pct = self.params['stop_loss_pct'] / 100
                
                # Scale out at +2R
                if price_change >= self.params['scale_out_r'] * stop_loss_pct:
                    self.logger.debug(f"Scale out at +{self.params['scale_out_r']}R")
                    return True
                
                # Move to breakeven at +1R
                if price_change >= self.params['break_even_r'] * stop_loss_pct:
                    # Update stop to entry price
                    position.stop = self.entry_price
                    self.logger.debug("Moved stop to breakeven")
            
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
            self.entry_price = position.entry_price
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Override exit_trade to track position state."""
        trade = super().exit_trade(row, reason)
        if trade:
            self.open_position = False
            self.entry_price = None
        return trade
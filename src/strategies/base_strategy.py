from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import traceback

# Custom exception for strategy errors
class StrategyError(Exception):
    """Custom exception for strategy-related errors."""
    pass

@dataclass
class Position:
    """Position dataclass with all required attributes."""
    side: int  # 1 for long, -1 for short
    entry_bar: int
    entry_price: float
    size: float
    stop: float
    tp: float
    trail: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate position data after initialization."""
        if self.side not in [1, -1]:
            raise StrategyError(f"Invalid side: {self.side}. Must be 1 (long) or -1 (short)")
        if self.size <= 0:
            raise StrategyError(f"Invalid size: {self.size}. Must be positive")
        if self.entry_price <= 0:
            raise StrategyError(f"Invalid entry_price: {self.entry_price}. Must be positive")

class BaseStrategy(ABC):
    """Base class for all trading strategies with enhanced framework."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        
        # Initialize logger with propagation off
        self.logger = logging.getLogger(self.name)
        self.logger.propagate = False
        
        # Start with defaults and update with user params
        self.params = self._get_default_params()
        if params:
            self.params.update(params)
        
        # Initialize state
        self.trades: List[Dict[str, Any]] = []
        self.current_position: Optional[Position] = None
        self.trading_enabled: bool = True
        
        # Risk control state
        self.daily_loss_r: float = 0.0
        self.daily_loss_currency: float = 0.0
        self.last_reset_date: Optional[datetime] = None
        
        # Performance tracking
        self.initial_capital = self.params.get('initial_capital', 100.0)
        self.current_balance = self.initial_capital
        
        self.logger.info(f"Initialized {self.name} with params: {self.params}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for all strategies."""
        return {
            # Capital Management
            'initial_capital': 100.0,
            'min_balance_threshold': 0.0,
            'position_size_pct': 0.02,
            'position_size_fixed': None,
            'min_notional': 0.0,
            
            # Risk Management
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'stop_loss_atr_multiplier': None,
            'take_profit_atr_multiplier': None,
            'max_daily_loss_r': 3.0,  # Maximum daily loss in R-multiples
            'max_daily_loss_currency': None,  # Maximum daily loss in currency
            
            # Trade Management
            'max_position_time': None,
            'trailing_stop_pct': None,
            'break_even_after_pct': None,
            
            # Strategy-specific
            'use_global_exit_rules': True,
        }
    
    def _reset_daily_limits(self, current_time: datetime) -> None:
        """Reset daily loss limits if it's a new day."""
        if (self.last_reset_date is None or 
            current_time.date() != self.last_reset_date.date()):
            self.daily_loss_r = 0.0
            self.daily_loss_currency = 0.0
            self.last_reset_date = current_time
            self.trading_enabled = True
            self.logger.info(f"Reset daily limits for {current_time.date()}")
    
    def _check_daily_limits(self, trade_r: float, trade_currency: float) -> bool:
        """Check if daily loss limits are breached."""
        max_daily_loss_r = self.params.get('max_daily_loss_r')
        max_daily_loss_currency = self.params.get('max_daily_loss_currency')
        
        if max_daily_loss_r and self.daily_loss_r >= max_daily_loss_r:
            self.trading_enabled = False
            self.logger.warning(f"Daily R-loss limit breached: {self.daily_loss_r:.2f}R")
            return False
        
        if max_daily_loss_currency and self.daily_loss_currency >= max_daily_loss_currency:
            self.trading_enabled = False
            self.logger.warning(f"Daily currency loss limit breached: ${self.daily_loss_currency:.2f}")
            return False
        
        return True
    
    def _update_daily_loss(self, trade_r: float, trade_currency: float) -> None:
        """Update daily loss tracking."""
        if trade_r < 0:
            self.daily_loss_r += abs(trade_r)
        if trade_currency < 0:
            self.daily_loss_currency += abs(trade_currency)
    
    def _calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on configuration."""
        # Use fixed position size if specified
        if self.params.get('position_size_fixed'):
            return self.params['position_size_fixed']
        
        # Calculate based on risk percentage
        risk_pct = self.params.get('position_size_pct', 0.02)
        risk_amount = self.current_balance * risk_pct
        
        # Calculate position size based on stop loss
        stop_loss_pct = self.params.get('stop_loss_pct', 1.0) / 100
        position_size = risk_amount / (entry_price * stop_loss_pct)
        
        # Apply minimum notional constraint
        min_notional = self.params.get('min_notional', 0.0)
        if min_notional > 0:
            position_size = max(position_size, min_notional / entry_price)
        
        return position_size
    
    def _validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate that required columns exist in the dataframe."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise StrategyError(f"Missing required columns: {missing_columns}")
    
    def enter_trade(self, row: pd.Series, **kwargs) -> Optional[Position]:
        """Enter a new trade position."""
        if not self.trading_enabled:
            self.logger.debug("Trading disabled due to daily limits")
            return None
        
        if self.current_position is not None:
            self.logger.debug("Already in position, cannot enter new trade")
            return None
        
        # Validate required columns
        required_columns = ['time', 'close', 'high', 'low']
        self._validate_required_columns(pd.DataFrame([row]), required_columns)
        
        # Calculate position size
        entry_price = row['close']
        position_size = self._calculate_position_size(entry_price)
        
        # Create position
        position = Position(
            side=kwargs.get('side', 1),
            entry_bar=kwargs.get('entry_bar', 0),
            entry_price=entry_price,
            size=position_size,
            stop=kwargs.get('stop', entry_price * (1 - kwargs.get('side', 1) * self.params['stop_loss_pct'] / 100)),
            tp=kwargs.get('tp', entry_price * (1 + kwargs.get('side', 1) * self.params['take_profit_pct'] / 100)),
            trail=kwargs.get('trail'),
            extra=kwargs.get('extra', {})
        )
        
        self.current_position = position
        self.logger.info(f"Entered {position.side} position at {entry_price:.5f}, size: {position_size:.2f}")
        
        return position
    
    def exit_trade(self, row: pd.Series, reason: str) -> Optional[Dict[str, Any]]:
        """Exit current trade position."""
        if self.current_position is None:
            self.logger.debug("No position to exit")
            return None
        
        # Validate required columns
        required_columns = ['time', 'close']
        self._validate_required_columns(pd.DataFrame([row]), required_columns)
        
        exit_price = row['close']
        position = self.current_position
        
        # Calculate profit/loss
        price_change = (exit_price - position.entry_price) / position.entry_price
        profit_currency = position.size * price_change * position.side
        profit_pct = price_change * 100 * position.side
        
        # Calculate R-multiple
        stop_loss_pct = self.params.get('stop_loss_pct', 1.0)
        r_multiple = profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        
        # Update balance
        self.current_balance += profit_currency
        
        # Update daily loss tracking
        self._update_daily_loss(r_multiple, profit_currency)
        
        # Create trade record
        trade = {
            'entry_time': row['time'],
            'exit_time': row['time'],
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'side': position.side,
            'position_size': position.size,
            'profit': profit_currency,
            'profit_pct': profit_pct,
            'r_multiple': r_multiple,
            'duration': 0,  # Will be calculated in execute_trades
            'strategy': self.name,
            'exit_reason': reason,
            'balance_before': self.current_balance - profit_currency,
            'balance_after': self.current_balance
        }
        
        self.trades.append(trade)
        self.current_position = None
        
        self.logger.info(f"Exited position at {exit_price:.5f}, P&L: {profit_currency:.2f} ({profit_pct:.2f}%), R: {r_multiple:.2f}")
        
        return trade
    
    @abstractmethod
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Check if we should enter a position.
        
        Args:
            row: Current data row
            
        Returns:
            Entry signal dict with 'side' and optional parameters, or None
        """
        pass
    
    @abstractmethod
    def should_exit(self, row: pd.Series) -> bool:
        """
        Check if we should exit the current position.
        
        Args:
            row: Current data row
            
        Returns:
            True if should exit, False otherwise
        """
        pass
    
    def execute_trades(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Execute the strategy and return trade list."""
        self.trades = []
        self.current_position = None
        self.current_balance = self.initial_capital
        
        # Validate required columns
        required_columns = ['time', 'open', 'high', 'low', 'close']
        self._validate_required_columns(df, required_columns)
        
        for i, row in df.iterrows():
            try:
                # Reset daily limits if needed
                self._reset_daily_limits(row['time'])
                
                # Check for exit if in position
                if self.current_position is not None:
                    if self.should_exit(row):
                        self.exit_trade(row, "strategy_exit")
                    else:
                        # Check global exit conditions
                        should_exit_global, reason = self._check_global_exit_conditions(row)
                        if should_exit_global:
                            self.exit_trade(row, reason)
                
                # Check for entry if not in position and trading enabled
                if self.current_position is None and self.trading_enabled:
                    entry_signal = self.should_entry(row)
                    if entry_signal:
                        # Check daily limits before entering
                        if self._check_daily_limits(0, 0):
                            self.enter_trade(row, **entry_signal)
                        else:
                            self.logger.debug("Skipping entry due to daily limits")
                
            except Exception as e:
                self.logger.error(f"Error processing row {i}: {e}")
                self.logger.debug(traceback.format_exc())
                continue
        
        # Calculate duration for all trades
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
                trade['duration'] = duration
        
        return self.trades
    
    def _check_global_exit_conditions(self, row: pd.Series) -> tuple[bool, str]:
        """Check global exit conditions (stop loss, take profit, time-based)."""
        if not self.params.get('use_global_exit_rules', True) or self.current_position is None:
            return False, ""
        
        position = self.current_position
        current_price = row['close']
        
        # Check stop loss
        if position.side == 1:  # Long
            if current_price <= position.stop:
                return True, "stop_loss"
        else:  # Short
            if current_price >= position.stop:
                return True, "stop_loss"
        
        # Check take profit
        if position.side == 1:  # Long
            if current_price >= position.tp:
                return True, "take_profit"
        else:  # Short
            if current_price <= position.tp:
                return True, "take_profit"
        
        # Check maximum position time
        max_time = self.params.get('max_position_time')
        if max_time:
            duration_minutes = (row['time'] - self.trades[-1]['entry_time']).total_seconds() / 60
            if duration_minutes >= max_time:
                return True, "max_time"
        
        return False, ""
    
    def get_performance_metrics(self, trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate performance metrics for the strategy."""
        if trades is None:
            trades = self.trades
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_r_multiple': 0,
                'max_drawdown': 0,
                'avg_duration': 0,
                'initial_capital': self.initial_capital,
                'closing_capital': self.initial_capital,
                'total_return_pct': 0.0,
                'profit_factor': 0,
                'stopped_due_to_balance': False,
                'min_balance_reached': self.initial_capital
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        avg_profit = df_trades['profit'].mean() if not df_trades['profit'].empty else 0
        total_profit = df_trades['profit'].sum() if not df_trades['profit'].empty else 0
        
        # Capital tracking
        closing_capital = self.current_balance
        total_return_pct = ((closing_capital - self.initial_capital) / self.initial_capital) * 100
        
        # R-multiple metrics
        avg_r_multiple = df_trades['r_multiple'].mean() if not df_trades['r_multiple'].empty else 0
        
        # Duration metrics
        avg_duration = df_trades['duration'].mean() if not df_trades['duration'].empty else 0
        
        # Drawdown calculation
        balance_series = pd.concat([pd.Series([self.initial_capital]), df_trades['balance_after']])
        running_max = balance_series.expanding().max()
        drawdown = balance_series - running_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Profit factor
        winning_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()
        losing_profit = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
        
        if losing_profit > 0:
            profit_factor = winning_profit / losing_profit
        elif winning_profit > 0:
            profit_factor = 999999
        else:
            profit_factor = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'avg_r_multiple': avg_r_multiple,
            'max_drawdown': max_drawdown,
            'avg_duration': avg_duration,
            'initial_capital': self.initial_capital,
            'closing_capital': closing_capital,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'stopped_due_to_balance': closing_capital <= self.params.get('min_balance_threshold', 0.0),
            'min_balance_reached': balance_series.min() if not balance_series.empty else self.initial_capital
        }
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        # Start with defaults and update with user params
        self.params = self._get_default_params()
        if params:
            # Update with user params
            self.params.update(params)
        self.trades = []
        self.current_position = None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for all strategies."""
        return {
            # Capital Management
            'initial_capital': 100.0,           # Starting capital amount
            'min_balance_threshold': 0.0,       # Minimum balance to continue trading
            'position_size_pct': 0.02,          # Percentage of balance to risk per trade (2%)
            'position_size_fixed': None,        # Fixed position size (overrides percentage if set)
            
            # Risk Management
            'stop_loss_pct': 1.0,               # Stop loss as percentage of entry price (1%)
            'take_profit_pct': 2.0,             # Take profit as percentage of entry price (2%)
            'stop_loss_atr_multiplier': None,   # Alternative: stop loss as ATR multiple
            'take_profit_atr_multiplier': None, # Alternative: take profit as ATR multiple
            
            # Trade Management
            'max_position_time': None,          # Max time to hold position (minutes)
            'trailing_stop_pct': None,          # Trailing stop percentage
            'break_even_after_pct': None,       # Move to breakeven after this profit %
            
            # Strategy-specific (can be overridden by individual strategies)
            'use_global_exit_rules': True,      # Whether to use global exit rules
        }
    
    def _calculate_position_size(self, current_balance: float, entry_price: float) -> float:
        """Calculate position size based on configuration."""
        # Use fixed position size if specified
        if self.params.get('position_size_fixed'):
            return self.params['position_size_fixed']
        
        # Otherwise use percentage of balance
        position_size_pct = self.params.get('position_size_pct', 0.02)
        return current_balance * position_size_pct
    
    def _check_global_exit_conditions(self, row: pd.Series, position: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check global exit conditions (stop loss, take profit, time-based).
        Returns (should_exit, reason)
        """
        if not self.params.get('use_global_exit_rules', True):
            return False, ""
        
        entry_price = position['entry_price']
        current_price = row['close']
        side = position['side']
        entry_time = position['entry_time']
        current_time = row['time']
        
        # Calculate price change
        if side == 1:  # Long position
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # Short position
            price_change_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check stop loss
        stop_loss_pct = self.params.get('stop_loss_pct')
        if stop_loss_pct and price_change_pct <= -abs(stop_loss_pct):
            return True, "stop_loss"
        
        # Check take profit
        take_profit_pct = self.params.get('take_profit_pct')
        if take_profit_pct and price_change_pct >= abs(take_profit_pct):
            return True, "take_profit"
        
        # Check ATR-based stops if configured
        if 'atr_14' in row:
            atr = row['atr_14']
            
            # ATR-based stop loss
            stop_loss_atr = self.params.get('stop_loss_atr_multiplier')
            if stop_loss_atr:
                if side == 1:  # Long
                    if current_price <= entry_price - (stop_loss_atr * atr):
                        return True, "atr_stop_loss"
                else:  # Short
                    if current_price >= entry_price + (stop_loss_atr * atr):
                        return True, "atr_stop_loss"
            
            # ATR-based take profit
            take_profit_atr = self.params.get('take_profit_atr_multiplier')
            if take_profit_atr:
                if side == 1:  # Long
                    if current_price >= entry_price + (take_profit_atr * atr):
                        return True, "atr_take_profit"
                else:  # Short
                    if current_price <= entry_price - (take_profit_atr * atr):
                        return True, "atr_take_profit"
        
        # Check maximum position time
        max_time = self.params.get('max_position_time')
        if max_time:
            duration_minutes = (current_time - entry_time).total_seconds() / 60
            if duration_minutes >= max_time:
                return True, "max_time"
        
        return False, ""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals for the strategy.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with signal columns added
        """
        pass
    
    @abstractmethod
    def should_entry(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Check if we should enter a position.
        
        Args:
            row: Current data row
            
        Returns:
            Entry signal dict or None
        """
        pass
    
    @abstractmethod
    def should_exit(self, row: pd.Series, position: Dict[str, Any]) -> bool:
        """
        Check if we should exit a position.
        
        Args:
            row: Current data row
            position: Current position information
            
        Returns:
            True if should exit, False otherwise
        """
        pass
    
    def execute_trades(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Execute the strategy and return trade list.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        current_position = None
        current_balance = self.params.get('initial_capital', 100.0)  # Starting balance
        min_balance_threshold = self.params.get('min_balance_threshold', 0.0)  # Minimum balance to continue trading
        position_size_pct = self.params.get('position_size_pct', 1.0)  # Percentage of balance to risk per trade
        
        # Add 1 pip lag simulation - store pending signals
        pending_entry_signal = None
        pending_exit = False
        
        for i, row in df.iterrows():
            # First, execute any pending signals from previous bar (1 pip lag simulation)
            
            # Execute pending exit signal
            if current_position and pending_exit:
                # Close position using current bar's price (1 bar delay from signal)
                exit_price = row['close']
                price_change = (exit_price - current_position['entry_price']) / current_position['entry_price']
                
                # Calculate profit based on position size
                position_value = current_position['position_size']
                profit = position_value * price_change * current_position['side']
                profit_pct = price_change * 100 * current_position['side']
                
                # Update balance
                current_balance += profit
                
                # Calculate R-multiple based on percentage risk
                stop_loss_pct = self.params.get('stop_loss_pct', 0.8)  # Default fallback
                r_multiple = profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
                
                trade = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': row['time'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'side': current_position['side'],
                    'position_size': current_position['position_size'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'r_multiple': r_multiple,
                    'duration': (row['time'] - current_position['entry_time']).total_seconds() / 60,  # minutes
                    'strategy': self.name,
                    'volatility_at_entry': row.get('atr_14', 0.001),
                    'signal_confidence': current_position.get('confidence', 0.5),
                    'balance_before': current_position['balance_before'],
                    'balance_after': current_balance,
                    'signal_lag': True,  # Flag to indicate this trade used signal lag
                    'exit_reason': current_position.get('exit_reason', 'unknown')  # Track why trade was closed
                }
                trades.append(trade)
                current_position = None
                pending_exit = False
            
            # Execute pending entry signal
            if not current_position and pending_entry_signal and current_balance > min_balance_threshold:
                # Calculate position size based on configuration
                position_size = self._calculate_position_size(current_balance, row['close'])
                
                # Only enter if we have enough balance for the position
                if position_size > 0:
                    current_position = {
                        'entry_time': row['time'],
                        'entry_price': row['close'],  # Use current bar's price (1 bar delay from signal)
                        'side': pending_entry_signal['side'],
                        'confidence': pending_entry_signal.get('confidence', 0.5),
                        'position_size': position_size,
                        'balance_before': current_balance,
                        'signal_time': pending_entry_signal['signal_time']  # Track when signal was detected
                    }
                pending_entry_signal = None
            
            # Now check for new signals on current bar (to be executed next bar)
            
            # Check for new exit signal if we have a position
            if current_position and not pending_exit:
                # First check global exit conditions (stop loss, take profit, etc.)
                should_exit_global, exit_reason = self._check_global_exit_conditions(row, current_position)
                
                if should_exit_global:
                    pending_exit = True
                    current_position['exit_reason'] = exit_reason
                # If no global exit, check strategy-specific exit conditions
                elif self.should_exit(row, current_position):
                    pending_exit = True
                    current_position['exit_reason'] = 'strategy_specific'
            
            # Check for new entry signal if we don't have a position and no pending entry
            elif not current_position and not pending_entry_signal and current_balance > min_balance_threshold:
                entry_signal = self.should_entry(row)
                if entry_signal:
                    # Store signal for execution on next bar
                    pending_entry_signal = {
                        'side': entry_signal['side'],
                        'confidence': entry_signal.get('confidence', 0.5),
                        'signal_time': row['time']  # Track when signal was detected
                    }
        
        return trades
    
    def get_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with performance metrics including capital tracking
        """
        initial_capital = self.params.get('initial_capital', 100.0)
        min_balance_threshold = self.params.get('min_balance_threshold', 0.0)
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_r_multiple': 0,
                'max_drawdown': 0,
                'avg_duration': 0,
                'initial_capital': initial_capital,
                'closing_capital': initial_capital,
                'total_return_pct': 0.0,
                'profit_factor': 0,
                'stopped_due_to_balance': False,
                'min_balance_reached': initial_capital
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics - handle potential NaN values
        avg_profit = df_trades['profit'].mean() if not df_trades['profit'].empty else 0
        avg_profit = avg_profit if not pd.isna(avg_profit) else 0
        
        total_profit = df_trades['profit'].sum() if not df_trades['profit'].empty else 0
        total_profit = total_profit if not pd.isna(total_profit) else 0
        
        # Capital tracking - use actual final balance from last trade
        if 'balance_after' in df_trades.columns and not df_trades['balance_after'].empty:
            closing_capital = df_trades['balance_after'].iloc[-1]
            # Find minimum balance reached during execution
            min_balance_reached = df_trades['balance_after'].min()
            # Check if strategy stopped due to insufficient balance
            stopped_due_to_balance = closing_capital <= min_balance_threshold
        else:
            # Fallback to old calculation if balance tracking not available
            closing_capital = initial_capital + total_profit
            min_balance_reached = closing_capital
            stopped_due_to_balance = False
        
        total_return_pct = ((closing_capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
        
        # R-multiple metrics - handle potential NaN values
        avg_r_multiple = df_trades['r_multiple'].mean() if not df_trades['r_multiple'].empty else 0
        avg_r_multiple = avg_r_multiple if not pd.isna(avg_r_multiple) else 0
        
        # Duration metrics - handle potential NaN values
        avg_duration = df_trades['duration'].mean() if not df_trades['duration'].empty else 0
        avg_duration = avg_duration if not pd.isna(avg_duration) else 0
        
        # Drawdown calculation based on balance progression
        if 'balance_after' in df_trades.columns and not df_trades['balance_after'].empty:
            balance_series = pd.concat([pd.Series([initial_capital]), df_trades['balance_after']])
            running_max = balance_series.expanding().max()
            drawdown = balance_series - running_max
            max_drawdown = drawdown.min() if not drawdown.empty else 0
        else:
            # Fallback to profit-based drawdown
            cumulative_profit = df_trades['profit'].cumsum()
            running_max = cumulative_profit.expanding().max()
            drawdown = cumulative_profit - running_max
            max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        max_drawdown = max_drawdown if not pd.isna(max_drawdown) else 0
        
        # Profit factor calculation - avoid infinity
        winning_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()
        losing_profit = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
        
        # Handle NaN values and avoid infinity
        winning_profit = winning_profit if not pd.isna(winning_profit) else 0
        losing_profit = losing_profit if not pd.isna(losing_profit) else 0
        
        if losing_profit > 0:
            profit_factor = winning_profit / losing_profit
        elif winning_profit > 0:
            profit_factor = 999999  # Large number instead of infinity
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
            'initial_capital': initial_capital,
            'closing_capital': closing_capital,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'stopped_due_to_balance': stopped_due_to_balance,
            'min_balance_reached': min_balance_reached
        }
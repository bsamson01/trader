#!/usr/bin/env python3
"""
Test script for the refactored forex trading framework.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the refactored components
from src.core.data_loader import DataLoader
from src.core.indicator_engine import IndicatorEngine
from src.core.strategy_engine import StrategyEngine
from src.strategies.hybrid_trend_reversion import HybridTrendReversionStrategy
from src.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategies.trend_atr_breakout import TrendATRBreakoutStrategy

def create_test_data():
    """Create synthetic test data."""
    np.random.seed(42)
    
    # Generate 1000 bars of synthetic data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    
    # Generate price data with some trend and volatility
    base_price = 1.2000
    trend = np.cumsum(np.random.normal(0, 0.0001, 1000))
    noise = np.random.normal(0, 0.0005, 1000)
    
    close_prices = base_price + trend + noise
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        # Generate realistic OHLC
        volatility = abs(np.random.normal(0, 0.0002))
        high = close + volatility
        low = close - volatility
        open_price = close + np.random.normal(0, 0.0001)
        
        # Ensure OHLC relationship
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def test_individual_strategies():
    """Test individual strategies with the new interface."""
    print("Testing individual strategies...")
    
    # Create test data
    df = create_test_data()
    
    # Test Hybrid Trend Reversion Strategy
    print("\n1. Testing Hybrid Trend Reversion Strategy")
    hybrid_strategy = HybridTrendReversionStrategy({
        'rsi_period': 14,
        'ema_short': 50,
        'ema_long': 200,
        'max_daily_loss_r': 3.0
    })
    
    try:
        trades = hybrid_strategy.execute_trades(df)
        metrics = hybrid_strategy.get_performance_metrics(trades)
        print(f"   Trades: {len(trades)}")
        print(f"   Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test VWAP Mean Reversion Strategy
    print("\n2. Testing VWAP Mean Reversion Strategy")
    vwap_strategy = VWAPMeanReversionStrategy({
        'deviation_multiplier': 1.5,
        'rsi_period': 2,
        'max_daily_loss_r': 3.0
    })
    
    try:
        trades = vwap_strategy.execute_trades(df)
        metrics = vwap_strategy.get_performance_metrics(trades)
        print(f"   Trades: {len(trades)}")
        print(f"   Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Trend ATR Breakout Strategy
    print("\n3. Testing Trend ATR Breakout Strategy")
    atr_strategy = TrendATRBreakoutStrategy({
        'atr_multiplier_entry': 0.5,
        'atr_multiplier_stop': 2.0,
        'max_daily_loss_r': 3.0
    })
    
    try:
        trades = atr_strategy.execute_trades(df)
        metrics = atr_strategy.get_performance_metrics(trades)
        print(f"   Trades: {len(trades)}")
        print(f"   Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
    except Exception as e:
        print(f"   Error: {e}")

def test_strategy_engine():
    """Test the strategy engine with the new interface."""
    print("\nTesting Strategy Engine...")
    
    # Create test data
    df = create_test_data()
    
    # Initialize strategy engine
    engine = StrategyEngine()
    engine.initialize_strategies()
    
    # Test running all strategies
    try:
        results = engine.run_all_strategies(df)
        
        print(f"Total strategies: {results['summary']['total_strategies']}")
        print(f"Successful strategies: {results['summary']['successful_strategies']}")
        print(f"Failed strategies: {results['summary']['failed_strategies']}")
        print(f"Total trades: {results['summary']['total_trades']}")
        
        if results['summary']['best_performer']:
            print(f"Best performer: {results['summary']['best_performer']}")
        if results['summary']['worst_performer']:
            print(f"Worst performer: {results['summary']['worst_performer']}")
        
        # Show performance ranking
        print("\nPerformance Ranking:")
        for i, rank in enumerate(results['comparative_analysis']['performance_ranking'][:5]):
            print(f"  {i+1}. {rank['strategy']}: {rank['return_pct']:.2f}% "
                  f"({rank['total_trades']} trades, {rank['win_rate']:.2f} win rate)")
        
    except Exception as e:
        print(f"Error testing strategy engine: {e}")

def test_position_dataclass():
    """Test the Position dataclass."""
    print("\nTesting Position dataclass...")
    
    from src.strategies.base_strategy import Position, StrategyError
    
    try:
        # Test valid position
        position = Position(
            side=1,
            entry_bar=100,
            entry_price=1.2000,
            size=1000.0,
            stop=1.1980,
            tp=1.2040,
            trail=1.1990,
            extra={'confidence': 0.8}
        )
        print("   ✓ Valid position created successfully")
        
        # Test invalid side
        try:
            invalid_position = Position(
                side=0,  # Invalid side
                entry_bar=100,
                entry_price=1.2000,
                size=1000.0,
                stop=1.1980,
                tp=1.2040
            )
        except StrategyError as e:
            print(f"   ✓ Invalid side correctly rejected: {e}")
        
        # Test invalid size
        try:
            invalid_position = Position(
                side=1,
                entry_bar=100,
                entry_price=1.2000,
                size=-1000.0,  # Invalid size
                stop=1.1980,
                tp=1.2040
            )
        except StrategyError as e:
            print(f"   ✓ Invalid size correctly rejected: {e}")
        
    except Exception as e:
        print(f"   ✗ Error testing Position dataclass: {e}")

def test_daily_loss_limits():
    """Test daily loss limits functionality."""
    print("\nTesting daily loss limits...")
    
    # Create test data
    df = create_test_data()
    
    # Test strategy with daily loss limits
    strategy = HybridTrendReversionStrategy({
        'max_daily_loss_r': 1.0,  # Very low limit for testing
        'position_size_pct': 0.05  # Higher risk for faster testing
    })
    
    try:
        trades = strategy.execute_trades(df)
        metrics = strategy.get_performance_metrics(trades)
        
        print(f"   Trades executed: {len(trades)}")
        print(f"   Final balance: {metrics.get('closing_capital', 0):.2f}")
        print(f"   Daily loss limit: {strategy.params.get('max_daily_loss_r')}R")
        
        # Check if trading was disabled
        if not strategy.trading_enabled:
            print("   ✓ Trading correctly disabled due to daily limits")
        else:
            print("   - Trading remained enabled")
        
    except Exception as e:
        print(f"   ✗ Error testing daily loss limits: {e}")

if __name__ == "__main__":
    print("Testing Refactored Forex Trading Framework")
    print("=" * 50)
    
    # Test Position dataclass
    test_position_dataclass()
    
    # Test daily loss limits
    test_daily_loss_limits()
    
    # Test individual strategies
    test_individual_strategies()
    
    # Test strategy engine
    test_strategy_engine()
    
    print("\n" + "=" * 50)
    print("Testing completed!")
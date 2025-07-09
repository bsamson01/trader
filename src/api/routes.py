from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import tempfile
import logging
from typing import Dict, Any, Optional

from ..core.data_loader import DataLoader
from ..core.indicator_engine import IndicatorEngine
from ..core.strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables to store current analysis
current_data = None
current_results = None
data_loader = DataLoader()
indicator_engine = IndicatorEngine()
strategy_engine = StrategyEngine()

class TradingConfig(BaseModel):
    """Configuration model for trading parameters."""
    # Capital Management
    initial_capital: float = Field(default=100.0, description="Starting capital amount", gt=0)
    min_balance_threshold: float = Field(default=0.0, description="Minimum balance to continue trading", ge=0)
    position_size_pct: float = Field(default=0.02, description="Percentage of balance to risk per trade", gt=0, le=1)
    position_size_fixed: Optional[float] = Field(default=None, description="Fixed position size (overrides percentage if set)", gt=0)
    
    # Risk Management
    stop_loss_pct: float = Field(default=1.0, description="Stop loss as percentage of entry price", gt=0)
    take_profit_pct: float = Field(default=2.0, description="Take profit as percentage of entry price", gt=0)
    stop_loss_atr_multiplier: Optional[float] = Field(default=None, description="Alternative: stop loss as ATR multiple", gt=0)
    take_profit_atr_multiplier: Optional[float] = Field(default=None, description="Alternative: take profit as ATR multiple", gt=0)
    
    # Trade Management
    max_position_time: Optional[int] = Field(default=None, description="Max time to hold position (minutes)", gt=0)
    use_global_exit_rules: bool = Field(default=True, description="Whether to use global exit rules")
    
    # Strategy-specific overrides
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Strategy-specific parameter overrides")

def clean_for_json(obj):
    """
    Recursively clean data structure to remove NaN and infinity values 
    that can't be serialized to JSON.
    """
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        # Convert pandas objects to dict/list and clean
        if isinstance(obj, pd.DataFrame):
            return clean_for_json(obj.to_dict('records'))
        else:
            return clean_for_json(obj.to_list())
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj) or obj is None:
        return None
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None if obj < 0 else 999999999  # Large number instead of infinity
        else:
            return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    else:
        # For other types, try to convert to string or return None
        try:
            return str(obj)
        except:
            return None

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and validate CSV data."""
    global current_data
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        print(f"[DEBUG] Temp file path: {tmp_file_path}")
        with open(tmp_file_path) as f:
            for i in range(5):
                print(f.readline().rstrip())
        
        # Load and validate data
        df = data_loader.load_csv(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Store data globally
        current_data = df
        
        # Get data info
        data_info = data_loader.get_data_info()
        
        return {
            "message": "Data uploaded successfully",
            "data_info": data_info,
            "rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/analyze")
async def analyze_strategies(config: Optional[TradingConfig] = None):
    """Run strategy analysis on uploaded data with configurable parameters."""
    global current_data, current_results
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload CSV data first.")
    
    try:
        print(f"[DEBUG] current_data: {current_data.head()}")

        # Compute indicators
        df_with_indicators = indicator_engine.compute_all_indicators(current_data)

        print(f"[DEBUG] df_with_indicators: {df_with_indicators.head()}")
        
        # Prepare strategy parameters
        strategy_params = {}
        
        if config:
            # Convert config to dict and apply to all strategies
            global_config = config.dict(exclude={'strategy_params'})
            
            # Apply global config to all strategies
            for strategy_name in ['trend_volatility_breakout', 'vwap_mean_reversion', 
                                'opening_range_breakout', 'hybrid_trend_reversion',
                                'vwap_mean_reversion_scalper', 'trend_atr_breakout',
                                'opening_range_breakout_orb', 'bollinger_squeeze_expansion',
                                'rsi_pullback_trend', 'donchian_channel_breakout',
                                'macd_adx_filter', 'breakout_pullback_continuation',
                                'heikin_ashi_trend_ride', 'volume_spike_reversal']:
                strategy_params[strategy_name] = global_config.copy()
                
                # Apply strategy-specific overrides if provided
                if config.strategy_params and strategy_name in config.strategy_params:
                    strategy_params[strategy_name].update(config.strategy_params[strategy_name])
        
        # Initialize strategies with configuration
        strategy_engine.initialize_strategies(strategy_params)
        
        # Run all strategies
        results = strategy_engine.run_all_strategies(df_with_indicators)
        
        # Store results globally
        current_results = results
        
        # Add configuration info to response
        response = {
            "message": "Analysis completed successfully",
            "summary": results['summary'],
            "strategies_analyzed": len(results['individual_results']),
            "configuration": config.dict() if config else "default"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

@router.get("/results")
async def get_results():
    """Get the current analysis results."""
    if current_results is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Run analysis first.")
    
    # Clean the results before returning to handle NaN values
    cleaned_results = clean_for_json(current_results)
    return cleaned_results

@router.get("/results/{strategy_name}")
async def get_strategy_results(strategy_name: str):
    """Get results for a specific strategy."""
    if current_results is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Run analysis first.")
    
    if strategy_name not in current_results['individual_results']:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
    
    # Clean the results before returning to handle NaN values
    strategy_result = current_results['individual_results'][strategy_name]
    cleaned_result = clean_for_json(strategy_result)
    return cleaned_result

@router.get("/data/info")
async def get_data_info():
    """Get information about the uploaded data."""
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    return data_loader.get_data_info()

@router.get("/data/sample")
async def get_sample_data(rows: int = 100):
    """Get a sample of the uploaded data."""
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    sample = data_loader.get_sample_data(rows)
    return {
        "sample_data": sample.to_dict('records'),
        "total_rows": len(current_data)
    }

@router.get("/indicators/summary")
async def get_indicators_summary():
    """Get summary of computed indicators."""
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    df_with_indicators = indicator_engine.compute_all_indicators(current_data)
    return indicator_engine.get_indicator_summary(df_with_indicators)

@router.get("/strategies/list")
async def get_available_strategies():
    """Get list of available strategies."""
    return {
        "strategies": [
            # Original strategies
            {
                "name": "trend_volatility_breakout",
                "description": "Trend + Volatility Breakout Strategy",
                "type": "Breakout"
            },
            {
                "name": "vwap_mean_reversion",
                "description": "VWAP Mean Reversion Strategy",
                "type": "Mean Reversion"
            },
            {
                "name": "opening_range_breakout",
                "description": "Opening Range Breakout Strategy",
                "type": "Breakout"
            },
            {
                "name": "hybrid_trend_reversion",
                "description": "Hybrid Trend-Reversion Strategy",
                "type": "Hybrid"
            },
            
            # New strategies
            {
                "name": "vwap_mean_reversion_scalper",
                "description": "VWAP Mean-Reversion Scalper Strategy",
                "type": "Scalping"
            },
            {
                "name": "trend_atr_breakout",
                "description": "Trend + ATR Breakout Strategy",
                "type": "Breakout"
            },
            {
                "name": "opening_range_breakout_orb",
                "description": "Opening Range Breakout (ORB) Strategy",
                "type": "Breakout"
            },
            {
                "name": "bollinger_squeeze_expansion",
                "description": "Bollinger Band Squeeze Expansion Strategy",
                "type": "Breakout"
            },
            {
                "name": "rsi_pullback_trend",
                "description": "RSI-2 Pullback in Trend Strategy",
                "type": "Mean Reversion"
            },
            {
                "name": "donchian_channel_breakout",
                "description": "Donchian Channel 20-Bar Breakout Strategy",
                "type": "Breakout"
            },
            {
                "name": "macd_adx_filter",
                "description": "MACD Cross with ADX Filter Strategy",
                "type": "Momentum"
            },
            {
                "name": "breakout_pullback_continuation",
                "description": "Breakout-Pullback Continuation Strategy",
                "type": "Continuation"
            },
            {
                "name": "heikin_ashi_trend_ride",
                "description": "Heikin-Ashi Trend Ride Strategy",
                "type": "Trend Following"
            },
            {
                "name": "volume_spike_reversal",
                "description": "Volume Spike Reversal Strategy",
                "type": "Reversal"
            }
        ]
    }

@router.get("/trades/chart")
async def get_trade_chart_data():
    """Get trade data combined with price data for visualization."""
    if current_results is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Run analysis first.")
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No price data available.")
    
    try:
        # Prepare price data
        price_data = current_data.copy()
        
        # Collect all trades from all strategies
        all_trades = []
        strategy_trades = {}
        
        for strategy_name, result in current_results['individual_results'].items():
            if 'trades' in result and result['trades']:
                trades = result['trades']
                strategy_trades[strategy_name] = trades
                
                # Add strategy name to each trade
                for trade in trades:
                    trade_with_strategy = trade.copy()
                    trade_with_strategy['strategy'] = strategy_name
                    all_trades.append(trade_with_strategy)
        
        # Prepare chart data
        chart_data = {
            'price_data': {
                'time': price_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'open': price_data['open'].tolist(),
                'high': price_data['high'].tolist(),
                'low': price_data['low'].tolist(),
                'close': price_data['close'].tolist(),
                'volume': price_data['volume'].tolist()
            },
            'strategies': {},
            'all_trades': all_trades
        }
        
        # Organize trades by strategy for easier filtering
        for strategy_name, trades in strategy_trades.items():
            strategy_display_name = strategy_name.replace('_', ' ').title()
            
            entries = []
            exits = []
            
            for trade in trades:
                # Entry point
                entries.append({
                    'time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': trade['entry_price'],
                    'side': 'Long' if trade['side'] == 1 else 'Short',
                    'position_size': trade['position_size'],
                    'confidence': trade.get('signal_confidence', 0.5)
                })
                
                # Exit point
                exits.append({
                    'time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': trade['exit_price'],
                    'profit': trade['profit'],
                    'profit_pct': trade['profit_pct'],
                    'exit_reason': trade.get('exit_reason', 'unknown'),
                    'duration': trade['duration']
                })
            
            chart_data['strategies'][strategy_name] = {
                'display_name': strategy_display_name,
                'entries': entries,
                'exits': exits,
                'total_trades': len(trades),
                'metrics': current_results['individual_results'][strategy_name].get('metrics', {})
            }
        
        return clean_for_json(chart_data)
        
    except Exception as e:
        logger.error(f"Error preparing chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Error preparing chart data: {str(e)}")

@router.get("/config/default")
async def get_default_config():
    """Get default trading configuration parameters."""
    return {
        "default_config": TradingConfig().dict(),
        "parameter_descriptions": {
            "initial_capital": "Starting capital amount (must be > 0)",
            "min_balance_threshold": "Minimum balance to continue trading (>= 0)",
            "position_size_pct": "Percentage of balance to risk per trade (0-1)",
            "position_size_fixed": "Fixed position size in currency units (overrides percentage)",
            "stop_loss_pct": "Stop loss as percentage of entry price",
            "take_profit_pct": "Take profit as percentage of entry price", 
            "stop_loss_atr_multiplier": "Alternative: stop loss as multiple of ATR",
            "take_profit_atr_multiplier": "Alternative: take profit as multiple of ATR",
            "max_position_time": "Maximum time to hold position (minutes)",
            "use_global_exit_rules": "Whether to use global exit rules",
            "strategy_params": "Strategy-specific parameter overrides"
        },
        "example_config": {
            "initial_capital": 1000.0,
            "position_size_pct": 0.01,
            "stop_loss_pct": 0.5,
            "take_profit_pct": 1.5,
            "max_position_time": 240,
            "use_global_exit_rules": True,
            "strategy_params": {
                "vwap_mean_reversion": {
                    "stop_loss_pct": 0.3,
                    "take_profit_pct": 0.8
                }
            }
        }
    }

@router.delete("/clear")
async def clear_data():
    """Clear current data and results."""
    global current_data, current_results
    current_data = None
    current_results = None
    
    return {"message": "Data and results cleared successfully"}
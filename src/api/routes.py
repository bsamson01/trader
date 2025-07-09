from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
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
async def analyze_strategies(strategy_params: Optional[Dict[str, Any]] = None):
    """Run strategy analysis on uploaded data."""
    global current_data, current_results
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload CSV data first.")
    
    try:
        # Compute indicators
        df_with_indicators = indicator_engine.compute_all_indicators(current_data)
        
        # Initialize strategies
        strategy_engine.initialize_strategies(strategy_params)
        
        # Run all strategies
        results = strategy_engine.run_all_strategies(df_with_indicators)
        
        # Store results globally
        current_results = results
        
        return {
            "message": "Analysis completed successfully",
            "summary": results['summary'],
            "strategies_analyzed": len(results['individual_results'])
        }
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

@router.get("/results")
async def get_results():
    """Get the current analysis results."""
    if current_results is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Run analysis first.")
    
    return current_results

@router.get("/results/{strategy_name}")
async def get_strategy_results(strategy_name: str):
    """Get results for a specific strategy."""
    if current_results is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Run analysis first.")
    
    if strategy_name not in current_results['individual_results']:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
    
    return current_results['individual_results'][strategy_name]

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
            }
        ]
    }

@router.delete("/clear")
async def clear_data():
    """Clear current data and results."""
    global current_data, current_results
    current_data = None
    current_results = None
    
    return {"message": "Data and results cleared successfully"}
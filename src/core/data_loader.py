import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and validation of OHLCV data from CSV files."""
    
    def __init__(self):
        self.required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        self.data = None
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate CSV data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with validated OHLCV data
        """
        try:
            # Always use tab-separated for this format
            df = pd.read_csv(file_path, header=None, names=self.required_columns, sep='\t', engine='python')
            print("[DEBUG] Loaded raw data head:")
            print(df.head())
            print("[DEBUG] NaN counts after load:")
            print(df.isna().sum())
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert volume to int
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(int)
            # Remove any rows with NaN values
            df = df.dropna()
            print("[DEBUG] NaN counts after conversion and dropna:")
            print(df.isna().sum())
            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)
            self.data = df
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the loaded data for required format and quality."""
        # Check if we have the right number of columns
        if len(df.columns) != len(self.required_columns):
            raise ValueError(f"Expected {len(self.required_columns)} columns, got {len(df.columns)}")
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] < df['low'])
        )
        if invalid_ohlc.any():
            raise ValueError("Found invalid OHLC relationships")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "total_rows": len(self.data),
            "date_range": {
                "start": self.data['time'].min().isoformat(),
                "end": self.data['time'].max().isoformat()
            },
            "timeframe": self._detect_timeframe(),
            "symbol": "EURUSD",  # Could be extracted from filename
            "columns": list(self.data.columns)
        }
    
    def _detect_timeframe(self) -> str:
        """Detect the timeframe of the data."""
        if self.data is None or len(self.data) < 2:
            return "unknown"
        
        time_diff = self.data['time'].diff().dropna()
        median_diff = time_diff.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return "1m"
        elif median_diff <= pd.Timedelta(minutes=5):
            return "5m"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "15m"
        elif median_diff <= pd.Timedelta(hours=1):
            return "1h"
        else:
            return "daily"
    
    def get_sample_data(self, n_rows: int = 100) -> pd.DataFrame:
        """Get a sample of the data for testing."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        return self.data.head(n_rows)
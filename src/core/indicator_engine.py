import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class IndicatorEngine:
    """Computes technical indicators for strategy analysis."""
    
    def __init__(self):
        self.indicators = {}
        
    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Compute moving averages
        df = self._compute_ema(df, 50)
        df = self._compute_ema(df, 200)
        
        # Compute VWAP
        df = self._compute_vwap(df)
        
        # Compute ATR
        df = self._compute_atr(df, 14)
        
        # Compute RSI
        df = self._compute_rsi(df, 2)
        df = self._compute_rsi(df, 14)
        
        # Compute ADX
        df = self._compute_adx(df, 14)
        
        # Compute Bollinger Bands
        df = self._compute_bollinger_bands(df, 20, 2)
        
        # Compute opening range
        df = self._compute_opening_range(df)
        
        # Compute volatility metrics
        df = self._compute_volatility_metrics(df)
        
        return df
    
    def _compute_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute Exponential Moving Average."""
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        return df
    
    def _compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Volume Weighted Average Price."""
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute Average Directional Index."""
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), -low_diff, 0)
        
        # Calculate TR (True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        plus_di = pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth * 100
        minus_di = pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth * 100
        
        # Calculate DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        df[f'adx_{period}'] = dx.rolling(window=period).mean()
        
        return df
    
    def _compute_bollinger_bands(self, df: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df[f'bb_upper_{period}'] = sma + (std * std_dev)
        df[f'bb_middle_{period}'] = sma
        df[f'bb_lower_{period}'] = sma - (std * std_dev)
        
        return df
    
    def _compute_opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute opening range (first 30 minutes of each day)."""
        df['date'] = df['time'].dt.date
        df['time_of_day'] = df['time'].dt.time
        
        # Group by date and find first 30 minutes
        opening_range = df.groupby('date').apply(
            lambda x: x[x['time_of_day'] <= pd.Timestamp('09:30').time()]
        )
        
        # Calculate high and low for opening range
        opening_high = opening_range.groupby('date')['high'].max()
        opening_low = opening_range.groupby('date')['low'].min()
        
        # Map back to original dataframe
        df['opening_high'] = df['date'].map(opening_high)
        df['opening_low'] = df['date'].map(opening_low)
        
        return df
    
    def _compute_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additional volatility metrics."""
        # Price volatility
        df['price_volatility'] = df['close'].rolling(window=20).std()
        
        # Volume volatility
        df['volume_volatility'] = df['volume'].rolling(window=20).std()
        
        # Volume spike indicator
        volume_ma = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] / volume_ma
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for all indicators."""
        summary = {}
        
        # RSI ranges
        if 'rsi_14' in df.columns:
            summary['rsi_14'] = {
                'min': df['rsi_14'].min(),
                'max': df['rsi_14'].max(),
                'mean': df['rsi_14'].mean()
            }
        
        # ATR ranges
        if 'atr_14' in df.columns:
            summary['atr_14'] = {
                'min': df['atr_14'].min(),
                'max': df['atr_14'].max(),
                'mean': df['atr_14'].mean()
            }
        
        # ADX ranges
        if 'adx_14' in df.columns:
            summary['adx_14'] = {
                'min': df['adx_14'].min(),
                'max': df['adx_14'].max(),
                'mean': df['adx_14'].mean()
            }
        
        return summary
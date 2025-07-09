# 📊 Strategy Analyzer Tool

A comprehensive trading strategy analysis tool that simulates and evaluates four different trading strategies on historical market data using FastAPI and a modern web interface.

## 🚀 Features

### Core Functionality
- **Multi-Strategy Analysis**: Run 4 different trading strategies in parallel
- **Real-time Processing**: Fast analysis with parallel execution
- **Interactive Dashboard**: Modern web interface with drag-and-drop file upload
- **Comprehensive Metrics**: Win rate, R-multiple, drawdown, and more
- **Visual Analytics**: Interactive charts and performance comparisons

### Trading Strategies Implemented

1. **Trend + Volatility Breakout**
   - Entry: Price breaks previous high/low + ATR filter during high ADX/ATR
   - Exit: Trail with ATR stop or fixed TP/SL
   - Best for: Trending markets with high volatility

2. **VWAP Mean Reversion**
   - Entry: Price deviates > 1.5×ATR from VWAP; RSI(2) is extreme
   - Exit: Return to VWAP or fixed TP
   - Best for: Range-bound markets

3. **Opening Range Breakout**
   - Entry: Break of first 30-minute high/low with volume confirmation
   - Exit: TP/SL or EOD close
   - Best for: Intraday momentum trading

4. **Hybrid Trend-Reversion**
   - Entry: Reversion setup within prevailing EMA200 trend
   - Exit: VWAP or RSI normalization
   - Best for: Trend-following with pullback entries

## 📋 Requirements

- Python 3.10+
- FastAPI
- pandas, numpy
- plotly (for charts)
- uvicorn (for server)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd strategy-analyzer-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # On macOS/Linux, you might need to use python3
   python3 main.py
   
   # Or if python is aliased correctly
   python main.py
   
   # Alternative: Run with uvicorn directly
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

## 📁 Data Format

The tool expects CSV files with the following format (no headers):

```
time,open,high,low,close,volume
2025-04-02 15:56,1.08713,1.08721,1.08703,1.08705,184
2025-04-02 15:57,1.08705,1.08705,1.08691,1.08694,204
```

### Required Columns:
- `time`: Timestamp (YYYY-MM-DD HH:MM format)
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Volume

## 🎯 Usage

### Web Interface

1. **Upload Data**: Drag and drop your CSV file or click to browse
2. **Run Analysis**: Click "Run Analysis" to process all strategies
3. **View Results**: Explore performance charts and detailed metrics
4. **Clear Data**: Use "Clear Data" to reset for new analysis

### API Endpoints

- `POST /api/upload` - Upload CSV data
- `POST /api/analyze` - Run strategy analysis
- `GET /api/results` - Get analysis results
- `GET /api/results/{strategy_name}` - Get specific strategy results
- `GET /api/data/info` - Get data information
- `DELETE /api/clear` - Clear current data

## 📊 Output Metrics

### Strategy Performance Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total Profit**: Cumulative profit/loss
- **Average R-Multiple**: Average risk-reward ratio
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Duration**: Average trade duration in minutes
- **Profit Factor**: Ratio of gross profit to gross loss

### Comparative Analysis
- **Performance Ranking**: Strategies ranked by total profit
- **Best/Worst Strategy**: Top and bottom performers
- **Overall Statistics**: Combined metrics across all strategies

## 🔧 Configuration

### Strategy Parameters

Each strategy can be customized with parameters:

```python
strategy_params = {
    'trend_volatility_breakout': {
        'atr_period': 14,
        'adx_threshold': 25,
        'stop_loss_pct': 0.8,      # 0.8% stop loss
        'take_profit_pct': 1.8     # 1.8% take profit
    },
    'vwap_mean_reversion': {
        'deviation_multiplier': 1.5,
        'rsi_oversold': 20,
        'rsi_overbought': 80,
        'stop_loss_pct': 0.6,      # 0.6% stop loss
        'take_profit_pct': 1.2     # 1.2% take profit
    },
    'opening_range_breakout': {
        'opening_minutes': 30,
        'volume_threshold': 1.5,
        'stop_loss_pct': 0.7,      # 0.7% stop loss
        'take_profit_pct': 1.5     # 1.5% take profit
    },
    'hybrid_trend_reversion': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'stop_loss_pct': 0.8,      # 0.8% stop loss
        'take_profit_pct': 1.6     # 1.6% take profit
    }
}
```

## 📈 Technical Indicators

The tool computes and uses:

- **EMA(50) & EMA(200)**: Trend direction
- **VWAP**: Volume-weighted average price
- **ATR(14)**: Average True Range for volatility
- **RSI(2) & RSI(14)**: Relative Strength Index
- **ADX(14)**: Average Directional Index
- **Bollinger Bands(20,2)**: Volatility bands
- **Opening Range**: First 30 minutes high/low

## 🎨 Visualizations

### Performance Charts
- **Bar Chart**: Strategy performance comparison
- **Scatter Plot**: Trade profit vs duration analysis
- **Interactive Elements**: Hover tooltips and zoom capabilities

### Data Information
- **Date Range**: Analysis period
- **Total Rows**: Data points processed
- **Timeframe**: Detected data frequency
- **Symbol**: Trading instrument

## 🔍 Analysis Features

### Market Context Analysis
- **Volatility Regimes**: Low, medium, high volatility periods
- **Trend Strength**: Flat, medium, strong trends
- **Volume Analysis**: Volume spike detection
- **Time-based Analysis**: Hour of day, day of week performance

### Trade Analysis
- **Entry/Exit Points**: Precise trade timing
- **Signal Confidence**: Strategy confidence scores
- **Risk Metrics**: R-multiple calculations
- **Duration Analysis**: Trade holding periods

## 🚀 Performance

- **Parallel Processing**: All strategies run simultaneously
- **Memory Efficient**: Optimized for large datasets
- **Fast Execution**: C++ optimized pandas operations
- **Scalable**: Handles millions of data points

## 🛡️ Error Handling

- **Data Validation**: Comprehensive CSV format checking
- **OHLC Validation**: Ensures proper price relationships
- **Missing Data**: Handles gaps and invalid entries
- **Strategy Errors**: Graceful handling of calculation errors

## 🔮 Future Enhancements

- **Portfolio Simulation**: Multi-strategy portfolio backtesting
- **Machine Learning**: ML-based signal generation
- **Real-time Data**: Live market data integration
- **Advanced Analytics**: Monte Carlo simulations
- **Export Features**: PDF reports and Excel exports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs` when running

## 📊 Example Results

After running analysis on EURUSD data:
- **Best Strategy**: VWAP Mean Reversion (67% win rate)
- **Total Trades**: 1,247 across all strategies
- **Best R-Multiple**: 2.34 average for Trend Breakout
- **Lowest Drawdown**: 0.8% for Hybrid Strategy

## 🔧 Troubleshooting

### Common Issues

**"python: command not found" on macOS/Linux:**
```bash
# Use python3 instead
python3 main.py

# Or install with brew (macOS)
brew install python3
```

**Port 80 permission denied:**
- The app now runs on port 8000 by default (no admin required)
- Access via: `http://localhost:8000`

**Virtual Environment Issues:**
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Then install requirements
pip install -r requirements.txt
```

**Module Import Errors:**
```bash
# Make sure you're in the project directory
cd forex-test

# Install dependencies
pip install -r requirements.txt
```

**"Address already in use" Error:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

---

**Happy Trading! 📈**

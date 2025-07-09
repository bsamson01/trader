# Forex Trading Strategy Analyzer

A comprehensive web-based platform for analyzing and backtesting 14 different forex trading strategies. The platform provides real-time analysis, performance metrics, and comparative results across multiple strategies.

## 🚀 Features

- **14 Trading Strategies**: From scalping to trend following
- **Real-time Analysis**: Upload CSV data and get instant results
- **Performance Metrics**: Win rate, profit factor, drawdown analysis
- **Comparative Analysis**: Rank strategies by performance
- **Interactive Charts**: Visualize results with Plotly
- **Web Interface**: Modern, responsive UI

## 📊 Available Strategies

### 1. VWAP Mean-Reversion Scalper
**Edge**: Price rarely stays far from VWAP intraday; exploit snap-backs.

**Implementation**:
- Long if price ≤ VWAP – 1.5 × ATR and RSI(2)<10
- Short if ≥ VWAP + 1.5 × ATR and RSI(2)>90
- Exit at VWAP touch or after N bars

**Tips**: Use exchange/true volume for VWAP—CFD "volume" can be synthetic.

### 2. Trend + ATR Breakout
**Edge**: Trade only when volatility expands with the prevailing trend.

**Implementation**:
- Trend filter: price above EMA200 (long bias) or below (short)
- Entry on close > previous high + 0.5 × ATR (inverse for shorts)
- ATR-based trailing stop

**Tips**: Pre-compute ATR in ticks to avoid floating-point slippage errors.

### 3. Opening Range Breakout (ORB)
**Edge**: First 15-min high/low often sets the day's bias.

**Implementation**:
- Define range from 00:00–00:15 (crypto) or market open (indices/gold)
- Buy break of range high with volume>50% of 20-bar avg
- SL = ½ range, TP = 2× range

**Tips**: Back-test separate rules for Monday vs. Friday—edge often dissimilar.

### 4. Bollinger Band Squeeze Expansion
**Edge**: Volatility contraction signals an impending trend burst.

**Implementation**:
- Monitor 20-period BB width; fire only when width < 10-bar percentile 15
- Enter on candle close beyond band with MACD histogram above zero (long) / below zero (short)

**Tips**: Log BB width percentile to verify you're not calibrating to one market regime.

### 5. RSI-2 Pullback in Trend
**Edge**: Buy the dip inside a strong up-trend; quick snap-backs produce high hit rate.

**Implementation**:
- Up-trend: price > EMA200; enter when RSI(2)≤10
- Exit when RSI crosses 50 or +0.6 × ATR

**Tips**: Clamp RSI smash orders to 80% of max position size to soften slippage.

### 6. Donchian Channel 20-Bar Breakout
**Edge**: Classic Turtle logic—ride medium-term breakouts.

**Implementation**:
- Buy break of 20-bar high; initial SL at 10-bar low
- Scale in at +0.5 × ATR

**Tips**: Run two instances: one on 1-min, one on 5-min data.

### 7. MACD Cross with ADX Filter
**Edge**: Momentum crosses are more reliable when trend strength (ADX) is elevated.

**Implementation**:
- MACD(12,26,9) bullish cross and ADX(14) > 25 = long
- Bearish cross + ADX>25 = short
- Exit on opposite MACD cross or +2 × ATR

**Tips**: Throttle signals: ignore crosses occurring within 3 bars of the previous trade exit.

### 8. Breakout-Pullback Continuation
**Edge**: Most breakouts retest; entering on the pullback improves R/R and win-rate.

**Implementation**:
- Detect breakout candle (range > 1.5× ATR, closes beyond resistance)
- Place limit order at 38–62% Fibonacci retrace of breakout candle
- Stop beyond 78%

**Tips**: Track "retest depth" stat—refine fib zone that statistically fills and holds.

### 9. Heikin-Ashi Trend Ride
**Edge**: HA candles filter noise, letting you stay in trending waves longer.

**Implementation**:
- Enter after 3 consecutive HA candles in trend direction and true-range slope rising
- Exit on first HA color flip or PSAR hit

**Tips**: Back-test with true-range-based trailing stop instead of HA flip.

### 10. Volume Spike Reversal
**Edge**: Extreme volume often marks capitulation; price snaps back.

**Implementation**:
- Identify volume >2× 20-bar average and candle range >1.5× ATR
- Enter opposite direction on confirmation Doji/hammer
- TP at midpoint of spike bar

**Tips**: Require divergence to filter fake spikes.

### 11. Trend Volatility Breakout (Original)
**Edge**: Breakout strategies with volatility confirmation.

**Implementation**:
- Combines trend analysis with volatility breakout signals
- Uses multiple timeframe analysis

### 12. VWAP Mean Reversion (Original)
**Edge**: Mean reversion around VWAP levels.

**Implementation**:
- Trades reversions from VWAP extremes
- Uses volume-weighted price levels

### 13. Opening Range Breakout (Original)
**Edge**: Breakout from daily opening range.

**Implementation**:
- Identifies and trades opening range breakouts
- Uses volume confirmation

### 14. Hybrid Trend Reversion (Original)
**Edge**: Combines trend and reversion signals.

**Implementation**:
- Hybrid approach using multiple signal types
- Adaptive position sizing

## �️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd forex-test
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python main.py
```

4. **Access the web interface**:
Open your browser and go to `http://localhost:80`

## 📁 Project Structure

```
forex-test/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── EURUSD1.csv           # Sample forex data
├── src/
│   ├── api/
│   │   └── routes.py     # API endpoints
│   ├── core/
│   │   ├── data_loader.py      # Data loading utilities
│   │   ├── indicator_engine.py # Technical indicators
│   │   └── strategy_engine.py  # Strategy execution
│   ├── strategies/       # All trading strategies
│   │   ├── base_strategy.py
│   │   ├── vwap_mean_reversion_scalper.py
│   │   ├── trend_atr_breakout.py
│   │   ├── opening_range_breakout_orb.py
│   │   ├── bollinger_squeeze_expansion.py
│   │   ├── rsi_pullback_trend.py
│   │   ├── donchian_channel_breakout.py
│   │   ├── macd_adx_filter.py
│   │   ├── breakout_pullback_continuation.py
│   │   ├── heikin_ashi_trend_ride.py
│   │   ├── volume_spike_reversal.py
│   │   └── [original strategies...]
│   └── ui/
│       └── dashboard.py   # Dashboard components
└── static/
    └── index.html        # Web interface
```

## � Data Format

The platform expects CSV files with the following columns:
- `time`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Volume (optional, will be set to 1 if missing)

## 🔧 API Endpoints

### Data Management
- `POST /api/upload` - Upload CSV data
- `GET /api/data/info` - Get data information
- `GET /api/data/sample` - Get sample data
- `DELETE /api/clear` - Clear current data

### Analysis
- `POST /api/analyze` - Run strategy analysis
- `GET /api/results` - Get analysis results
- `GET /api/results/{strategy_name}` - Get specific strategy results

### Information
- `GET /api/strategies/list` - List available strategies
- `GET /api/indicators/summary` - Get indicator summary

## 📈 Performance Metrics

Each strategy provides comprehensive performance metrics:

- **Total Trades**: Number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Total Profit**: Sum of all trade profits/losses
- **Average R-Multiple**: Average risk-reward ratio
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Return %**: Percentage return on initial capital
- **Profit Factor**: Ratio of gross profit to gross loss

## 🎯 Usage

1. **Upload Data**: Drag and drop your CSV file or use the file browser
2. **Run Analysis**: Click "Run Analysis" to execute all strategies
3. **Review Results**: 
   - Portfolio summary shows overall performance
   - Performance chart compares all strategies
   - Detailed results show individual strategy metrics
   - Trade analysis visualizes trade distribution

## � Strategy Customization

Each strategy can be customized with parameters:

```python
# Example: Customize VWAP Mean Reversion Scalper
strategy_params = {
    'vwap_mean_reversion_scalper': {
        'vwap_multiplier': 1.5,
        'rsi_period': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 90,
        'max_bars_in_trade': 20
    }
}
```

## 🚨 Risk Disclaimer

This platform is for educational and research purposes only. Past performance does not guarantee future results. Always:

- Test strategies thoroughly before live trading
- Use proper risk management
- Never risk more than you can afford to lose
- Consider market conditions and regime changes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Note**: This platform is designed for educational purposes. Always validate strategies with proper backtesting and paper trading before using real money.

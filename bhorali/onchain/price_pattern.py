from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec


def analyze_price_patterns(ohlcv_data: list) -> dict:
    """
    Analyze price patterns and technical indicators from OHLCV data

    Args:
        ohlcv_data (list): List of hourly OHLCV data points where each point is
                          [timestamp, open, high, low, close, volume]

    Returns:
        dict: Price analysis with the following metrics:
            - price_trends: Overall trend direction and strength
            - volatility_metrics: Various volatility measures
            - support_resistance: Key price levels
            - candlestick_patterns: Identified patterns
            - technical_indicators: Common indicators like RSI, MACD
    """
    # Extract price data
    timestamps = [point[0] for point in ohlcv_data]
    opens = [point[1] for point in ohlcv_data]
    highs = [point[2] for point in ohlcv_data]
    lows = [point[3] for point in ohlcv_data]
    closes = [point[4] for point in ohlcv_data]

    # Calculate basic price metrics
    current_price = closes[-1]
    price_change = ((closes[-1] - closes[0]) / closes[0]) * 100

    # Calculate volatility
    daily_returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))
    ]
    volatility = (sum(r**2 for r in daily_returns) / len(daily_returns)) ** 0.5 * 100

    # Calculate Average True Range (ATR)
    true_ranges = []
    for i in range(1, len(ohlcv_data)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        true_ranges.append(
            max(
                [
                    high - low,  # Current high-low
                    abs(high - prev_close),  # Current high - prev close
                    abs(low - prev_close),  # Current low - prev close
                ]
            )
        )
    atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0

    # Identify support and resistance levels using price clusters
    price_points = sorted(lows + highs)
    clusters = []
    current_cluster = [price_points[0]]

    for price in price_points[1:]:
        if price - current_cluster[-1] <= atr:  # Use ATR as threshold
            current_cluster.append(price)
        else:
            if len(current_cluster) > len(price_points) * 0.05:  # Min 5% of points
                clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [price]

    # Calculate RSI
    def calculate_rsi(prices: list, periods: int = 14) -> float:
        if len(prices) < periods:
            return 50  # Default value if not enough data

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-periods:]) / periods
        avg_loss = sum(losses[-periods:]) / periods

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closes)

    # Identify candlestick patterns
    patterns = []
    for i in range(len(ohlcv_data) - 1):
        # Doji
        if abs(opens[i] - closes[i]) <= abs(highs[i] - lows[i]) * 0.1:
            patterns.append(
                {
                    "pattern": "Doji",
                    "timestamp": datetime.fromtimestamp(timestamps[i]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

        # Hammer
        body = abs(opens[i] - closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        upper_wick = highs[i] - max(opens[i], closes[i])
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns.append(
                {
                    "pattern": "Hammer",
                    "timestamp": datetime.fromtimestamp(timestamps[i]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

    return {
        "price_trends": {
            "current_price": current_price,
            "price_change_percentage": price_change,
            "trend_direction": (
                "Uptrend"
                if price_change > 5
                else "Downtrend" if price_change < -5 else "Sideways"
            ),
            "trend_strength": abs(price_change) / volatility if volatility > 0 else 0,
        },
        "volatility_metrics": {
            "volatility_percentage": volatility,
            "atr": atr,
            "price_range": {
                "high": max(highs),
                "low": min(lows),
                "range_percentage": (max(highs) - min(lows)) / min(lows) * 100,
            },
        },
        "support_resistance": {
            "support_levels": sorted(
                [level for level in clusters if level < current_price]
            )[-3:],
            "resistance_levels": sorted(
                [level for level in clusters if level > current_price]
            )[:3],
        },
        "technical_indicators": {
            "rsi": rsi,
            "rsi_signal": (
                "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            ),
            "price_vs_sma": {
                "above_20_sma": (
                    current_price > sum(closes[-20:]) / 20
                    if len(closes) >= 20
                    else None
                ),
                "above_50_sma": (
                    current_price > sum(closes[-50:]) / 50
                    if len(closes) >= 50
                    else None
                ),
            },
        },
        "candlestick_patterns": patterns,
        "analysis_period": {
            "start": datetime.fromtimestamp(timestamps[-1]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": datetime.fromtimestamp(timestamps[0]).strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def calculate_fibonacci_levels(ohlcv_data: list) -> dict:
    """
    Calculate Fibonacci retracement and extension levels

    Args:
        ohlcv_data (list): OHLCV data points

    Returns:
        dict: Fibonacci levels and their prices
    """
    highs = [point[2] for point in ohlcv_data]
    lows = [point[3] for point in ohlcv_data]

    swing_high = max(highs)
    swing_low = min(lows)
    price_range = swing_high - swing_low

    # Common Fibonacci ratios
    fib_ratios = {
        "Extension 1.618": 1.618,
        "Extension 1.272": 1.272,
        "Extension 1.000": 1.000,
        "Retracement 0.786": 0.786,
        "Retracement 0.618": 0.618,
        "Retracement 0.500": 0.500,
        "Retracement 0.382": 0.382,
        "Retracement 0.236": 0.236,
    }

    fib_levels = {}
    for name, ratio in fib_ratios.items():
        if "Extension" in name:
            fib_levels[name] = swing_high + (price_range * (ratio - 1))
        else:
            fib_levels[name] = swing_high - (price_range * ratio)

    return fib_levels


def calculate_macd(
    prices: list, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        prices (list): List of closing prices
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period

    Returns:
        dict: MACD line, signal line, and histogram values
    """

    def ema(data: list, period: int) -> list:
        multiplier = 2 / (period + 1)
        ema_values = [data[0]]
        for price in data[1:]:
            ema_values.append(
                (price * multiplier) + (ema_values[-1] * (1 - multiplier))
            )
        return ema_values

    # Calculate MACD line
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)
    macd_line = [
        fast - slow for fast, slow in zip(fast_ema[-len(slow_ema) :], slow_ema)
    ]

    # Calculate signal line
    signal_line = ema(macd_line, signal_period)

    # Calculate histogram
    histogram = [
        macd - signal
        for macd, signal in zip(macd_line[-len(signal_line) :], signal_line)
    ]

    return {
        "macd_line": macd_line[-len(histogram) :],
        "signal_line": signal_line,
        "histogram": histogram,
    }


def calculate_bollinger_bands(
    prices: list, period: int = 20, std_dev: float = 2.0
) -> dict:
    """
    Calculate Bollinger Bands

    Args:
        prices (list): List of closing prices
        period (int): Moving average period
        std_dev (float): Number of standard deviations

    Returns:
        dict: Upper band, middle band (SMA), and lower band values
    """
    if len(prices) < period:
        return None

    bands = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        sma = sum(window) / period
        std = (sum((x - sma) ** 2 for x in window) / period) ** 0.5

        bands.append(
            {
                "middle_band": sma,
                "upper_band": sma + (std_dev * std),
                "lower_band": sma - (std_dev * std),
            }
        )

    return bands


def calculate_vwap(ohlcv_data: list) -> list:
    """
    Calculate Volume Weighted Average Price (VWAP)

    Args:
        ohlcv_data (list): OHLCV data points

    Returns:
        list: VWAP values
    """
    cumulative_pv = 0
    cumulative_volume = 0
    vwap_values = []

    for point in ohlcv_data:
        typical_price = (point[2] + point[3] + point[4]) / 3  # (high + low + close) / 3
        volume = point[5]

        cumulative_pv += typical_price * volume
        cumulative_volume += volume

        vwap = (
            cumulative_pv / cumulative_volume
            if cumulative_volume > 0
            else typical_price
        )
        vwap_values.append(vwap)

    return vwap_values


def calculate_momentum_indicators(ohlcv_data: list) -> dict:
    """
    Calculate various momentum indicators

    Args:
        ohlcv_data (list): OHLCV data points

    Returns:
        dict: Various momentum indicators
    """
    closes = [point[4] for point in ohlcv_data]

    # Rate of Change (ROC)
    period = 14
    roc = []
    for i in range(period, len(closes)):
        roc_val = ((closes[i] - closes[i - period]) / closes[i - period]) * 100
        roc.append(roc_val)

    # Stochastic Oscillator
    def calculate_stochastic(
        high: float, low: float, close: float, period: int = 14
    ) -> tuple:
        lowest_low = min(low)
        highest_high = max(high)
        k = (
            100 * (close - lowest_low) / (highest_high - lowest_low)
            if highest_high != lowest_low
            else 0
        )
        return k

    highs = [point[2] for point in ohlcv_data]
    lows = [point[3] for point in ohlcv_data]

    stoch_k = []
    for i in range(period, len(closes)):
        k = calculate_stochastic(highs[i - period : i], lows[i - period : i], closes[i])
        stoch_k.append(k)

    # Simple moving average of Stochastic K to get D
    stoch_d = []
    d_period = 3
    for i in range(d_period - 1, len(stoch_k)):
        d = sum(stoch_k[i - d_period + 1 : i + 1]) / d_period
        stoch_d.append(d)

    return {
        "roc": roc,
        "stochastic": {"k_line": stoch_k, "d_line": stoch_d},
        "momentum": [
            closes[i] - closes[i - period] for i in range(period, len(closes))
        ],
        "interpretation": {
            "roc_signal": "Bullish" if roc[-1] > 0 else "Bearish",
            "stochastic_signal": (
                "Overbought"
                if stoch_k[-1] > 80
                else "Oversold" if stoch_k[-1] < 20 else "Neutral"
            ),
        },
    }


def analyze_technical_indicators(ohlcv_data: list) -> dict:
    """
    Perform comprehensive technical analysis using all indicators

    Args:
        ohlcv_data (list): OHLCV data points

    Returns:
        dict: Complete technical analysis with all indicators
    """
    closes = [point[4] for point in ohlcv_data]

    # Calculate all indicators
    fib_levels = calculate_fibonacci_levels(ohlcv_data)
    macd_data = calculate_macd(closes)
    bollinger = calculate_bollinger_bands(closes)
    vwap = calculate_vwap(ohlcv_data)
    momentum = calculate_momentum_indicators(ohlcv_data)

    # Combine signals for trading decisions
    current_price = closes[-1]
    bb_latest = bollinger[-1] if bollinger else None

    signals = []

    # MACD signals
    if macd_data["histogram"][-1] > 0 and macd_data["histogram"][-2] <= 0:
        signals.append("MACD bullish crossover")
    elif macd_data["histogram"][-1] < 0 and macd_data["histogram"][-2] >= 0:
        signals.append("MACD bearish crossover")

    # Bollinger Bands signals
    if bb_latest:
        if current_price > bb_latest["upper_band"]:
            signals.append("Price above upper Bollinger Band - potential overbought")
        elif current_price < bb_latest["lower_band"]:
            signals.append("Price below lower Bollinger Band - potential oversold")

    # VWAP signals
    if current_price > vwap[-1]:
        signals.append("Price above VWAP - bullish")
    else:
        signals.append("Price below VWAP - bearish")

    return {
        "fibonacci_levels": fib_levels,
        "macd": {
            "values": macd_data,
            "trend": "Bullish" if macd_data["histogram"][-1] > 0 else "Bearish",
        },
        "bollinger_bands": {
            "current": bb_latest,
            "volatility": (
                (bb_latest["upper_band"] - bb_latest["lower_band"])
                / bb_latest["middle_band"]
                if bb_latest
                else None
            ),
        },
        "vwap": {
            "current": vwap[-1],
            "trend": "Above VWAP" if current_price > vwap[-1] else "Below VWAP",
        },
        "momentum_indicators": momentum,
        "trading_signals": {
            "signals": signals,
            "strength": len([s for s in signals if "bullish" in s.lower()])
            - len([s for s in signals if "bearish" in s.lower()]),
            "overall_bias": (
                "Bullish"
                if len([s for s in signals if "bullish" in s.lower()])
                > len([s for s in signals if "bearish" in s.lower()])
                else "Bearish"
            ),
        },
    }


def get_essential_indicators(full_analysis) -> dict:
    """
    Extract only essential technical indicators (Bollinger Bands, VWAP, and trading signals)

    Args:
        ohlcv_data (list): OHLCV data points

    Returns:
        dict: Essential technical indicators and signals
    """

    # Extract only the needed components
    essential_analysis = {
        "current_price": full_analysis["macd"]["values"]["macd_line"][
            -1
        ],  # Get current price
        "bollinger_bands": {
            "current": full_analysis["bollinger_bands"]["current"],
            "volatility": full_analysis["bollinger_bands"]["volatility"],
        },
        "vwap": {
            "current": round(full_analysis["vwap"]["current"], 6),
            "trend": full_analysis["vwap"]["trend"],
        },
        "trading_signals": {
            "signals": full_analysis["trading_signals"]["signals"][
                :3
            ],  # Only keep last 3 signals
            "overall_bias": full_analysis["trading_signals"]["overall_bias"],
        },
    }

    return essential_analysis


def visualize_technical_analysis(
    ohlcv_data: list, analysis_results: dict
) -> plt.Figure:
    """
    Create visualizations for technical analysis results using standard libraries

    Args:
        ohlcv_data (list): OHLCV data points
        analysis_results (dict): Output from analyze_technical_indicators

    Returns:
        plt.Figure: The generated matplotlib figure containing the visualizations
    """

    try:
        # Convert data to pandas DataFrame
        timestamps = [datetime.fromtimestamp(point[0]) for point in ohlcv_data]
        df = pd.DataFrame(
            {
                "Open": [point[1] for point in ohlcv_data],
                "High": [point[2] for point in ohlcv_data],
                "Low": [point[3] for point in ohlcv_data],
                "Close": [point[4] for point in ohlcv_data],
                "Volume": [point[5] for point in ohlcv_data],
            },
            index=timestamps,
        )

        # Set style
        plt.style.use("seaborn")

        # Create figure with subplots - adjust height_ratios and hspace
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(
            3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.3
        )  # Adjust height ratios and vertical spacing

        # 1. Price and Volume Chart
        ax1 = fig.add_subplot(gs[0, :])
        # Plot price
        ax1.plot(df.index, df["Close"], label="Price", color="blue", linewidth=1)

        # Add Bollinger Bands if available
        if "bollinger_bands" in analysis_results:
            bb_data = analysis_results["bollinger_bands"]
            if isinstance(bb_data, list) and bb_data:  # Check if it's a non-empty list
                upper_band = [b["upper_band"] for b in bb_data]
                middle_band = [b["middle_band"] for b in bb_data]
                lower_band = [b["lower_band"] for b in bb_data]

                ax1.plot(
                    df.index[-len(upper_band) :],
                    upper_band,
                    "r--",
                    label="Upper BB",
                    alpha=0.7,
                )
                ax1.plot(
                    df.index[-len(middle_band) :],
                    middle_band,
                    "g--",
                    label="Middle BB",
                    alpha=0.7,
                )
                ax1.plot(
                    df.index[-len(lower_band) :],
                    lower_band,
                    "b--",
                    label="Lower BB",
                    alpha=0.7,
                )

        # Add VWAP if available
        if "vwap" in analysis_results:
            vwap_values = analysis_results["vwap"]
            if isinstance(vwap_values, list):
                ax1.plot(
                    df.index[-len(vwap_values) :],
                    vwap_values,
                    "purple",
                    label="VWAP",
                    alpha=0.7,
                )

        # Add volume bars with reduced height
        volume_ax = ax1.twinx()
        volume_ax.bar(df.index, df["Volume"], alpha=0.3, color="gray", width=0.8)
        volume_ax.set_ylim(0, df["Volume"].max() * 3)  # Adjust volume scale

        ax1.set_title("Price Action with Technical Indicators")
        ax1.legend(loc="upper left")
        ax1.legend(loc="upper right")

        # 2. MACD
        if "macd" in analysis_results and "values" in analysis_results["macd"]:
            ax2 = fig.add_subplot(gs[1, 0])
            macd_data = analysis_results["macd"]["values"]

            macd_line = macd_data["macd_line"]
            signal_line = macd_data["signal_line"]
            histogram = macd_data["histogram"]

            # Plot MACD components
            ax2.plot(df.index[-len(macd_line) :], macd_line, label="MACD", color="blue")
            ax2.plot(
                df.index[-len(signal_line) :],
                signal_line,
                label="Signal",
                color="orange",
            )
            ax2.bar(
                df.index[-len(histogram) :],
                histogram,
                label="Histogram",
                color="gray",
                alpha=0.5,
            )

            ax2.set_title("MACD")
            ax2.legend()

        # 3. Stochastic Oscillator
        if "momentum_indicators" in analysis_results:
            ax3 = fig.add_subplot(gs[1, 1])
            momentum_data = analysis_results["momentum_indicators"]

            if "stochastic" in momentum_data:
                k_line = momentum_data["stochastic"]["k_line"]
                d_line = momentum_data["stochastic"]["d_line"]

                ax3.plot(
                    df.index[-len(k_line) :],
                    k_line,
                    label="Stochastic %K",
                    color="blue",
                )
                ax3.plot(
                    df.index[-len(d_line) :],
                    d_line,
                    label="Stochastic %D",
                    color="orange",
                )

                # Add overbought/oversold lines
                ax3.axhline(y=80, color="r", linestyle="--", alpha=0.5)
                ax3.axhline(y=20, color="g", linestyle="--", alpha=0.5)

                ax3.set_title("Stochastic Oscillator")
                ax3.legend()

        # 4. Rate of Change (ROC)
        if "momentum_indicators" in analysis_results:
            ax4 = fig.add_subplot(gs[2, 0])
            momentum_data = analysis_results["momentum_indicators"]

            if "roc" in momentum_data:
                roc_values = momentum_data["roc"]
                ax4.plot(
                    df.index[-len(roc_values) :],
                    roc_values,
                    label="Rate of Change",
                    color="purple",
                )
                ax4.axhline(y=0, color="r", linestyle="--", alpha=0.3)
                ax4.set_title("Rate of Change (ROC)")
                ax4.legend()

        # 5. Fibonacci Levels
        if "fibonacci_levels" in analysis_results:
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(df.index, df["Close"], color="black", alpha=0.7)

            fib_levels = analysis_results["fibonacci_levels"]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(fib_levels)))

            for (level_name, price), color in zip(fib_levels.items(), colors):
                ax5.axhline(
                    y=price,
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                    label=f"{level_name}: {price:.6f}",
                )

            ax5.set_title("Fibonacci Levels")
            ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add overall title with key signals
        if "trading_signals" in analysis_results:
            signals = analysis_results["trading_signals"].get("signals", [])
            bias = analysis_results["trading_signals"].get("overall_bias", "Unknown")

            # Get last two signals and format them nicely
            latest_signals = signals[-2:] if signals else ["No signals"]
            signal_text = "\n".join(latest_signals)  # Put each signal on new line

            plt.suptitle(
                "Technical Analysis Overview\n\n"  # Add extra newline for spacing
                + f"Overall Bias: {bias}\n\n"  # Add extra newline for spacing
                + f"Latest Signals:\n{signal_text}",
                fontsize=12,  # Reduced font size
                y=0.98,  # Moved title up slightly
                verticalalignment="top",  # Align from top
            )

        # Adjust layout to prevent text overlap but maintain tighter spacing
        plt.tight_layout(
            rect=[0, 0, 1, 0.95], h_pad=0.8
        )  # Reduced padding between plots
        return fig

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback

        traceback.print_exc()
        return None

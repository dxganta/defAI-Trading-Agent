from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from .utils import get_web3, ERC20_ABI
import requests
from matplotlib.dates import DateFormatter


def get_ohlcv_data(token_address: str) -> dict:
    """
    Get OHLCV (Open, High, Low, Close, Volume) data for a token from geckoterminal

    Args:
        token_address (str): The token contract address

    Returns:
        list: A list of OHLCV data points in ASCENDING order (oldest first, newest last):
        [timestamp, open, high, low, close, volume]
    """

    base_url = "https://api.geckoterminal.com/api/v2"

    # get the top pools for the token
    pools_url = f"{base_url}/networks/eth/tokens/{token_address}/pools"
    response = requests.get(pools_url)

    top_pool = response.json()["data"][0]["attributes"]["address"]

    # Get OHLCV data
    # For 1 month of data, using 1h interval as it's the best balance for AI analysis
    ohlcv_url = f"{base_url}/networks/eth/pools/{top_pool}/ohlcv/hour"

    params = {"currency": "usd", "limit": 1000}  # Get prices in USD

    response = requests.get(ohlcv_url, params=params)

    # Data from API is typically in descending order (newest first)
    # We reverse it to ensure ascending order (oldest first, newest last)
    return response.json()["data"]["attributes"]["ohlcv_list"][::-1]


def calculate_average_daily_volume(ohlcv_data: list) -> float:
    """
    Calculate the average daily trading volume from hourly OHLCV data

    Args:
        ohlcv_data (list): List of hourly OHLCV data points where each point is
                          [timestamp, open, high, low, close, volume]

    Returns:
        float: Average daily trading volume (sum of 24 hourly volumes per day,
               averaged across all days)
    """
    # Create a dictionary to store daily volumes
    daily_volumes = {}

    for datapoint in ohlcv_data:
        timestamp = datapoint[0]
        volume = datapoint[5]

        # Convert timestamp to date string (YYYY-MM-DD)
        date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        # Add volume to daily total
        if date in daily_volumes:
            daily_volumes[date] += volume
        else:
            daily_volumes[date] = volume

    # Calculate average across all days
    total_volume = sum(daily_volumes.values())
    num_days = len(daily_volumes)

    return total_volume / num_days if num_days > 0 else 0


def analyze_volume_trends(ohlcv_data: list, ma_periods: tuple = (24, 72)) -> dict:
    """
    Analyze trading volume trends from OHLCV data

    Args:
        ohlcv_data (list): List of hourly OHLCV data points where each point is
                          [timestamp, open, high, low, close, volume]
        ma_periods (tuple): Periods for short and long moving averages in hours
                          (default: 24h and 72h)

    Returns:
        dict: Volume trend analysis with the following metrics:
            - recent_trend: Direction of volume trend ('increasing', 'decreasing', or 'stable')
            - volume_volatility: Coefficient of variation of volumes
            - moving_averages: Short and long-term moving averages
            - peak_volume: Highest volume in the period
            - peak_volume_time: Timestamp of highest volume
            - trend_strength: Strength of trend from 0-1 based on MA crossovers
            - volume_distribution: Distribution of volume across time periods
    """
    if not ohlcv_data:
        return None

    # Extract volumes and timestamps
    volumes = [point[5] for point in ohlcv_data]
    timestamps = [point[0] for point in ohlcv_data]

    # Calculate moving averages
    short_ma = []
    long_ma = []
    short_period, long_period = ma_periods

    for i in range(len(volumes)):
        if i >= short_period:
            short_ma.append(sum(volumes[i - short_period : i]) / short_period)
        if i >= long_period:
            long_ma.append(sum(volumes[i - long_period : i]) / long_period)

    # Calculate volume volatility (coefficient of variation)
    mean_volume = sum(volumes) / len(volumes)
    std_dev = (sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)) ** 0.5
    volatility = std_dev / mean_volume if mean_volume > 0 else 0

    # Find peak volume
    peak_volume = max(volumes)
    peak_index = volumes.index(peak_volume)
    peak_timestamp = timestamps[peak_index]

    # Determine trend direction using recent moving averages
    if len(short_ma) > 2:
        recent_short_ma = short_ma[-3:]
        if recent_short_ma[-1] > recent_short_ma[0] * 1.05:  # 5% increase
            trend = "increasing"
        elif recent_short_ma[-1] < recent_short_ma[0] * 0.95:  # 5% decrease
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient data"

    # Calculate trend strength based on MA crossovers
    trend_strength = 0
    if len(short_ma) > 0 and len(long_ma) > 0:
        # Compare last values of MAs
        if short_ma[-1] > long_ma[-1]:
            trend_strength = min((short_ma[-1] / long_ma[-1] - 1), 1)
        else:
            trend_strength = max((short_ma[-1] / long_ma[-1] - 1), -1)

    # Analyze volume distribution
    quartiles = {
        "q1": sorted(volumes)[len(volumes) // 4],
        "median": sorted(volumes)[len(volumes) // 2],
        "q3": sorted(volumes)[3 * len(volumes) // 4],
    }

    return {
        "recent_trend": trend,
        "volume_volatility": volatility,
        "moving_averages": {
            f"{short_period}h": short_ma[-1] if short_ma else None,
            f"{long_period}h": long_ma[-1] if long_ma else None,
        },
        "peak_volume": {
            "amount": peak_volume,
            "timestamp": datetime.fromtimestamp(peak_timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
        "trend_strength": trend_strength,
        "volume_distribution": quartiles,
        "analysis_period": {
            "start": datetime.fromtimestamp(timestamps[-1]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": datetime.fromtimestamp(timestamps[0]).strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def detect_wash_trading(ohlcv_data: list, threshold_multiplier: float = 2.0) -> dict:
    """
    Detect potential wash trading patterns in OHLCV data

    Args:
        ohlcv_data (list): List of hourly OHLCV data points where each point is
                          [timestamp, open, high, low, close, volume]
        threshold_multiplier (float): Multiplier for standard deviation to detect anomalies
                                    (default: 2.0)

    Returns:
        dict: Wash trading analysis with the following metrics:
            {
                'summary': {
                    'wash_trading_risk': str,      # 'High', 'Medium', 'Low'
                    'confidence_score': float,     # 0-1 score of confidence in detection
                    'suspicious_periods': int      # Number of suspicious periods detected
                },
                'patterns': {
                    'volume_price_divergence': [   # Periods where volume and price patterns diverge
                        {
                            'timestamp': str,
                            'volume': float,
                            'price_change': float,
                            'divergence_score': float
                        },
                        ...
                    ],
                    'volume_anomalies': [         # Unusual volume patterns
                        {
                            'timestamp': str,
                            'volume': float,
                            'expected_volume': float,
                            'deviation': float
                        },
                        ...
                    ]
                },
                'metrics': {
                    'price_volume_correlation': float,  # Correlation between price and volume
                    'volume_consistency_score': float,  # Measure of volume pattern consistency
                    'price_impact_score': float        # Measure of volume's impact on price
                },
                'analysis_period': {
                    'start': str,
                    'end': str
                }
            }
    """
    if not ohlcv_data:
        return None

    # Extract data
    volumes = [point[5] for point in ohlcv_data]
    timestamps = [point[0] for point in ohlcv_data]
    prices = [point[4] for point in ohlcv_data]  # Using close prices

    # Calculate baseline metrics
    mean_volume = sum(volumes) / len(volumes)
    std_dev_volume = (
        sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
    ) ** 0.5

    # Calculate price changes
    price_changes = [
        ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
        for i in range(1, len(prices))
    ]
    price_changes.insert(0, 0)  # Add 0 for first point

    # Detect volume anomalies
    volume_anomalies = []
    for i, volume in enumerate(volumes):
        deviation = (volume - mean_volume) / std_dev_volume if std_dev_volume > 0 else 0
        if abs(deviation) > threshold_multiplier:
            volume_anomalies.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamps[i]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "volume": volume,
                    "expected_volume": mean_volume,
                    "deviation": deviation,
                }
            )

    # Detect price-volume divergence
    divergence_patterns = []
    for i in range(len(volumes)):
        if i > 0:
            volume_change = (
                (volumes[i] - volumes[i - 1]) / volumes[i - 1]
                if volumes[i - 1] > 0
                else 0
            )
            price_change = price_changes[i]

            # Check for divergence (high volume with little price change or vice versa)
            if (abs(volume_change) > 0.5 and abs(price_change) < 0.1) or (
                abs(volume_change) < 0.1 and abs(price_change) > 0.5
            ):
                divergence_score = abs(volume_change - price_change)
                divergence_patterns.append(
                    {
                        "timestamp": datetime.fromtimestamp(timestamps[i]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "volume": volumes[i],
                        "price_change": price_change,
                        "divergence_score": divergence_score,
                    }
                )

    # Calculate price-volume correlation
    def calculate_correlation(x, y):
        if len(x) != len(y):
            return 0
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
        std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5
        if std_x == 0 or std_y == 0:
            return 0
        return covariance / (std_x * std_y)

    price_volume_correlation = calculate_correlation(
        volumes, [abs(p) for p in price_changes]
    )

    # Calculate volume consistency score
    volume_changes = [
        abs((volumes[i] - volumes[i - 1]) / volumes[i - 1]) if volumes[i - 1] > 0 else 0
        for i in range(1, len(volumes))
    ]
    volume_consistency = 1 - (sum(volume_changes) / len(volume_changes))

    # Calculate wash trading risk score
    risk_factors = [
        len(volume_anomalies) / len(volumes),  # Proportion of anomalous volumes
        len(divergence_patterns) / len(volumes),  # Proportion of divergent patterns
        1
        - abs(
            price_volume_correlation
        ),  # Low correlation indicates potential wash trading
        1 - volume_consistency,  # High volatility indicates potential wash trading
    ]
    risk_score = sum(risk_factors) / len(risk_factors)

    # Determine risk level
    if risk_score > 0.7:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "summary": {
            "wash_trading_risk": risk_level,
            "confidence_score": 1
            - (risk_score / 2),  # Convert risk score to confidence
            "suspicious_periods": len(volume_anomalies) + len(divergence_patterns),
        },
        "patterns": {
            "volume_price_divergence": sorted(
                divergence_patterns, key=lambda x: x["divergence_score"], reverse=True
            ),
            "volume_anomalies": sorted(
                volume_anomalies, key=lambda x: abs(x["deviation"]), reverse=True
            ),
        },
        "metrics": {
            "price_volume_correlation": price_volume_correlation,
            "volume_consistency_score": volume_consistency,
            "price_impact_score": 1 - abs(price_volume_correlation),
        },
        "analysis_period": {
            "start": datetime.fromtimestamp(timestamps[-1]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": datetime.fromtimestamp(timestamps[0]).strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def analyze_volume_seasonality(ohlcv_data: list) -> dict:
    """
    Analyze trading volume seasonality and liquidity patterns

    Args:
        ohlcv_data (list): List of hourly OHLCV data points where each point is
                          [timestamp, open, high, low, close, volume]

    Returns:
        dict: Volume seasonality analysis with the following metrics:
            {
                'hourly_patterns': {
                    'peak_trading_hours': list,    # Hours with highest average volume
                    'low_volume_hours': list,      # Hours with lowest average volume
                    'hour_volatility': dict        # Volume volatility by hour
                },
                'liquidity_metrics': {
                    'average_hourly_volume': float,
                    'volume_stability': float,     # Measure of volume consistency
                    'liquidity_score': float      # Overall liquidity rating
                },
                'volume_concentration': {
                    'top_5_hours_volume': float,  # % of volume in top 5 hours
                    'gini_coefficient': float     # Measure of volume distribution
                },
                'slippage_estimation': {
                    'low_volume_slippage': float,  # Estimated slippage in low volume
                    'high_volume_slippage': float  # Estimated slippage in high volume
                }
            }
    """
    if not ohlcv_data:
        return None

    # Group volumes by hour
    hourly_volumes = {}
    for point in ohlcv_data:
        timestamp = point[0]
        volume = point[5]
        hour = datetime.fromtimestamp(timestamp).hour

        if hour not in hourly_volumes:
            hourly_volumes[hour] = []
        hourly_volumes[hour].append(volume)

    # Calculate hourly statistics
    hourly_stats = {}
    for hour, volumes in hourly_volumes.items():
        mean_volume = sum(volumes) / len(volumes)
        std_dev = (sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)) ** 0.5

        hourly_stats[hour] = {
            "mean_volume": mean_volume,
            "std_dev": std_dev,
            "volatility": std_dev / mean_volume if mean_volume > 0 else 0,
            "sample_count": len(volumes),
        }

    # Identify peak and low volume hours
    sorted_hours = sorted(
        hourly_stats.items(), key=lambda x: x[1]["mean_volume"], reverse=True
    )

    peak_hours = [hour for hour, _ in sorted_hours[:5]]
    low_hours = [hour for hour, _ in sorted_hours[-5:]]

    # Calculate volume concentration
    total_volume = sum(stats["mean_volume"] for stats in hourly_stats.values())
    top_5_volume = sum(hourly_stats[hour]["mean_volume"] for hour in peak_hours)
    volume_concentration = (top_5_volume / total_volume) if total_volume > 0 else 0

    # Calculate liquidity metrics
    avg_hourly_volume = total_volume / 24
    volume_stability = 1 - (
        sum(stats["volatility"] for stats in hourly_stats.values()) / 24
    )

    # Estimate slippage based on volume patterns
    # Using a simple model where slippage is inversely proportional to volume
    base_slippage = 0.001  # 0.1% base slippage
    low_vol_slippage = base_slippage * (
        avg_hourly_volume / min(stats["mean_volume"] for stats in hourly_stats.values())
    )
    high_vol_slippage = base_slippage * (
        avg_hourly_volume / max(stats["mean_volume"] for stats in hourly_stats.values())
    )

    # Calculate liquidity score (0-1)
    liquidity_factors = [
        volume_stability,
        1 - volume_concentration,  # Lower concentration is better
        min(1, avg_hourly_volume / 10000),  # Normalize volume
    ]
    liquidity_score = sum(liquidity_factors) / len(liquidity_factors)

    return {
        "hourly_patterns": {
            "peak_trading_hours": [f"{hour:02d}:00" for hour in peak_hours],
            "low_volume_hours": [f"{hour:02d}:00" for hour in low_hours],
            "hour_volatility": {
                f"{hour:02d}:00": stats["volatility"]
                for hour, stats in hourly_stats.items()
            },
        },
        "liquidity_metrics": {
            "average_hourly_volume": avg_hourly_volume,
            "volume_stability": volume_stability,
            "liquidity_score": liquidity_score,
        },
        "volume_concentration": {
            "top_5_hours_volume_percentage": volume_concentration * 100,
            "hourly_distribution": {
                f"{hour:02d}:00": stats["mean_volume"] / total_volume
                for hour, stats in hourly_stats.items()
            },
        },
        "slippage_estimation": {
            "low_volume_slippage": low_vol_slippage,
            "high_volume_slippage": high_vol_slippage,
            "slippage_volatility": low_vol_slippage / high_vol_slippage,
        },
        "trading_recommendations": {
            "best_hours_to_trade": [f"{hour:02d}:00" for hour in peak_hours[:3]],
            "hours_to_avoid": [f"{hour:02d}:00" for hour in low_hours[:3]],
            "liquidity_warning": (
                "Low"
                if liquidity_score < 0.3
                else "Medium" if liquidity_score < 0.7 else "High"
            ),
        },
    }


def analyze_token_volume_comprehensive(token_address: str, ohlcv_data: list) -> dict:
    """
    Perform comprehensive volume analysis combining all volume-related metrics

    Args:
        token_address (str): The token contract address to analyze
        ohlcv_data (list): List of OHLCV data points

    Returns:
        dict: Comprehensive volume analysis with structured data suitable for AI processing:
            {
                'token_info': {
                    'address': str,
                    'symbol': str,
                    'analysis_period': {
                        'start': str,
                        'end': str,
                        'duration_days': int
                    }
                },
                'volume_metrics': {
                    'basic_metrics': {
                        'average_daily_volume': float,
                        'average_hourly_volume': float,
                        'peak_volume': float,
                        'min_volume': float,
                        'volume_std_dev': float
                    },
                    'trend_analysis': {
                        'current_trend': str,
                        'trend_strength': float,
                        'moving_averages': dict,
                        'volatility': float
                    }
                },
                'trading_patterns': {
                    'seasonality': {
                        'peak_hours': list,
                        'low_volume_hours': list,
                        'hourly_distribution': dict
                    },
                    'wash_trading_indicators': {
                        'risk_level': str,
                        'confidence_score': float,
                        'suspicious_patterns': list
                    }
                },
                'liquidity_analysis': {
                    'liquidity_score': float,
                    'volume_stability': float,
                    'slippage_estimates': dict,
                    'concentration_metrics': dict
                },
                'anomaly_detection': {
                    'volume_anomalies': list,
                    'price_volume_divergences': list,
                    'unusual_patterns': list
                },
                'ai_insights': {
                    'key_findings': list,
                    'risk_factors': list,
                    'trading_recommendations': list,
                    'market_health_indicators': dict
                }
            }
    """
    try:
        web3 = get_web3()
        # Get token contract for metadata
        token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
        token_symbol = token_contract.functions.symbol().call()

        # Perform all analyses
        volume_trends = analyze_volume_trends(ohlcv_data)
        wash_trading = detect_wash_trading(ohlcv_data)
        seasonality = analyze_volume_seasonality(ohlcv_data)
        avg_daily_volume = calculate_average_daily_volume(ohlcv_data)

        # Extract timestamps for period analysis
        timestamps = [point[0] for point in ohlcv_data]
        start_time = datetime.fromtimestamp(min(timestamps))
        end_time = datetime.fromtimestamp(max(timestamps))
        duration_days = (end_time - start_time).days

        # Extract volumes for basic metrics
        volumes = [point[5] for point in ohlcv_data]

        # Generate key findings and recommendations
        key_findings = []
        risk_factors = []
        recommendations = []

        # Analyze volume trends
        if volume_trends["recent_trend"] == "increasing":
            key_findings.append("Volume showing upward trend")
        elif volume_trends["recent_trend"] == "decreasing":
            key_findings.append("Volume showing downward trend")
            risk_factors.append("Declining volume may indicate reduced market interest")

        # Analyze wash trading risk
        if wash_trading["summary"]["wash_trading_risk"] == "High":
            risk_factors.append("High risk of wash trading detected")
            recommendations.append(
                "Exercise caution due to potential market manipulation"
            )

        # Analyze liquidity
        if seasonality["liquidity_metrics"]["liquidity_score"] < 0.5:
            risk_factors.append("Low liquidity conditions")
            recommendations.append("Consider trading during peak volume hours")

        # Structured comprehensive analysis
        return {
            "token_info": {
                "address": token_address,
                "symbol": token_symbol,
                "analysis_period": {
                    "start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_days": duration_days,
                },
            },
            "volume_metrics": {
                "basic_metrics": {
                    "average_daily_volume": avg_daily_volume,
                    "average_hourly_volume": sum(volumes) / len(volumes),
                    "peak_volume": max(volumes),
                    "min_volume": min(volumes),
                    "volume_std_dev": (
                        sum((v - avg_daily_volume / 24) ** 2 for v in volumes)
                        / len(volumes)
                    )
                    ** 0.5,
                },
                "trend_analysis": {
                    "current_trend": volume_trends["recent_trend"],
                    "trend_strength": volume_trends["trend_strength"],
                    "moving_averages": volume_trends["moving_averages"],
                    "volatility": volume_trends["volume_volatility"],
                },
            },
            "trading_patterns": {
                "seasonality": {
                    "peak_hours": seasonality["hourly_patterns"]["peak_trading_hours"],
                    "low_volume_hours": seasonality["hourly_patterns"][
                        "low_volume_hours"
                    ],
                    "hourly_distribution": seasonality["volume_concentration"][
                        "hourly_distribution"
                    ],
                },
                "wash_trading_indicators": {
                    "risk_level": wash_trading["summary"]["wash_trading_risk"],
                    "confidence_score": wash_trading["summary"]["confidence_score"],
                    "suspicious_patterns": wash_trading["patterns"][
                        "volume_price_divergence"
                    ],
                },
            },
            "liquidity_analysis": {
                "liquidity_score": seasonality["liquidity_metrics"]["liquidity_score"],
                "volume_stability": seasonality["liquidity_metrics"][
                    "volume_stability"
                ],
                "slippage_estimates": seasonality["slippage_estimation"],
                "concentration_metrics": {
                    "top_hours_concentration": seasonality["volume_concentration"][
                        "top_5_hours_volume_percentage"
                    ],
                    "volume_distribution": volume_trends["volume_distribution"],
                },
            },
            "anomaly_detection": {
                "volume_anomalies": wash_trading["patterns"]["volume_anomalies"],
                "price_volume_divergences": wash_trading["patterns"][
                    "volume_price_divergence"
                ],
                "unusual_patterns": [
                    p
                    for p in wash_trading["patterns"]["volume_price_divergence"]
                    if p["divergence_score"] > 0.5
                ],
            },
            "ai_insights": {
                "key_findings": key_findings,
                "risk_factors": risk_factors,
                "trading_recommendations": recommendations,
                "market_health_indicators": {
                    "overall_health": (
                        "High"
                        if len(risk_factors) == 0
                        else "Medium" if len(risk_factors) <= 2 else "Low"
                    ),
                    "liquidity_rating": seasonality["trading_recommendations"][
                        "liquidity_warning"
                    ],
                    "manipulation_risk": wash_trading["summary"]["wash_trading_risk"],
                    "volume_stability": (
                        "High"
                        if seasonality["liquidity_metrics"]["volume_stability"] > 0.7
                        else (
                            "Medium"
                            if seasonality["liquidity_metrics"]["volume_stability"]
                            > 0.4
                            else "Low"
                        )
                    ),
                },
            },
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}", "token_address": token_address}


def visualize_volume_analysis(ohlcv_data: list, analysis_results: dict) -> plt.Figure:
    """
    Create visualizations from volume analysis results

    Args:
        ohlcv_data (list): Raw OHLCV data used for analysis
        analysis_results (dict): Output from analyze_token_volume_comprehensive

    Returns:
        plt.Figure: The generated matplotlib figure containing all visualizations
    """
    # Extract data
    volumes = [point[5] for point in ohlcv_data]
    timestamps = [datetime.fromtimestamp(point[0]) for point in ohlcv_data]
    prices = [point[4] for point in ohlcv_data]

    # Calculate moving averages
    short_period, long_period = 24, 72
    short_ma = []
    long_ma = []
    short_ma_timestamps = []
    long_ma_timestamps = []

    for i in range(len(volumes)):
        if i >= short_period - 1:
            short_ma.append(sum(volumes[i - (short_period - 1) : i + 1]) / short_period)
            short_ma_timestamps.append(timestamps[i])
        if i >= long_period - 1:
            long_ma.append(sum(volumes[i - (long_period - 1) : i + 1]) / long_period)
            long_ma_timestamps.append(timestamps[i])

    # Create figure with subplots
    plt.style.use("seaborn")
    fig = plt.figure(figsize=(20, 15))
    # Using a simpler approach with more space at the top
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, top=0.85)

    # 1. Volume Over Time with Moving Averages
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, volumes, label="Volume", alpha=0.5, color="gray")

    # Plot moving averages as lines
    if short_ma:
        ax1.plot(short_ma_timestamps, short_ma, label="24h MA", color="blue")
    if long_ma:
        ax1.plot(long_ma_timestamps, long_ma, label="72h MA", color="red")

    ax1.set_title("Volume Over Time with Moving Averages")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Volume")
    ax1.legend()

    # Add better axis formatting
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Format volume numbers with K/M suffixes
    def format_volume(x, p):
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        elif x >= 1e3:
            return f"{x/1e3:.1f}K"
        return f"{x:.1f}"

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_volume))

    # 2. Hourly Volume Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    hourly_dist = analysis_results["trading_patterns"]["seasonality"][
        "hourly_distribution"
    ]
    hours = list(hourly_dist.keys())
    volumes_by_hour = list(hourly_dist.values())

    ax2.bar(hours, volumes_by_hour)
    ax2.set_title("Hourly Volume Distribution")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Volume Percentage")
    plt.xticks(rotation=45)

    # 3. Volume Anomalies
    ax3 = fig.add_subplot(gs[1, 1])
    anomalies = analysis_results["anomaly_detection"]["volume_anomalies"]

    ax3.plot(timestamps, volumes, color="gray", alpha=0.5, label="Volume")
    if anomalies:
        try:
            anomaly_times = [
                datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
                for a in anomalies
            ]
            anomaly_volumes = [a["volume"] for a in anomalies]
            ax3.scatter(
                anomaly_times, anomaly_volumes, color="red", label="Anomalies", s=100
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not plot anomalies due to {str(e)}")

    ax3.set_title("Volume Anomalies")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Volume")
    ax3.legend()

    # Add better axis formatting
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Format volume numbers with K/M suffixes
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_volume))

    # 4. Price-Volume Correlation
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(volumes, prices, alpha=0.5)
    ax4.set_title("Price vs Volume Correlation")
    ax4.set_xlabel("Volume")
    ax4.set_ylabel("Price")

    # 5. Volume Stability Heatmap
    ax5 = fig.add_subplot(gs[2, 1])
    stability_data = []
    if len(volumes) >= 24:  # Only create heatmap if we have at least a day of data
        for i in range(0, len(volumes) - 23, 24):  # Ensure we have complete days
            if i + 24 <= len(volumes):
                day_data = volumes[i : i + 24]
                # Normalize the day's data to make patterns more visible
                day_mean = sum(day_data) / len(day_data)
                if day_mean > 0:  # Avoid division by zero
                    day_data = [v / day_mean for v in day_data]
                stability_data.append(day_data)

    if stability_data:
        sns.heatmap(
            stability_data,
            cmap="YlOrRd",
            ax=ax5,
            xticklabels=[f"{i:02d}:00" for i in range(24)],
            yticklabels=[f"Day {i+1}" for i in range(len(stability_data))],
            cbar_kws={"label": "Relative Volume"},
        )

    # Create a single title with line breaks instead of separate text elements
    avg_volume = analysis_results["volume_metrics"]["basic_metrics"][
        "average_hourly_volume"
    ]
    volume_trend = analysis_results["volume_metrics"]["trend_analysis"]["current_trend"]
    liquidity_score = analysis_results["liquidity_analysis"]["liquidity_score"]

    title_text = (
        f"Volume Analysis for {analysis_results['token_info']['symbol']}\n"
        f"Avg Vol: {avg_volume:,.2f} | Trend: {volume_trend} | Liquidity Score: {liquidity_score:.2f}"
    )
    fig.suptitle(title_text, fontsize=14, y=0.93, linespacing=1.5)

    # Adjust layout to ensure no overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def get_volume_snapshot(full_analysis: dict) -> dict:
    """
    Get current volume metrics and health indicators without temporal analysis

    Args:
        token_address (str): The token contract address to analyze
        ohlcv_data (list): List of OHLCV data points

    Returns:
        dict: Current volume snapshot with key metrics
    """
    if "error" in full_analysis:
        return full_analysis

    return {
        "token_info": {
            "address": full_analysis["token_info"]["address"],
            "symbol": full_analysis["token_info"]["symbol"],
        },
        "current_metrics": {
            "average_volume": full_analysis["volume_metrics"]["basic_metrics"][
                "average_hourly_volume"
            ],
            "peak_volume": full_analysis["volume_metrics"]["basic_metrics"][
                "peak_volume"
            ],
            "min_volume": full_analysis["volume_metrics"]["basic_metrics"][
                "min_volume"
            ],
            "volume_std_dev": full_analysis["volume_metrics"]["basic_metrics"][
                "volume_std_dev"
            ],
            "volume_stability": full_analysis["liquidity_analysis"]["volume_stability"],
            "liquidity_score": full_analysis["liquidity_analysis"]["liquidity_score"],
            "wash_trading_risk": full_analysis["trading_patterns"][
                "wash_trading_indicators"
            ]["risk_level"],
            "market_impact": {
                "slippage_estimate": full_analysis["liquidity_analysis"][
                    "slippage_estimates"
                ]["low_volume_slippage"],
                "high_volume_slippage": full_analysis["liquidity_analysis"][
                    "slippage_estimates"
                ]["high_volume_slippage"],
                "slippage_volatility": full_analysis["liquidity_analysis"][
                    "slippage_estimates"
                ]["slippage_volatility"],
                "price_impact_score": full_analysis["trading_patterns"][
                    "wash_trading_indicators"
                ]["confidence_score"],
            },
        },
        "health_indicators": {
            **full_analysis["ai_insights"]["market_health_indicators"],
            "key_findings": full_analysis["ai_insights"]["key_findings"],
            "risk_factors": full_analysis["ai_insights"]["risk_factors"],
            "trading_recommendations": full_analysis["ai_insights"][
                "trading_recommendations"
            ],
        },
        "key_metrics": {
            "price_volume_correlation": (
                full_analysis["anomaly_detection"]["price_volume_divergences"][0][
                    "divergence_score"
                ]
                if full_analysis["anomaly_detection"]["price_volume_divergences"]
                else 0
            ),
            "volume_concentration": full_analysis["liquidity_analysis"][
                "concentration_metrics"
            ]["top_hours_concentration"],
            "volume_distribution": full_analysis["liquidity_analysis"][
                "concentration_metrics"
            ]["volume_distribution"],
            "suspicious_patterns_count": len(
                full_analysis["anomaly_detection"]["unusual_patterns"]
            ),
            "peak_trading_hours": full_analysis["trading_patterns"]["seasonality"][
                "peak_hours"
            ],
            "low_volume_hours": full_analysis["trading_patterns"]["seasonality"][
                "low_volume_hours"
            ],
        },
    }

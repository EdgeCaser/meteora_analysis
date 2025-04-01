from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import time
import logging
import json
import requests
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load environment variables
load_dotenv()

class MeteoraPoolScraper:
    def __init__(self, cache_file=None):
        """
        Initialize the MeteoraPoolScraper.
        
        Args:
            cache_file (str, optional): Path to a CSV file containing cached pool data
        """
        self.base_url = "https://www.meteora.ag"
        self.POOLS_URL = self.base_url + "/clmm-api/pair/all"  # Use all pairs endpoint
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self.cache_file = cache_file
        self.cached_pools = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Load cached data if available
        if cache_file and os.path.exists(cache_file):
            try:
                self.cached_pools = pd.read_csv(cache_file)
                self.logger.info(f"Loaded {len(self.cached_pools)} pools from cache: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Error loading cache file: {e}")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Load cached data if available
        if cache_file and os.path.exists(cache_file):
            try:
                self.cached_pools = pd.read_csv(cache_file)
                self.logger.info(f"Loaded {len(self.cached_pools)} pools from cache: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Error loading cache file: {e}")

    def setup_logging(self):
        """Set up logging configuration."""
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def analyze_pool_risk(self, pool_address: str) -> Dict:
        """
        Analyze risk factors for a specific pool using on-chain data.
        
        Args:
            pool_address (str): The pool's contract address
            
        Returns:
            Dict containing risk metrics
        """
        risk_metrics = {
            'volume_volatility': None,
            'price_impact': None,
            'unique_traders': None,
            'trade_frequency': None
        }
        
        try:
            # For now, we only use public Meteora API data
            # Future versions may integrate additional data sources
            return risk_metrics
                
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {e}")
            return risk_metrics

    def get_recommended_pools(self, max_pairs=None, min_liquidity=10000, max_apy=1000, min_volume_24h=5000):
        """Get recommended pools from Meteora API with filtering."""
        try:
            # Add query parameters for filtering
            params = {
                'page': 0,
                'limit': 1000,
                'sort_key': 'volume',
                'order_by': 'desc'
            }
            
            response = requests.get(self.POOLS_URL, params=params)
            if response.status_code != 200:
                self.logger.error(f"Failed to get pools: {response.status_code}")
                return pd.DataFrame()

            self.logger.info(f"Pools response: {response.status_code}")
            pools = response.json()  # Response is a list of pools
            self.logger.info(f"Total pools found: {len(pools)}")
            
            # Suspicious token name patterns
            suspicious_patterns = [
                'fart', 'shit', 'cum', 'cock', 'penis', 'nigger', 'nazi',
                'hitler', 'porn', 'sex', 'fuck', 'ass', 'dick', 'pussy'
            ]
            
            processed_pools = []
            for i, pool in enumerate(pools):
                if i < 2:  # Log first few pools for debugging
                    self.logger.info(f"Processing pool {i}: {json.dumps(pool, indent=2)}")
                try:
                    # Skip blacklisted pools
                    if pool.get('is_blacklisted', False):
                        continue
                        
                    # Skip pools with suspicious token names
                    pool_name = pool.get('name', '').lower()
                    if any(pattern in pool_name for pattern in suspicious_patterns):
                        continue
                    
                    # Convert numeric fields with validation
                    try:
                        liquidity = float(pool.get('liquidity', 0))
                        volume_24h = float(pool.get('trade_volume_24h', 0))
                        fees_24h = float(pool.get('fees_24h', 0))
                        
                        # Calculate actual fee/TVL ratio
                        fee_tvl_ratio = (fees_24h / liquidity * 100) if liquidity > 0 else 0
                        
                        apy = float(pool.get('apy', 0))
                        current_price = float(pool.get('current_price', 0))
                        base_fee = float(pool.get('base_fee_percentage', 0))
                        max_fee = float(pool.get('max_fee_percentage', 0))
                        protocol_fee = float(pool.get('protocol_fee_percentage', 0))
                        bin_step = float(pool.get('bin_step', 0))
                        
                        # Calculate effective fee rate (fees/volume)
                        effective_fee_rate = (fees_24h / volume_24h * 100) if volume_24h > 0 else 0
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error converting numeric fields for pool {pool.get('name')}: {e}")
                        continue

                    # Validate metrics before processing
                    if not all([
                        liquidity >= min_liquidity,
                        volume_24h >= min_volume_24h,
                        apy <= max_apy,
                        current_price > 0,
                        volume_24h <= liquidity * 50,  # Volume shouldn't be more than 50x liquidity
                        volume_24h > 0,  # Must have some volume
                        fees_24h > 0,  # Must have some fees
                        fee_tvl_ratio > 0,  # Must have positive fee/TVL ratio
                        effective_fee_rate <= max_fee,  # Fee rate shouldn't exceed max fee
                        fee_tvl_ratio <= 10  # Fee/TVL ratio shouldn't be unrealistically high
                    ]):
                        if i < 5:  # Log first few pools that fail validation
                            self.logger.info(f"Pool {pool.get('name')} failed validation:")
                            self.logger.info(f"- Liquidity: {liquidity} >= {min_liquidity}: {liquidity >= min_liquidity}")
                            self.logger.info(f"- Volume 24h: {volume_24h} >= {min_volume_24h}: {volume_24h >= min_volume_24h}")
                            self.logger.info(f"- APY: {apy} <= {max_apy}: {apy <= max_apy}")
                            self.logger.info(f"- Price > 0: {current_price > 0}")
                            self.logger.info(f"- Volume/Liquidity <= 50: {volume_24h <= liquidity * 50}")
                            self.logger.info(f"- Effective fee rate: {effective_fee_rate:.2f}% <= {max_fee}%: {effective_fee_rate <= max_fee}")
                            self.logger.info(f"- Fee/TVL ratio: {fee_tvl_ratio:.2f}% <= 10%: {fee_tvl_ratio <= 10}")
                        continue
                        
                    self.logger.info(f"Pool {pool.get('name')} metrics:")
                    self.logger.info(f"- Volume 24h: {volume_24h:,.2f}")
                    self.logger.info(f"- Fee/TVL 24h: {fee_tvl_ratio:.2f}%")
                    self.logger.info(f"- Liquidity: {liquidity:,.2f}")
                    self.logger.info(f"- Effective fee rate: {effective_fee_rate:.2f}%")
                    self.logger.info(f"- Base/Max fees: {base_fee}%/{max_fee}%")
                    
                    processed_pools.append({
                        'token_pair': pool.get('name'),
                        'liquidity': liquidity,
                        'trade_volume_24h': volume_24h,
                        'fees_24h': fees_24h,
                        'fee_tvl_ratio_24h': fee_tvl_ratio,
                        'effective_fee_rate': effective_fee_rate,
                        'base_fee_percentage': base_fee,
                        'max_fee_percentage': max_fee,
                        'apy': apy,
                        'current_price': current_price,
                        'protocol_fee_percentage': protocol_fee,
                        'bin_step': bin_step,
                        'address': pool.get('address'),
                        'last_updated': datetime.now().isoformat()
                    })
                    self.logger.info(f"Added pool {pool.get('name')} to processed pools")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing pool {pool.get('name')}: {e}")
                    continue
                    
                if max_pairs and len(processed_pools) >= max_pairs:
                    break
            
            df = pd.DataFrame(processed_pools)
            if df.empty:
                self.logger.warning("No pools passed validation")
                return df
                
            self.logger.info("Created DataFrame with {} pools".format(len(df)))
            self.logger.info("Sample of processed data:")
            self.logger.info(df.head())
            
            # Sort by fee/TVL ratio and show top pools
            df_sorted = df.sort_values('fee_tvl_ratio_24h', ascending=False)
            self.logger.info("Top pools after filtering:")
            self.logger.info(df_sorted[['token_pair', 'liquidity', 'trade_volume_24h', 'fee_tvl_ratio_24h', 'effective_fee_rate']].head())
            
            return df_sorted
            
        except Exception as e:
            self.logger.error(f"Error getting recommended pools: {e}")
            return pd.DataFrame()

    def calculate_dynamic_ratios(self, pool_data: dict) -> dict:
        """
        Calculate dynamic ratios based on historical transaction data.
        
        Args:
            pool_data: Pool data containing volume and liquidity metrics
            
        Returns:
            Dictionary containing calculated ratios and metrics
        """
        # Get time series data
        volume_series = {
            '30m': pool_data.get('volume', {}).get('min_30', 0),
            '1h': pool_data.get('volume', {}).get('hour_1', 0),
            '2h': pool_data.get('volume', {}).get('hour_2', 0),
            '4h': pool_data.get('volume', {}).get('hour_4', 0),
            '12h': pool_data.get('volume', {}).get('hour_12', 0),
            '24h': pool_data.get('volume', {}).get('hour_24', 0)
        }
        
        fee_tvl_series = {
            '30m': pool_data.get('fee_tvl_ratio', {}).get('min_30', 0),
            '1h': pool_data.get('fee_tvl_ratio', {}).get('hour_1', 0),
            '2h': pool_data.get('fee_tvl_ratio', {}).get('hour_2', 0),
            '4h': pool_data.get('fee_tvl_ratio', {}).get('hour_4', 0),
            '12h': pool_data.get('fee_tvl_ratio', {}).get('hour_12', 0),
            '24h': pool_data.get('fee_tvl_ratio', {}).get('hour_24', 0)
        }
        
        # Calculate weighted averages giving more weight to recent data
        weights = {
            '30m': 0.3,
            '1h': 0.25,
            '2h': 0.2,
            '4h': 0.15,
            '12h': 0.07,
            '24h': 0.03
        }
        
        # Calculate weighted volume
        total_volume = 0
        total_weight = 0
        for period, volume in volume_series.items():
            if volume > 0:  # Only consider periods with activity
                total_volume += volume * weights[period]
                total_weight += weights[period]
        
        weighted_volume = total_volume / total_weight if total_weight > 0 else 0
        
        # Calculate weighted fee/TVL ratio
        total_fee_tvl = 0
        total_weight = 0
        for period, ratio in fee_tvl_series.items():
            if ratio > 0:  # Only consider periods with activity
                total_fee_tvl += ratio * weights[period]
                total_weight += weights[period]
        
        weighted_fee_tvl = total_fee_tvl / total_weight if total_weight > 0 else 0
        
        # Calculate volatility (standard deviation of changes)
        volume_values = [v for v in volume_series.values() if v > 0]
        fee_tvl_values = [r for r in fee_tvl_series.values() if r > 0]
        
        volume_volatility = np.std(volume_values) / np.mean(volume_values) if volume_values else 0
        fee_tvl_volatility = np.std(fee_tvl_values) / np.mean(fee_tvl_values) if fee_tvl_values else 1.0
        
        # Calculate activity score based on number of active periods
        active_periods = sum(1 for v in volume_series.values() if v > 0)
        activity_score = active_periods / len(volume_series)
        
        return {
            'weighted_volume': weighted_volume,
            'weighted_fee_tvl': weighted_fee_tvl,
            'volume_volatility': volume_volatility,
            'fee_tvl_volatility': fee_tvl_volatility,
            'activity_score': activity_score,
            'time_series': {
                'volume': volume_series,
                'fee_tvl': fee_tvl_series
            }
        }

    def forecast_fee_revenue(self, initial_liquidity: float, market_cap: float, 
                           token_supply: float, bin_step: int = 150) -> Dict:
        """
        Forecast expected fee revenue based on pool parameters and historical data.
        
        Args:
            initial_liquidity: Initial pool liquidity in USD
            market_cap: Token market cap in USD
            token_supply: Total token supply
            bin_step: Bin step for the pool (default 150 for new volatile tokens)
        """
        # Get historical data for similar pools
        pools_data = self.get_recommended_pools(
            min_liquidity=initial_liquidity * 0.5,
            max_pairs=1000
        )
        
        if pools_data.empty:
            return {"error": "No historical data available"}
            
        # Calculate metrics for similar pools
        similar_pools_metrics = []
        for pool in pools_data.itertuples():
            try:
                # Calculate effective fee rate based on actual fees and volume
                if pool.trade_volume_24h > 0 and pool.fees_24h > 0:
                    effective_fee_rate = (pool.fees_24h / pool.trade_volume_24h) * 100
                else:
                    # Default to base fee if no volume
                    effective_fee_rate = float(pool.base_fee_percentage) if pool.base_fee_percentage > 0 else 0.1
                
                # Validate the fee rate is reasonable
                if effective_fee_rate > 100:  # Cap at 100% as sanity check
                    effective_fee_rate = float(pool.base_fee_percentage) if pool.base_fee_percentage > 0 else 0.1
                
                # Calculate volume to liquidity ratio for the pool
                volume_to_liquidity = pool.trade_volume_24h / pool.liquidity if pool.liquidity > 0 else 0
                
                # Calculate volatility using hourly volume data
                hourly_volumes = [
                    pool.trade_volume_24h / 24,  # hour_1
                    pool.trade_volume_24h / 12,  # hour_2
                    pool.trade_volume_24h / 6,   # hour_4
                    pool.trade_volume_24h / 2,   # hour_12
                    pool.trade_volume_24h        # hour_24
                ]
                volume_volatility = np.std(hourly_volumes) / np.mean(hourly_volumes) if np.mean(hourly_volumes) > 0 else 1.0
                
                metrics = {
                    'volume_to_liquidity': volume_to_liquidity,
                    'fee_tvl_ratio': pool.fee_tvl_ratio_24h,
                    'effective_fee_rate': effective_fee_rate / 100,  # Convert to decimal
                    'volume_volatility': volume_volatility
                }
                similar_pools_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error calculating metrics for pool {pool.token_pair}: {e}")
                continue
        
        # Extract metrics for calculations
        volume_ratios = [m['volume_to_liquidity'] for m in similar_pools_metrics]
        fee_tvl_ratios = [m['fee_tvl_ratio'] for m in similar_pools_metrics]
        effective_fee_rates = [m['effective_fee_rate'] for m in similar_pools_metrics]
        volume_volatilities = [m['volume_volatility'] for m in similar_pools_metrics]
        
        # Filter out any extreme outliers in fee rates (more than 3 std devs from mean)
        fee_mean = np.mean(effective_fee_rates)
        fee_std = np.std(effective_fee_rates)
        effective_fee_rates = [r for r in effective_fee_rates if abs(r - fee_mean) <= 3 * fee_std]
        
        # Filter out extreme volume ratios (more than 5 std devs from mean to keep more high-volume pools)
        volume_mean = np.mean([r for r in volume_ratios if r > 0])  # Only consider pools with volume
        volume_std = np.std([r for r in volume_ratios if r > 0])
        volume_ratios = [r for r in volume_ratios if r > 0 and abs(r - volume_mean) <= 5 * volume_std]
        
        scenarios = {
            'conservative': {
                'percentile': 25,
                'volume_ratio': max(4.0, np.percentile(volume_ratios, 25)),  # At least 4x liquidity daily volume
                'fee_tvl_ratio': np.percentile(fee_tvl_ratios, 25),
                'effective_fee_rate': min(0.002, np.percentile(effective_fee_rates, 25))  # Cap at 0.2% (0.002) for stablecoin pairs
            },
            'moderate': {
                'percentile': 50,
                'volume_ratio': max(6.0, np.percentile(volume_ratios, 50)),  # At least 6x liquidity daily volume
                'fee_tvl_ratio': np.percentile(fee_tvl_ratios, 50),
                'effective_fee_rate': min(0.01, np.percentile(effective_fee_rates, 50))  # Cap at 1% (0.01) for major pairs
            },
            'optimistic': {
                'percentile': 75,
                'volume_ratio': max(12.0, np.percentile(volume_ratios, 75)),  # At least 12x liquidity daily volume
                'fee_tvl_ratio': np.percentile(fee_tvl_ratios, 75),
                'effective_fee_rate': min(0.03, np.percentile(effective_fee_rates, 75))  # Cap at 3% (0.03) for volatile pairs
            }
        }
        
        # Calculate average metrics for similar pools
        similar_pools_stats = {
            'count': len(similar_pools_metrics),
            'avg_volume_to_liquidity': np.mean(volume_ratios),
            'avg_fee_tvl_ratio': np.mean(fee_tvl_ratios),
            'avg_effective_fee_rate': np.mean(effective_fee_rates),
            'volume_volatility': np.mean(volume_volatilities),
            'fee_tvl_volatility': np.std(fee_tvl_ratios) / np.mean(fee_tvl_ratios) if np.mean(fee_tvl_ratios) > 0 else 1.0
        }
        
        # Calculate forecasts for each scenario
        token_price = market_cap / token_supply
        metrics = {
            'input_parameters': {
                'initial_liquidity': initial_liquidity,
                'market_cap': market_cap,
                'token_supply': token_supply,
                'token_price': token_price,
                'bin_step': bin_step
            },
            'similar_pools_stats': similar_pools_stats,
            'scenarios': {}
        }
        
        for scenario, data in scenarios.items():
            daily_volume = initial_liquidity * data['volume_ratio']
            effective_fee_rate = data['effective_fee_rate']
            
            # Calculate daily and annual fees using effective fee rate
            daily_fee = (daily_volume * effective_fee_rate)
            annual_fee = daily_fee * 365
            
            # Calculate APR
            apr = (annual_fee / initial_liquidity) * 100
            
            metrics['scenarios'][scenario] = {
                'daily_volume': daily_volume,
                'volume_to_liquidity_ratio': float(data['volume_ratio']),
                'fee_tvl_ratio': float(data['fee_tvl_ratio']),
                'effective_fee_rate': float(effective_fee_rate),
                'daily_fee_revenue': daily_fee,
                'annual_fee_revenue': annual_fee,
                'apr': apr,
                'confidence_score': data['percentile'] / 100
            }
        
        return metrics

    def _calculate_risk_adjustments(self, market_cap: float, token_supply: float) -> Dict:
        """Calculate risk adjustment factors based on on-chain data."""
        adjustments = {}
        
        try:
            # Market cap based adjustment
            if market_cap < 100000:  # Small cap tokens typically have higher volatility
                adjustments['market_cap_factor'] = 0.2  # 20% premium for higher trading activity
            elif market_cap < 1000000:
                adjustments['market_cap_factor'] = 0.1
            else:
                adjustments['market_cap_factor'] = 0
            
            # Supply concentration adjustment
            token_price = market_cap / token_supply
            if token_price < 0.0001:
                adjustments['price_factor'] = 0.15  # 15% premium for micro-price tokens
            elif token_price < 0.01:
                adjustments['price_factor'] = 0.05
            else:
                adjustments['price_factor'] = 0
                
        except Exception as e:
            self.logger.error(f"Error calculating risk adjustments: {e}")
            return {}
            
        return adjustments

class MeteoraReportGenerator:
    def __init__(self, pools_data: pd.DataFrame, forecast_data: Dict):
        """
        Initialize report generator with pools and forecast data.
        
        Args:
            pools_data: DataFrame containing pool metrics
            forecast_data: Dictionary containing fee revenue forecast
        """
        self.pools_data = pools_data
        self.forecast_data = forecast_data
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def generate_fee_distribution_plot(self):
        """Generate distribution plots for fees and related metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Meteora Pool Metrics Distribution", fontsize=16)
        
        # Fee/TVL ratio distribution
        sns.histplot(data=self.pools_data, x='fee_tvl_ratio_24h', ax=axes[0,0], bins=30)
        axes[0,0].set_title('24h Fee/TVL Ratio Distribution')
        axes[0,0].set_xlabel('Fee/TVL Ratio')
        
        # Volume distribution (log scale)
        sns.histplot(data=self.pools_data, x='trade_volume_24h', ax=axes[0,1], bins=30)
        axes[0,1].set_title('24h Trading Volume Distribution')
        axes[0,1].set_xlabel('Volume (USD)')
        axes[0,1].set_xscale('log')
        
        # APY distribution
        sns.histplot(data=self.pools_data, x='apy', ax=axes[1,0], bins=30)
        axes[1,0].set_title('APY Distribution')
        axes[1,0].set_xlabel('APY (%)')
        
        # Liquidity distribution (log scale)
        sns.histplot(data=self.pools_data, x='liquidity', ax=axes[1,1], bins=30)
        axes[1,1].set_title('Liquidity Distribution')
        axes[1,1].set_xlabel('Liquidity (USD)')
        axes[1,1].set_xscale('log')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'fee_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def generate_forecast_report(self) -> str:
        """Generate a detailed forecast report with metrics and insights."""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = self.output_dir / f'forecast_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Meteora Pool Fee Revenue Forecast Report\n")
            f.write(f"Generated at: {report_time}\n")
            f.write("=" * 80 + "\n\n")
            
            # Input parameters
            f.write("Input Parameters:\n")
            f.write("-" * 50 + "\n")
            params = self.forecast_data['input_parameters']
            f.write(f"Initial Liquidity: ${params['initial_liquidity']:,.2f}\n")
            f.write(f"Market Cap: ${params['market_cap']:,.2f}\n")
            f.write(f"Token Price: ${params['token_price']:.4f}\n")
            f.write(f"Bin Step: {params['bin_step']}\n\n")
            
            # Similar pools statistics
            stats = self.forecast_data['similar_pools_stats']
            f.write("Market Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of Similar Pools: {stats['count']}\n")
            f.write(f"Average Volume/Liquidity Ratio: {stats['avg_volume_to_liquidity']:.4f}\n")
            f.write(f"Average Fee/TVL Ratio: {stats['avg_fee_tvl_ratio']:.4f}\n")
            f.write(f"Average Effective Fee Rate: {stats['avg_effective_fee_rate']:.4f}\n")
            f.write(f"Volume Volatility: {stats['volume_volatility']:.4f}\n")
            f.write(f"Fee/TVL Volatility: {stats['fee_tvl_volatility']:.4f}\n\n")
            
            # Scenario forecasts
            f.write("Revenue Forecasts by Scenario:\n")
            f.write("-" * 50 + "\n")
            
            for scenario, data in self.forecast_data['scenarios'].items():
                f.write(f"\n{scenario.title()} Scenario (Confidence: {data['confidence_score']:.2%})\n")
                f.write(f"Daily Volume: ${data['daily_volume']:,.2f}\n")
                f.write(f"Volume/Liquidity Ratio: {data['volume_to_liquidity_ratio']:.4f}\n")
                f.write(f"Fee/TVL Ratio: {data['fee_tvl_ratio']:.4f}\n")
                f.write(f"Effective Fee Rate: {data['effective_fee_rate']:.4f}\n")
                f.write(f"Daily Fee Revenue: ${data['daily_fee_revenue']:,.2f}\n")
                f.write(f"Annual Fee Revenue: ${data['annual_fee_revenue']:,.2f}\n")
                f.write(f"APR: {data['apr']:.2f}%\n")
            
            # Pool statistics
            f.write("\nCurrent Market Pool Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Active Pools: {len(self.pools_data)}\n")
            f.write(f"Median Pool Liquidity: ${self.pools_data['liquidity'].median():,.2f}\n")
            f.write(f"Median 24h Volume: ${self.pools_data['trade_volume_24h'].median():,.2f}\n")
            f.write(f"Median Fee/TVL Ratio: {self.pools_data['fee_tvl_ratio_24h'].median():.4f}\n")
            f.write(f"Median APY: {self.pools_data['apy'].median():.2f}%\n")
        
        return report_path

def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description='Meteora Pool Scraper')
    parser.add_argument('--min-liquidity', type=float, default=10000,
                      help='Minimum liquidity in USD')
    parser.add_argument('--min-volume', type=float, default=5000,
                      help='Minimum 24h volume in USD')
    parser.add_argument('--max-apy', type=float, default=1000,
                      help='Maximum APY to consider')
    parser.add_argument('--output-dir', type=str, default='reports',
                      help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize scraper
    scraper = MeteoraPoolScraper()
    
    # Get recommended pools
    pools = scraper.get_recommended_pools(
        min_liquidity=args.min_liquidity,
        min_volume_24h=args.min_volume,
        max_apy=args.max_apy
    )
    
    if pools.empty:
        print("No pools found matching criteria")
        return
        
    # Display top pools
    print("\nTop Pools by Fee/TVL Ratio:")
    print(pools[['token_pair', 'liquidity', 'trade_volume_24h', 'fee_tvl_ratio_24h', 'effective_fee_rate']].head(10))
    
    # Generate forecast report
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(args.output_dir, f"forecast_report_{report_time}.txt")
    plot_file = os.path.join(args.output_dir, "fee_distribution.png")
    
    # Example pool for forecasting
    forecast = scraper.forecast_fee_revenue(
        initial_liquidity=20000,
        market_cap=1000000,
        token_supply=1000000,
        bin_step=150
    )
    
    # Generate report
    report = MeteoraReportGenerator(pools, forecast)
    plot_path = report.generate_fee_distribution_plot()
    report_path = report.generate_forecast_report()
    print(f"\nReport generated: {report_path}")
    print(f"Plots generated: {plot_path}")
    
    # Print forecast summary
    print("\nFee Revenue Forecast:")
    print(json.dumps(forecast, indent=2))

if __name__ == "__main__":
    main()

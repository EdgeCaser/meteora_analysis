# Meteora Liquidity Pool Fee Revenue Forecast

## Methodology and Results

### Executive Summary

This analysis presents a data-driven approach to forecasting fee revenue for Meteora liquidity pools. The methodology combines empirical analysis of active pools with statistical modeling to predict revenue across different market conditions. Key findings from March 2025 data show:

1. Volume-weighted average fee rate of 0.098% across all pools
2. High-volume pools (>$100k daily) average 0.096% effective fee rate
3. Fee/TVL ratios typically range from 0.09% to 0.12%
4. Volume/liquidity ratios average 1.83x for high-volume pools

### Data Collection Methodology

1. **Data Source**
   - Direct API integration with Meteora's backend endpoints
   - Primary endpoint: `https://www.meteora.ag/clmm-api/pair/all`
   - Real-time pool metrics including volume, fees, and liquidity

2. **Pool Quality Filters**
   - Minimum liquidity: $10,000
   - Minimum 24h volume: $5,000
   - Maximum APY: 1,000%
   - Non-negative fee/TVL ratio
   - Valid current price
   - Volume/liquidity ratio ≤ 10x (more conservative limit)
   - Effective fee rate caps:
     - 0.3% for high-volume pools (>$100k daily)
     - 0.5% for lower-volume pools
   - Blacklist and suspicious token name filtering

### Fee Rate Analysis

Based on March 2025 data from 190 qualified pools:

1. **Volume-Weighted Metrics**
   - Fee Rate: 0.098%
   - Fee/TVL Ratio: 0.107%
   - Strong correlation between volume and lower effective rates

2. **High-Volume Pool Metrics**
   - Average Fee Rate: 0.096%
   - Average Fee/TVL: 0.121%
   - Average Volume/Liquidity: 1.83x

3. **Market Medians**
   - Pool Liquidity: $77,139
   - Daily Volume: $25,950
   - Fee/TVL Ratio: 0.090%
   - APY: 39.03%

### Revenue Forecast Model

The model uses three scenarios based on empirical data from March 2025:

1. **Conservative Scenario** (75% confidence)
   - Volume/Liquidity ratio: 0.92x
   - Effective Fee Rate: 0.077%
   - Daily Volume Cap: $18,313 (for $20k liquidity)
   - Expected APR: 25.73%

2. **Moderate Scenario** (50% confidence)
   - Volume/Liquidity ratio: 1.83x
   - Effective Fee Rate: 0.096%
   - Daily Volume Cap: $36,627 (for $20k liquidity)
   - Expected APR: 64.32%

3. **Optimistic Scenario** (25% confidence)
   - Volume/Liquidity ratio: 2.75x
   - Effective Fee Rate: 0.116%
   - Daily Volume Cap: $54,940 (for $20k liquidity)
   - Expected APR: 115.78%

### Key Model Improvements

1. **More Conservative Volume Assumptions**
   - Reduced max volume/liquidity ratio from 50x to 10x
   - Based volume projections on actual high-volume pool averages

2. **Realistic Fee Rate Caps**
   - Implemented volume-based fee rate limits
   - 0.3% cap for high-volume pools
   - 0.5% cap for lower-volume pools

3. **Better Risk Assessment**
   - Fee/TVL ratio cap lowered to 3% (based on observed max of ~2.11%)
   - Higher confidence in conservative scenarios (75%)
   - Lower confidence in optimistic scenarios (25%)

### Validation Against Real Data

Current top pools on Meteora (March 2025):

1. **GOLD-SOL**
   - TVL: $688,935
   - Daily Volume: $159,810
   - Fee/TVL: 2.11%
   - Volume/Liquidity: 0.23x

2. **Other Notable Pools**
   - Most pools show 0.2-5% nominal fee rates
   - Effective fee rates much lower than nominal rates
   - Many pools showing 0% fee/TVL despite activity

The model's predictions now align well with observed data:
- Conservative estimates match typical stablecoin pair performance
- Moderate scenario aligns with average high-volume pool metrics
- Optimistic scenario reflects realistic upper bounds based on market data

### Risk Factors

1. **Volume Dependency**
   - Higher volume typically leads to lower effective fee rates
   - Volume/liquidity ratios vary significantly by pair type

2. **Market Conditions**
   - Overall market sentiment affects trading activity
   - Competition from other pools can impact volume

3. **Token Characteristics**
   - Stablecoin pairs typically show lower fee rates but higher volume
   - New or volatile pairs may see higher fee rates but less consistent volume

### Conclusion

The improved model provides realistic revenue forecasts based on current market conditions, with built-in safeguards against over-optimistic projections. The volume-weighted approach and stricter validation rules ensure predictions align with actual pool performance on Meteora.

Note: All forecasts are based on March 2025 market data and should be periodically updated as market conditions evolve.

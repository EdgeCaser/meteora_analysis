# Meteora Pool Analysis

This repository contains analysis and forecasting tools for Meteora liquidity pools on Solana. The main focus is on fee revenue forecasting based on empirical data from comparable pools.

## Overview

The analysis includes:
- Fee revenue forecasting model based on real pool data
- Volume-to-liquidity ratio analysis
- Effective fee rate calculations
- Fee/TVL ratio benchmarking

## Key Findings

For pools in the $15K-25K liquidity range:
- Volume ratios typically range from 5-7x liquidity
- Fee/TVL ratios range from 0.3% to 0.7% for active pools
- Effective fee rates vary by pair type:
  - Stablecoin pairs: ~0.2%
  - Major pairs: ~1.0%
  - Volatile pairs: up to 3.0%

## Repository Structure

- `src/`: Source code for the pool scraper and analysis tools
- `reports/`: Generated reports and methodology documentation

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- requests

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pool scraper:
```bash
python src/meteora_pool_scraper.py
```

3. View the generated reports in the `reports/` directory.

## License

MIT License

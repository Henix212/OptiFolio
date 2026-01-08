# OptiFolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Henix212%2FCerberus-red)](https://github.com/Henix212/OptiFolio.git)


**Version 1.0**

OptiFolio is a portfolio optimization system based on Reinforcement Learning using the PPO (Proximal Policy Optimization) algorithm. The project enables training an intelligent agent to manage a portfolio of financial assets by optimizing returns while controlling target volatility.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Training Environment](#-training-environment)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

## ✨ Features

- **Automatic download** of market data from Yahoo Finance
- **Data cleaning and preparation** of financial data (prices, volumes, returns)
- **Advanced feature extraction** :
  - EWMA volatility (short and long term)
  - Average correlations between assets
  - Normalized returns
- **Custom Reinforcement Learning environment** (Gymnasium)
- **PPO training** with target volatility control
- **Data and feature visualization**
- **Performance tracking** via TensorBoard

## 🏗️ Architecture

The project follows a modular architecture organized into several components:

1. **Data management** (`utils/dataHandler.py`) : Download and cleaning of raw data
2. **Feature extraction** (`utils/featuresHandler.py`) : Calculation of advanced financial metrics
3. **Dataset creation** (`utils/datasetHandler.py`) : Aggregation of features into a unified dataset
4. **RL environment** (`env/optiFolioEnv.py`) : Custom Gymnasium environment for training
5. **Training** (`main.ipynb`) : Main PPO agent training script

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Installing Dependencies

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd OptiFolio

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- `yfinance` : Financial data download
- `pandas` : Data manipulation
- `numpy` : Numerical computations
- `gymnasium` : RL environment framework
- `stable_baselines3` : PPO implementation
- `tensorboard` : Training metrics visualization
- `matplotlib` : Data visualization
- `jupyterlab` : Interactive development environment

## 📖 Usage

### 1. Data Preparation

#### Step 1: Download and clean raw data

```python
from utils.dataHandler import main, tickers_list

# Downloads market data and generates cleaned files
main(tickers_list)
```

The default stock indices downloaded are:
- `^NDX` : NASDAQ-100
- `^FCHI` : CAC 40
- `^GDAXI` : DAX
- `^N225` : Nikkei 225
- `^HSI` : Hang Seng
- `^SSMI` : Swiss SMI

#### Step 2: Feature extraction

```python
from utils.featuresHandler import *

# Calculates financial features (volatility, correlations, etc.)
# The script runs automatically if called directly
```

#### Step 3: Create final dataset

```python
from utils.datasetHandler import create_dataset

# Combines all features into a synchronized dataset
create_dataset("data/features")
```

### 2. Agent Training

Open `main.ipynb` in JupyterLab and execute the cells to:

1. Load the prepared dataset
2. Initialize the `optiFolioEnv` environment
3. Train the PPO agent
4. Visualize results via TensorBoard

### 3. Visualization

The notebooks in the `notebook/` folder allow visualization of:
- Raw and cleaned data (`dataVisualization.ipynb`)
- Extracted features (`featureVisualization.ipynb`)

## 📁 Project Structure

```
OptiFolio/
├── data/
│   ├── raw/              # Downloaded raw data
│   │   └── yahoo/
│   ├── cleaned/          # Cleaned data (prices, volumes, returns)
│   ├── features/         # Calculated features (volatility, correlations)
│   └── dataset/          # Final combined dataset
├── env/
│   └── optiFolioEnv.py   # Custom Gymnasium environment
├── utils/
│   ├── dataHandler.py    # Raw data management
│   ├── featuresHandler.py # Feature extraction
│   └── datasetHandler.py  # Final dataset creation
├── notebook/
│   ├── dataVisualization.ipynb
│   └── featureVisualization.ipynb
├── ppo_tensorboard/      # TensorBoard training logs
├── main.ipynb            # Main training script
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## 🔄 Data Pipeline

The data processing pipeline follows these steps:

1. **Download** : Retrieval of historical data from Yahoo Finance
2. **Cleaning** : Extraction of closing prices, volumes and calculation of returns
3. **Feature Engineering** :
   - EWMA volatility (short λ = 0.94, long λ = 0.97)
   - Volatility ratio (short/long term)
   - EWMA average correlations
   - Volatility-normalized returns
4. **Aggregation** : Synchronization of all features on common dates
5. **Final dataset** : Creation of a unified CSV file for training

## 🎮 Training Environment

The `optiFolioEnv` environment is configured with the following parameters:

- **Initial capital** : 10,000 (default)
- **Lookback window** : 20 days
- **Maximum duration** : 252 days (1 trading year)
- **Target volatility** : 2% (default)

### Reward Function

The reward combines several components:

- **Logarithmic return** : Reward based on portfolio return
- **Alpha bonus** : Bonus for profitable trades
- **Volatility penalty** : Penalty if volatility exceeds target
- **Turnover penalty** : Discourages excessive reallocations

### Observation Space

- Sliding window of features over the last N days
- Target volatility as an additional feature

### Action Space

- Portfolio weights for each asset (normalized via softmax)

## 🛠️ Technologies Used

- **Python** : Main programming language
- **Yahoo Finance API** : Market data source
- **Gymnasium** : Standard for RL environments
- **Stable-Baselines3** : RL algorithms library
- **TensorBoard** : Metrics visualization tool
- **Pandas/NumPy** : Data manipulation and computations

## 📊 Metrics and Monitoring

Training metrics are recorded in `ppo_tensorboard/` and can be visualized with:

```bash
tensorboard --logdir=ppo_tensorboard
```

## 🔧 Advanced Configuration

### Customizing Indices

Modify the `tickers_list` in `utils/dataHandler.py` to add or remove indices.

### Environment Parameters

Environment parameters can be adjusted during initialization:

```python
env = optiFolioEnv(
    dataset_path="data/dataset/dataset.csv",
    initial_amount=10_000,
    lookback=20,
    max_days=252,
    target_vol=0.02
)
```

### EWMA Parameters

The decay factors for EWMA calculations can be modified in `utils/featuresHandler.py`:
- `lambda_short = 0.94` (short term)
- `lambda_long = 0.97` (long term)

## ⚠️ Warnings

- This project is for educational and research purposes
- Past performance does not guarantee future results
- Always perform thorough testing before any production use
- Market data may contain errors or gaps

## 📝 Version Notes

### Version 1.0

- Initial implementation of data pipeline
- Custom RL environment with PPO
- Advanced financial feature extraction
- Support for multiple stock indices
- Visualization via TensorBoard

## 👤 Author

Developed as part of a portfolio optimization project.

---

**License** : This project is provided as-is, without any warranty.

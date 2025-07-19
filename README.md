# CHARGED: A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15638530.svg)](https://doi.org/10.5281/zenodo.15638530)


A comprehensive framework for electric vehicle charging demand prediction and analysis across multiple global cities. This project provides harmonized datasets, advanced prediction models, and federated learning capabilities for city-scale EV charging behavior analysis and infrastructure planning.

## 📚 Citation

If this project is helpful to your research, please cite our papers:

**Guo, Z., You, L., Zhu, R. et al. A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis. Sci Data 12, 1254 (2025). https://doi.org/10.1038/s41597-025-05584-7**
**Li, H., Qu, H., Tan, X. et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Sci Data 12, 523 (2025). https://doi.org/10.1038/s41597-025-04874-4**
**You, L., Guo, Z., Yuen, C. et al. A framework reforming personalized Internet of Things by federated meta-learning. Nat Commun 16, 3739 (2025). https://doi.org/10.1038/s41467-025-59217-z**

```bibtex
@article{guo2025a,
  author={Guo, Zihan and You, Linlin and Zhu, Rui and Zhang, Yan and Yuen, Chau},
  title={A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis},
  journal={Scientific Data},
  volume={12},
  pages={1254},
  year={2025},
  doi={10.1038/s41597-025-05584-7},
}
@article{li2025urbanev,
  author={Li, Han and Qu, Haohao and Tan, Xiaojun and You, Linlin and Zhu, Rui and Fan, Wenqi},
  title={UrbanEV: An open benchmark dataset for urban electric vehicle charging demand prediction},
  journal={Scientific Data},
  volume={12},
  pages={523},
  year={2025},
  doi={10.1038/s41597-025-04874-4},
}
@article{you2025framework,
  author={You, Linlin and Guo, Zihan and Yuen, Chau and Chen, Calvin Yu-Chian and Zhang, Yan and Poor, H Vincent},
  title={A framework reforming personalized Internet of Things by federated meta-learning},
  journal={Nature communications},
  volume={16},
  pages={3739},
  year={2025},
  doi={10.1038/s41467-025-59217-z},
}
```

## 🌟 Key Features

- **Multi-City Dataset**: Harmonized charging data from 6 major cities worldwide
- **Advanced Prediction Models**: Support for state-of-the-art time series forecasting models
- **Federated Learning**: Privacy-preserving distributed training across cities
- **Comprehensive Preprocessing**: Data cleaning, anomaly detection, and feature engineering
- **Knowledge Transfer**: Cross-city model adaptation and transfer learning

## 📊 Supported Cities

| City | Country | Code |
|------|---------|------|
| Amsterdam | Netherlands | AMS |
| Johannesburg | South Africa | JHB |
| Los Angeles | United States | LOA |
| Melbourne | Australia | MEL |
| São Paulo | Brazil | SPO |
| Shenzhen | China | SZH |

## 🏗️ Project Structure

```
CHARGED/
├── api/                          # Core API modules
│   ├── dataset/                  # Data loading and processing
│   │   ├── common.py            # Single-city dataset handling
│   │   └── distributed.py       # Federated dataset management
│   ├── federated/               # Federated learning components
│   │   ├── client.py           # Client-side training logic
│   │   └── server.py           # Server-side aggregation
│   ├── model/                   # Prediction models
│   │   ├── config.py           # Model configuration
│   │   ├── layers.py           # Neural network layers
│   │   └── modules.py          # Model architectures
│   ├── parsing/                 # Command-line argument parsing
│   │   ├── common.py           # Standard prediction arguments
│   │   └── federated.py        # Federated learning arguments
│   ├── trainer/                 # Training utilities
│   │   ├── common.py           # Standard training loop
│   │   └── federated.py        # Federated training logic
│   └── utils.py                 # Utility functions
├── data/                        # Dataset storage
│   ├── [CITY]/                 # City-specific data directories
│   └── [CITY]_remove_zero/     # Preprocessed data (zero clusters removed)
├── example/                     # Usage examples and scripts
│   ├── knowledge_transfer.py   # Federated learning example
│   ├── univariate_prediction.py # Single-city prediction
│   ├── multivariate_prediction.py # Multi-feature prediction
│   └── test_*.py               # Model evaluation scripts
├── script/                      # Data processing and utility scripts
│   ├── aggregate/              # Data aggregation tools
│   ├── auxiliary/              # Auxiliary data collection
│   ├── optimize/               # Data optimization and cleaning
│   └── visualization/          # Data visualization tools
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch pandas numpy scikit-learn matplotlib seaborn
pip install geopandas osmnx tqdm pyproj shapely
pip install chronos momentfm gluonts uni2ts
```

### Basic Usage

#### 1. Single-City Prediction

```bash
# Run univariate prediction for a specific city
python example/univariate_prediction.py \
    --city SZH \
    --model transformer \
    --feature volume \
    --seq_l 24 \
    --pre_len 12 \
    --epoch 100 \
    --is_train True
```

#### 2. Federated Learning

```bash
# Run federated learning across multiple cities
python example/knowledge_transfer.py \
    --city AMS JHB LOA MEL SPO SZH \
    --model transformer \
    --feature volume \
    --pred_type site
    --global_epoch 50 \
    --local_epoch 10
```

#### 3. Model Evaluation

```bash
# Evaluate pretrained models (Chronos, Moirai, MOMENT)
python example/test_chronos_all.py
python example/test_moirai_all.py
python example/test_moment_all.py
```

## 📈 Supported Models

### Traditional Models
- **Linear Regression (LR)**: Baseline linear prediction
- **Auto-Regressive (AR)**: Time series autoregression
- **ARIMA**: Integrated autoregressive moving average

### Deep Learning Models
- **Transformer**: Attention-based sequence modeling
- **LSTM**: Long short-term memory networks
- **GRU**: Gated recurrent units
- **TCN**: Temporal convolutional networks
- **Informer**: Efficient transformer variant
- **Autoformer**: Auto-correlation transformer
- **FEDformer**: Frequency enhanced transformer
- **Pyraformer**: Pyramid attention transformer
- **LogTrans**: Log-sparse transformer
- **Reformer**: Efficient transformer with reversible layers
- **Performer**: Linear attention transformer
- **ConvTimeNet**: Convolutional time series network
- **ModernTCN**: Modern temporal convolutional network

### Foundation Models
- **Chronos**: Large language model for time series
- **Moirai**: Unified time series foundation model
- **MOMENT**: Multi-modal time series foundation model

## 🔧 Data Processing Pipeline

### Data Preprocessing
- **Anomaly Detection**: Identify and fix outliers using IQR method
- **Zero Sequence Handling**: Detect and interpolate continuous zero sequences
- **Site Clustering**: Geographic clustering using DBSCAN algorithm
- **Feature Engineering**: Temporal and spatial feature extraction

### Data Aggregation
- **Temporal Aggregation**: Hourly to daily aggregation
- **Spatial Aggregation**: Charger-level to sitevel aggregation
- **Cross-Validation**: Time-based data splitting for evaluation

## 📊 Data Schema

### Core Metrics
- **Volume**: Number of charging sessions per time period
- **Duration**: Average charging session duration

### Auxiliary Features
- **Fee**: Charging cost information
- **Weather**: Temperature, humidity, precipitation, wind speed
- **Temporal**: Hour of day, day of week, month
- **Spatial**: Site coordinates, cluster centroids, distance matrices
- **POI**: Nearby amenities, commercial areas, transportation hubs

## 🎯 Use Cases

### 1. Infrastructure Planning
- **Capacity Planning**: Predict future charging demand
- **Location Optimization**: Identify optimal siteacement
- **Grid Integration**: Plan electrical infrastructure upgrades

### 2. Operational Optimization
- **Load Balancing**: Distribute charging load across site
- **Pricing Strategy**: Dynamic pricing based on demand patterns
- **Maintenance Scheduling**: Predictive maintenance planning

### 3. Policy Making
- **Incentive Programs**: Design effective EV adoption incentives
- **Regulatory Framework**: Develop charging infrastructure regulations
- **Urban Planning**: Integrate EV charging into city development

## 🔬 Research Applications

### 1. Time Series Forecasting
- **Short-term Prediction**: Hourly/daily demand forecasting
- **Long-term Planning**: Monthly/quarterly trend analysis
- **Anomaly Detection**: Identify unusual charging patterns

### 2. Federated Learning
- **Privacy Preservation**: Train models without sharing raw data
- **Cross-City Learning**: Transfer knowledge between cities
- **Scalable Training**: Distributed training across multiple locations

### 3. Transfer Learning
- **Domain Adaptation**: Adapt models to new cities
- **Knowledge Transfer**: Leverage data from similar cities
- **Cold Start**: Handle cities with limited historical data


## 🤝 Contributing

We welcome contributions!

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: OpenStreetMap, Weather APIs, City Open Data Portals
- **Model Implementations**: PyTorch, HuggingFace
- **Research Community**: Time series forecasting and federated learning researchers

## 📞 Contact

- **Email**: guozh29@mail2.sysu.edu.cn
- **Issues**: [GitHub Issues](https://github.com/IntelligentSystemsLab/CHARGED/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IntelligentSystemsLab/CHARGED/discussions)

---

**Note**: This is a research project. Please ensure compliance with local data privacy regulations when using this dataset.

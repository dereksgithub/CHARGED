# LLM-Based EV Charging Prediction Experiments

This directory contains modular experiment notebooks and scripts for developing and evaluating LLM-based models for EV charging demand prediction.

## 📁 Directory Structure

```
experiment/
├── README.md                          # This file
├── 01_baseline_benchmark.ipynb        # Phase 1: Baseline evaluation
├── 02_foundation_model_finetuning.ipynb  # Phase 2: Foundation model fine-tuning
├── 03_transfer_learning.ipynb         # Phase 3: Transfer learning
├── utils/
│   ├── experiment_tracking.py         # Experiment management
│   ├── visualization.py               # (To be created) Plotting utilities
│   └── statistical_tests.py           # (To be created) Statistical analysis
├── transfer_learning/
│   ├── pretrain_multi_city.py         # (To be created) Multi-city pre-training
│   ├── finetune_target_city.py        # (To be created) Fine-tuning scripts
│   └── evaluate_transfer.py           # (To be created) Evaluation
└── advanced_techniques/                # (To be created) Phase 4 experiments
    ├── multitask_learning.py
    ├── ensemble_prediction.py
    └── federated_metalearning.py
```

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies (already in CHARGED)
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm

# Foundation models (install as needed)
pip install momentfm              # For MOMENT
pip install chronos-forecasting   # For Chronos
pip install uni2ts                # For Moirai
```

### Phase 1: Baseline Benchmarking

Run comprehensive baseline evaluation:

```bash
# Open Jupyter notebook
jupyter notebook 01_baseline_benchmark.ipynb

# Or run from command line (when script version is created)
python baseline_benchmark.py --city SZH --models moderntcn lstm --epochs 50
```

**What it does:**
- Evaluates all 10 baseline models (statistical + neural)
- Tests with/without auxiliary features
- Generates comparison visualizations
- Identifies best baseline to beat

**Expected output:**
- `results/baselines/comprehensive_baseline_results.csv`
- `results/baselines/baseline_comparison.png`
- Individual model results in `results/baselines/baseline_*/`

### Phase 2: Foundation Model Fine-Tuning

Fine-tune pre-trained models:

```bash
jupyter notebook 02_foundation_model_finetuning.ipynb
```

**What it does:**
- Loads pre-trained MOMENT/Chronos/Moirai
- Implements progressive unfreezing
- Incorporates auxiliary features
- Compares with Phase 1 baselines

**Target:** >10% MAE improvement over best baseline

### Phase 3: Transfer Learning

Demonstrate cross-city transfer learning:

```bash
jupyter notebook 03_transfer_learning.ipynb
```

**What it does:**
- Pre-trains on data-rich cities (SZH, AMS, LOA)
- Fine-tunes on data-sparse cities (MEL, JHB, SPO)
- Compares transfer learning vs training from scratch

**Target:** 15-30% improvement with transfer learning

## 📊 Experiment Tracking

All experiments use the `ExperimentTracker` utility:

```python
from experiment.utils.experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker('my_experiment')

# Log configuration
tracker.log_hyperparameters({
    'model': 'moderntcn',
    'lr': 1e-4,
    'batch_size': 32
})

# Log results
tracker.log_metrics({
    'MAE': 12.5,
    'RMSE': 18.3,
    'R²': 0.82
})

# Save predictions
tracker.save_predictions(predictions, labels)

# Create summary
tracker.create_summary_table()
```

Results are automatically organized:
- Hyperparameters → `hyperparameters.json`
- Metrics → `metrics.json`
- Predictions → `{split}_predictions.npy`, `{split}_labels.npy`
- Summary → `summary.csv`

## 🎯 Performance Targets

Based on the plan, here are the targets for each phase:

| Phase | Metric | Target | Baseline |
|-------|--------|--------|----------|
| Phase 1 | MAE | Establish baseline | Likely 15-20 (ModernTCN) |
| Phase 2 | MAE | <13.5 (10% improvement) | Best baseline |
| Phase 3 | MAE | <11.5 (transfer learning) | Phase 2 result |
| Phase 4 | MAE | <10.0 (30-40% total improvement) | Original baseline |

## 📝 Workflow

### Recommended Execution Order

1. **Start with Phase 1** (Required)
   - Run baseline benchmarks on Shenzhen first
   - Identify best baseline model
   - Document performance metrics

2. **Then Phase 2** (Foundation models)
   - Start with MOMENT (best multivariate support)
   - Try zero-shot first (run existing test scripts)
   - If needed, implement fine-tuning

3. **Then Phase 3** (Transfer learning)
   - Pre-train on SZH+AMS+LOA
   - Fine-tune on MEL (smallest target)
   - Compare with from-scratch training

4. **Finally Phase 4** (Advanced techniques)
   - Multi-task learning
   - Ensemble methods
   - Federated meta-learning

### Parallel Development

You can also work in parallel:
- **Path A**: Phase 1 → Analyze results → Write paper
- **Path B**: Study foundation model APIs → Implement wrappers
- **Path C**: Design transfer learning strategy

## 🔧 Troubleshooting

### Common Issues

**GPU Out of Memory:**
```python
# Reduce batch size
CONFIG['batch_size'] = 16  # or 8

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
```

**Module Import Errors:**
```bash
# Make sure you're in the CHARGED directory
cd /home/mengyuwsl/CHARGED

# Add to path in notebook
import sys
sys.path.insert(0, os.path.abspath('..'))
```

**Foundation Model Installation:**
```bash
# MOMENT
pip install momentfm

# Chronos (if above doesn't work)
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Moirai
pip install uni2ts
```

## 📚 References

### Key Files to Study

- `api/model/modules.py` - All 10 baseline models
- `api/trainer/common.py` - Training loop
- `api/utils.py` - Evaluation metrics
- `example/univariate_prediction.py` - Training workflow
- `example/test_moment_all.py` - MOMENT integration
- `example/test_chronos_all.py` - Chronos integration
- `example/test_moirai_all.py` - Moirai integration

### Documentation

- Main plan: `~/.claude/plans/lively-tumbling-tower.md`
- Data schema: `data/README.md`
- Model details: `api/model/modules.py` docstrings

## 🎓 Learning Resources

### Understanding the Models

**Statistical Baselines:**
- AR/ARIMA: Traditional time series models
- Simple but effective baselines

**Neural Baselines:**
- LSTM: Recurrent neural network
- ModernTCN: Temporal convolutional network with RevIN
- ConvTimeNet: Modern convolutional architecture

**Foundation Models:**
- MOMENT: Multi-modal transformer for time series
- Chronos: T5-based language model adapted for forecasting
- Moirai: Salesforce's unified time series model

### Tips for Success

1. **Start Small**: Test on single city, single model first
2. **Validate Early**: Check data shapes and formats
3. **Log Everything**: Use ExperimentTracker consistently
4. **Compare Often**: Always reference baseline performance
5. **Document Findings**: Update README with insights

## 📧 Support

For issues related to:
- **CHARGED codebase**: Check original repository issues
- **Foundation models**: Refer to their respective documentation
- **Experiment design**: Review the comprehensive plan

## 🎉 Expected Outcomes

By completing these experiments, you will:
1. ✅ Comprehensive baseline benchmarks for 6 cities
2. ✅ Fine-tuned foundation models beating baselines
3. ✅ Demonstrated transfer learning effectiveness
4. ✅ Publication-ready results and visualizations
5. ✅ Framework for future EV charging prediction research

Good luck with your experiments! 🚀

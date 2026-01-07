# 🚀 Quick Start Guide - LLM-Based EV Charging Prediction

**Goal**: Beat CHARGED baselines by 30-40% using fine-tuned foundation models + transfer learning

---

## ⚡ 5-Minute Start

```bash
# 1. Navigate to project
cd /home/mengyuwsl/CHARGED

# 2. Open baseline benchmarking notebook
jupyter notebook experiment/01_baseline_benchmark.ipynb

# 3. Run all cells to establish baselines
#    (Uses ModernTCN, ConvTimeNet, LSTM on Shenzhen)

# 4. Check results
cat results/baselines/comprehensive_baseline_results.csv
```

**That's it!** You now have baseline performance to beat.

---

## 📊 What Each Notebook Does

### `01_baseline_benchmark.ipynb` ⭐ START HERE
**Purpose**: Establish baseline performance
**Runtime**: ~1-2 hours (3 models × 2 configs)
**Output**: Comprehensive comparison table + plots
**When**: Run this FIRST

### `02_foundation_model_finetuning.ipynb`
**Purpose**: Fine-tune MOMENT/Chronos/Moirai
**Runtime**: ~2-4 hours per model
**Output**: Fine-tuned model beating baselines
**When**: After Phase 1, when baselines are established

### `03_transfer_learning.ipynb`
**Purpose**: Demonstrate cross-city transfer learning
**Runtime**: ~4-6 hours (pre-train + fine-tune)
**Output**: Proof that transfer learning improves performance
**When**: After Phase 2, when single-city fine-tuning works

---

## 🎯 Your Targets

Based on the plan, here's what to aim for:

```
Phase 1 (Baselines):        MAE ~15-20  ← Establish this first
                                ↓
Phase 2 (Fine-tuning):      MAE ~12-15  ← 10-25% improvement
                                ↓
Phase 3 (Transfer):         MAE ~10-13  ← 15-30% additional
                                ↓
Phase 4 (Advanced):         MAE ~9-12   ← 30-40% total improvement
```

---

## 🔧 Installation (If Needed)

```bash
# Core dependencies (likely already installed)
pip install torch pandas numpy scikit-learn matplotlib seaborn

# Foundation models (install when you reach Phase 2)
pip install momentfm              # For MOMENT
pip install chronos-forecasting   # For Chronos
pip install uni2ts                # For Moirai
```

---

## 📈 Typical Workflow

### Day 1: Baselines
```bash
# Morning: Run baseline benchmarks
jupyter notebook experiment/01_baseline_benchmark.ipynb

# Afternoon: Analyze results
# - Which model is best? (likely ModernTCN or ConvTimeNet)
# - What's the MAE to beat?
# - Do auxiliary features help? (expect 5-15% improvement)
```

### Day 2-3: Foundation Models (Zero-Shot)
```bash
# Test foundation models without fine-tuning
python example/test_moment_all.py
python example/test_chronos_all.py

# Compare with baselines
# If MOMENT already beats baselines → Great! Document this.
# If not → Proceed with fine-tuning in Phase 2
```

### Day 4-5: Fine-Tuning (If Needed)
```bash
# Open fine-tuning notebook
jupyter notebook experiment/02_foundation_model_finetuning.ipynb

# Implement fine-tuning loop
# Target: Beat Phase 1 baseline by >10%
```

### Week 2: Transfer Learning
```bash
# Open transfer learning notebook
jupyter notebook experiment/03_transfer_learning.ipynb

# Pre-train on SZH+AMS+LOA, fine-tune on MEL
# Target: Beat single-city training by 15-30%
```

---

## 📁 Where Are My Results?

```
results/
├── baselines/
│   ├── comprehensive_baseline_results.csv  ← Summary table
│   ├── baseline_comparison.png             ← Visualization
│   └── baseline_*/                         ← Individual runs
│
├── foundation_models/
│   └── foundation_*/                       ← Phase 2 results
│
└── transfer_learning/
    └── template_status.csv                 ← Phase 3 results
```

---

## 🆘 Common Issues

### "ModuleNotFoundError: No module named 'api'"
```python
# Add to notebook cell
import sys
sys.path.insert(0, '..')
```

### "CUDA out of memory"
```python
# Reduce batch size
CONFIG['batch_size'] = 16  # or even 8
```

### "Foundation model import error"
```bash
# Install the specific model
pip install momentfm  # or chronos-forecasting or uni2ts
```

---

## 📚 Key Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `IMPLEMENTATION_STATUS.md` | Detailed progress tracker | Check what's done |
| `experiment/README.md` | Comprehensive guide | Deep dive into experiments |
| `~/.claude/plans/lively-tumbling-tower.md` | Full implementation plan | Understand overall strategy |
| `api/model/foundation.py` | Foundation model wrappers | Phase 2 development |
| `experiment/utils/experiment_tracking.py` | Experiment logging | Use in all experiments |

---

## 💡 Pro Tips

1. **Start Small**: Run on Shenzhen only first (largest dataset)
2. **Log Everything**: Use `ExperimentTracker` in every experiment
3. **Compare Often**: Always compare against Phase 1 baselines
4. **Save Checkpoints**: Foundation models take time to train
5. **Use GPU**: Enable CUDA for faster training

---

## 🎓 Understanding Your Models

### Baselines (Phase 1)
- **ModernTCN**: Modern temporal convolutional network (likely best)
- **ConvTimeNet**: Convolutional encoder (strong competitor)
- **LSTM**: Classic recurrent network (good baseline)

### Foundation Models (Phase 2)
- **MOMENT**: Multi-modal transformer, best for multivariate data
- **Chronos**: T5-based, simple API
- **Moirai**: Salesforce model, any-variate support

---

## 🏁 Success Criteria

You've succeeded when:
- ✅ Baseline benchmarks complete (Phase 1)
- ✅ Foundation model beats best baseline by >10% (Phase 2)
- ✅ Transfer learning beats from-scratch by >15% (Phase 3)
- ✅ Total improvement >30-40% from original baseline
- ✅ Results are reproducible and well-documented

---

## 🚦 Current Status

- ✅ Infrastructure complete
- ✅ Notebooks ready
- ✅ Foundation model wrappers created
- ⏳ Baselines need to be run ← **YOU ARE HERE**
- ⏳ Fine-tuning to be implemented
- ⏳ Transfer learning to be demonstrated

---

## 🎯 Next 3 Actions

1. **Run**: `jupyter notebook experiment/01_baseline_benchmark.ipynb`
2. **Wait**: Let it run (1-2 hours)
3. **Analyze**: Check `results/baselines/comprehensive_baseline_results.csv`

Then come back to Phase 2! 🚀

---

**Questions?** Check:
- `experiment/README.md` for detailed guidance
- `IMPLEMENTATION_STATUS.md` for progress tracking
- Original examples in `example/` directory
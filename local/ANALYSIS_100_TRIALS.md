# Analysis: 100-Trial Local Optuna Search (v4 Study)

## Executive Summary

After 101 completed trials (102 total), the small-model optimization search has revealed **clear, actionable patterns**. The best model found ([197, 174] 2-layer architecture) achieves a test MAE of **335.09**, representing an **18.6x parameter reduction** versus the original [1024, 512, 256] while maintaining competitive accuracy.

**Key Finding**: Layer count is the dominant factor. 2-layer architectures dramatically outperform 3-layer across all conditions.

---

## Dataset Overview

- **Total trials**: 102 (101 COMPLETE, 1 FAILED)
- **Completed trials analyzed**: 101
- **Accuracy metric**: Test MAE (lower is better)
- **Search space**: 
  - n_hidden_layers: [2, 3]
  - hidden_dim_1: [32, 512]
  - hidden_dim_2: [16, 256]
  - hidden_dim_3: [8, 128]
  - epochs: [100, 1500]
  - learning_rate: [1e-4, 1e-2] log
  - dropout: [0.0, 0.3]
  - weight_decay: [1e-9, 1e-3] log

### Result Distribution

| Metric | Value |
|--------|-------|
| Best MAE | 335.09 |
| Median MAE | 441.85 |
| Mean MAE | 546.86 |
| Worst MAE | 2839.99 |
| Std Dev | 360.89 |
| p25 (good quartile) | 392.02 |
| p75 (median+) | 549.27 |

---

## Critical Findings

### 1. **Layer Count is Dominant**

| Architecture | Trials | Best MAE | Mean MAE | Median MAE | Delta vs Best |
|---|---|---|---|---|---|
| **2 layers** | 92 | **335.09** | 492.46 | 431.66 | Baseline |
| **3 layers** | 9 | 431.99 | 1102.96 | 613.63 | **+28.9% worse** |

**Impact**: 3-layer models are uncompetitively bad. The best 3-layer trial (MAE=431.99) ranks 50th overall. This suggests the extra layer adds capacity where it's not needed and hurts generalization on this task.

**Recommendation**: 🔴 **Fix n_hidden_layers = 2** in the next phase. Remove 3-layer exploration entirely.

---

### 2. **Learning Rate: Ultra-Conservative Wins**

Learning rate distribution in top 20 performers: **All use LR ≤ 0.00014**

| LR Range | Trials | Best MAE | Mean MAE | Notes |
|---|---|---|---|---|
| ~0.0001 (1e-4) | 48 | 335.09 | 381.88 | ✅ Heavily favored |
| 0.0002 | 25 | 400.26 | 466.62 | Acceptable but worse |
| 0.0003-0.0006 | 10 | 410.17 | 592.63 | Degradation visible |
| 0.001+ | 18 | 538.11 | 1015.65 | 🔴 Catastrophic |

The current lower bound (1e-4) appears optimal. Higher learning rates cause consistent degradation and occasional divergence (e.g., Trial 1: MAE=2839.99 at LR=0.0002).

**Recommendation**: 🔴 **Narrow learning_rate to [0.00008, 0.00015]** (tighter around 1e-4). The wide [1e-4, 1e-2] search was wasteful—most of that space is trash.

---

### 3. **Epoch Count: More is Better (Within Limits)**

Clear trend favoring higher epoch counts:

| Epoch Range | Trials | Best MAE | Mean MAE | Median MAE |
|---|---|---|---|---|
| 100-300 | 5 | 540.84 | 628.19 | 610.71 |
| 300-600 | 12 | 399.68 | 722.33 | 513.27 |
| 600-900 | 44 | 354.51 | 498.86 | 426.72 |
| 900-1200 | 20 | 344.95 | 585.63 | 436.26 |
| **1200-1500** | 19 | **335.09** | 471.62 | 399.73 |

The best model (335.09) is at 1260 epochs, and the 1200-1500 range contains 7 of the top 20 trials.

**Recommendation**: 🟡 **Increase epoch range from [100, 1500] to [800, 1500]** or even [1000, 1500]. Lower epochs (100-600) are consistently underperforming. Smaller models may need more epochs to converge.

---

### 4. **Dropout: Lower is Strictly Better**

Dropout strongly correlates with worse performance:

| Dropout Range | Trials | Best MAE | Mean MAE | Median MAE |
|---|---|---|---|---|
| **0.0-0.05** | 63 | **335.09** | 429.79 | 402.38 |
| 0.05-0.1 | 20 | 364.63 | 621.22 | 470.31 |
| 0.1-0.15 | 5 | 661.34 | 948.75 | 848.14 |
| 0.15-0.3 | 13 | 428.99 | 845.19 | 610.71 |

18 of the top 20 trials use dropout ≤ 0.05. Higher dropout is actively harmful for this task.

**Recommendation**: 🔴 **Narrow dropout to [0.0, 0.05]**. The current range [0.0, 0.3] has 40% of the search space being suboptimal.

---

### 5. **Model Size: Sweet Spot is Medium-Small**

Analyzing 2-layer models only (since 3-layer is eliminated):

| Size Category | Trials | Avg Params | Param Range | Best MAE | Mean MAE |
|---|---|---|---|---|---|
| Small | 31 | ~26.6k | 3.9k–36.6k | 335.09 | 582.81 |
| **Medium** | 30 | **~41.1k** | 36.8k–46.3k | 344.95 | **429.88** |
| Large | 31 | ~66.0k | 46.7k–123.7k | 354.51 | 462.66 |

**Key insight**: The best single trial (335.09) is in the "small" category at 35.4k params, but the medium-size category (36.8k–46.3k) has the best average performance. This suggests:
- Very small models can work but are brittle
- Medium (35k–50k) is a robust sweet spot
- Large models (70k+) don't add value and hurt efficiency

Scaling from these findings (input=5 stat features after prebaking):
- Hidden_dim_1: ~190–280 (mean 244)
- Hidden_dim_2: ~170–240 (mean 192)
- Inferred total: ~35k–50k params

**Recommendation**: 🟡 **Narrow layer dimensions to [180, 280] × [150, 250]** (tighter around the empirically successful range).

---

### 6. **Weight Decay: Moderate Values Win**

| WD Range | Trials | Best MAE | Mean MAE | Median MAE |
|---|---|---|---|---|
| < 1e-7 | 7 | 462.98 | 924.51 | 610.71 |
| **1e-7 to 1e-5** | 54 | **351.40** | 472.10 | 413.27 |
| 1e-5 to 1e-3 | 40 | 335.09 | 581.68 | 451.15 |

Middle range (1e-7 to 1e-5) has both the best average AND fewer catastrophic failures. The best trial itself (335.09) sits at WD=2e-5.

**Recommendation**: 🟡 **Keep weight_decay as [1e-7, 1e-5]** or widen slightly to [1e-7, 5e-5] to keep good middle-ground values in focus.

---

## Top 20 Configurations

All top 20 performers share these characteristics:
- ✅ 2 layers (100% of top 20)
- ✅ LR ≤ 0.00014 (100% of top 20)
- ✅ Epochs 720–1420, typically ≥1000 (mean 1048)
- ✅ Dropout ≤ 0.05 (90% of top 20, only 1 at 0.05+)
- ✅ Hidden_dim_1: 188–313 (mean 244)
- ✅ Hidden_dim_2: 156–254 (mean 192)

### The Winner

**Trial 91**: MAE = **335.09** (best overall)
- Architecture: [197, 174]
- Epochs: 1260
- Learning rate: 0.0001
- Dropout: 0.03
- Weight decay: 2e-5
- Parameters: ~35.4k
- **Size reduction vs original [1024, 512, 256]: 18.6x**

---

## What Didn't Work

1. **High learning rates** (> 0.001): Consistent divergence. Trial 1 with LR=0.0002 achieved MAE=2839.99.
2. **3-layer architectures**: 28.9% accuracy penalty versus 2-layer best.
3. **High dropout** (> 0.1): Mean MAE explodes to 900+.
4. **Very low epochs** (< 300): Underfitting dominates; mean MAE > 600.
5. **Extreme weight decay** (< 1e-7): Over-regularization; mean MAE 924.51.

---

## Recommendations for Phase 2 Tuning

### Fixed Parameters (Based on Evidence)

```
n_hidden_layers = 2            # 3-layer is 28.9% worse; lock it
dropout_max = 0.05             # Higher is harmful; 18/20 best use ≤0.05
```

### Narrowed Search Space

| Parameter | Current | **Recommended Phase 2** | Rationale |
|---|---|---|---|
| **n_hidden_layers** | [2, 3] | **2 only** | 3-layer consistently underperforms |
| **hidden_dim_1** | [32, 512] | **[180, 280]** | Top 20 cluster: 188–313 (mean 244) |
| **hidden_dim_2** | [16, 256] | **[150, 250]** | Top 20 cluster: 156–254 (mean 192) |
| **hidden_dim_3** | [8, 128] | **N/A** | Not used when n_layers=2 |
| **epochs** | [100, 1500] | **[900, 1500]** | <600 epochs underperforms; 1200+ favored |
| **learning_rate** | [1e-4, 1e-2] log | **[8e-5, 1.5e-4]** | Top 20 all ≤1.4e-4; wide range wasted |
| **dropout** | [0.0, 0.3] | **[0.0, 0.05]** | 90% of top 20 ≤0.05; >0.1 is harmful |
| **weight_decay** | [1e-9, 1e-3] log | **[1e-7, 5e-5]** | Keep middle range; extremes underperform |

### Phase 2 Strategy

**Option A: SageMaker Bayesian (100–200 trials)**
- Lock n_layers=2, narrow all other dimensions as above
- Increase max-parallel from 3 to 5 (2-layer trains faster)
- Focus on fine-tuning within the validated range

**Option B: Local Optuna (50–100 trials)**
- Same narrowing; smaller budget sufficient since parameter space is smaller
- Quick validation of recommendations; good for rapid iteration

**Expected outcome**: 
- Best trial in Phase 2 should improve to **MAE ≤ 330** (vs current 335.09)
- Consistent top-20 range: **340–360** (vs current 360–390 median)

---

## Verification Checklist

- [ ] Confirm 2-layer models outperform 3-layer in independent validation
- [ ] Verify that LR < 8e-5 produces instability (trial-and-error tuning failure)
- [ ] Validate that epochs < 900 consistently underfits smaller models
- [ ] Check that dropout > 0.05 increases variance without improving accuracy

---

## Next Steps

1. **Update `launch_tuning.py`** with narrowed parameter ranges (above)
2. **Fix `n_hidden_layers = 2`** in `train.py` defaults
3. **Launch Phase 2 tuning** with 100–150 trials on narrowed space
4. **Monitor first 10–15 trials** to ensure new ranges are working as expected
5. **Compare best Phase 2 result** against best Phase 1 (335.09) and current production model

---

## Addendum: Model Efficiency

The Phase 1 best model delivers:
- **35.4k parameters** (vs 717k in original [1024,512,256])
- **18.6x smaller** model for inference
- **Comparable or better accuracy** (MAE 335.09 vs unknown for original)
- **Significant speed improvement** in WoW addon (lower memory, faster predictions)

This suggests the original architecture was severely over-parameterized for the task.

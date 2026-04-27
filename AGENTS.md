# AGENTS.md: IRS Microwave Classification Project

## Project Overview
This project implements two parallel machine learning pipelines for 3-class microwave signal classification:
1. **Raw Data Pipeline** (`Raw_Data_Implementation/`): 1D CNN trained on preprocessed time-series signals
2. **Feature Pipeline** (`Feature_Implementation/`): MLP trained on hand-crafted statistical features

Both pipelines use identical data preprocessing and 5-fold stratified CV, enabling direct comparison of learned representations.

## Architecture & Data Flow

### Data Structure Convention
- **Raw signals**: `Dict[class_label, pd.DataFrame]` where DataFrames have shape `[N_samples, signal_length]`
- **Processed arrays**: `[N, C, L]` tensor for signals (N=samples, C=channels, L=length)
- **Features**: `[N, F]` tabular arrays (N=samples, F=features)
- **Labels**: `y` shape `[N]` integer class indices (CrossEntropyLoss format)

### Pipeline Stages
1. **Data Loading** → Excel read, groupby class label (`LOW`, `TARGET`, `HIGH`)
2. **Preprocessing** → Signal segments extraction, spike filtering, Savitzky-Golay smoothing, downsampling, derivative computation
3. **Multi-channel Dataset** → Combines original, first-diff, second-diff channels
4. **Train/Val Split** → 85% trainval / 15% test holdout
5. **CV Fold Creation** → 5-fold stratified split with fold-specific normalization (train only)
6. **Training** → Model fitting with early stopping, TensorBoard logging
7. **Visualization** → Plot predictions, samples, heatmaps

### Raw Data Pipeline Specifics
- **Models**: `OneDCNNClassifier`, `MultiScaleOneDCNNClassifier`, `DualBranchOneDCNNClassifier`, `TCNClassifier`
- **Input**: 2-channel signals `[N, 2, L]`
- **Output**: `TrainOutput` TypedDict with fold results, confusion matrices, per-class metrics
- **Feature extractor**: 4-layer CNN with configurable kernels/strides/dilations
- **Data augmentation**: Random temporal shifting during training (see `MicrowaveSignalDataset._apply_random_shift`)

### Feature Pipeline Specifics
- **Extraction**: ~100 statistical features per channel (mean, std, skewness, kurtosis, spectral entropy, zero-crossing rate, etc.)
- **Model**: `FeatureMLPClassifier` with hidden layers `[input_dim → 512 → 256 → 3]`
- **Normalization**: StandardScaler fitted only on training folds, applied to validation
- **Output**: `FeatureTrainOutput` TypedDict (compatible structure with raw pipeline)

## Critical File Organization

### Must-Read Files
- `Raw_Data_Implementation/Model_Structure.py` — Centralized config at top (lines 5–64), model definitions
- `Feature_Implementation/Feature_Extraction.py` — Statistical feature computation logic
- `Utility_Functions/Preprocessing.py` — Shared signal processing (Savgol filtering, resampling, rolling stats)
- `Raw_Data_Implementation/main.py` / `Feature_Implementation/main.py` — Complete workflow examples

### Key Cross-References
- Both pipelines import from `Utility_Functions.Preprocessing` for common preprocessing steps
- Both share identical CV fold indices via `build_stratified_cv_indices`
- Model configs stored as module-level constants (not JSON/YAML)
- Results logged to `runs/{model_type}_cv/fold_{i}/` for TensorBoard

## Workflow Patterns

### Development Iteration Pattern (From main.py)
```python
importlib.reload(Model_Training)  # Reload after editing model code
importlib.reload(Model_Structure)  # Required for grid search loop
```
**Why**: Allows running multiple training loops in same Python session without restart.

### Model Construction
- All models inherit from `nn.Module`
- Use `LazyLinear()` for adaptive input dimensions after CNN feature extraction
- Activation: `LeakyReLU` with configurable slope, not standard ReLU

### CV Fold Processing
Each fold dict contains:
```python
{
    "fold": int,
    "train_idx": np.ndarray,
    "val_idx": np.ndarray,
    "X_train": np.ndarray,           # Normalized [N, C, L]
    "y_train": np.ndarray,           # [N] integer labels
    "X_val": np.ndarray,             # Normalized with train statistics
    "y_val": np.ndarray,
    "feature_scaler": StandardScaler # For transform-time normalization
}
```
**Critical**: Validation is always normalized using training fold statistics (prevents leakage).

### Training Output Structure
```python
{
    "label_to_idx": {"LOW": 0, "TARGET": 1, "HIGH": 2},
    "idx_to_label": {0: "LOW", ...},
    "device": "cuda" or "cpu",
    "fold_results": [FoldResult, ...],
    "mean_best_val_acc": float,
    "overall_confusion_count": np.ndarray [3, 3]
}
```

## Project-Specific Conventions

### Signal Preprocessing Cascade
1. Slice segments: `(0, 1000)`, `(1800, 3500)` indices
2. Spike filter: Threshold-based with sqrt transform, `n_sigmas=3.0`
3. Savgol filter: `window_length=31, polyorder=3`
4. Downsample: `step=5` (120× reduction for 3500-point signals)
5. Compute derivatives: First and second-order central differences
6. Re-filter derivatives: Same Savgol parameters

**Rationale**: Multi-scale temporal representation; low-frequency + edges + accelerations.

### Grid Search Pattern
- Define search space as dict of parameter lists
- All parameter combinations tested exhaustively
- Results saved with timestamp: `runs/grid_search_results/YYYYMMDD_HHMMSS/`
- Output includes trial rank, best accuracy per trial, per-fold breakdowns

### Hyperparameter Defaults (Config at Module Top)
**Raw model** (1D CNN):
- Channels: `(32, 64, 128, 256)`
- Kernels: `(5, 5, 5, 5)`
- Strides: `(2, 2, 2, 2)`
- Dropout: `0.1`
- LeakyReLU slope: `0.05`

**Feature model** (MLP):
- Hidden: `[512, 256]`
- Dropout: varies by architecture
- LeakyReLU slope: `0.01`

### Plotting Interface
Controlled via dict-based config:
```python
PLOT_OPTIONS = {"threshold_hits": bool, "classification_examples": bool, ...}
PLOT_CONFIG = {"threshold_hits": {specific_params}, ...}
```
Functions in `Utility_Functions.Viewing.py` accept both fold data AND config dicts.

## Common Tasks & Patterns

### Adding a New Model Variant
1. Define model class in `Raw_Data_Implementation/Model_Structure.py`
2. Add config parameters at module top (lines 5–64)
3. Register in `Model_Training.py` imports and `train_1d_cnn_cv()` model selection
4. Test with single fold before grid search

### Modifying Preprocessing
1. Edit `Utility_Functions/Preprocessing.py` function (or add new one)
2. Call it in both `Raw_Data_Implementation/main.py` AND `Feature_Implementation/main.py` (keep pipelines aligned)
3. Verify output shape matches expectations: dict of DataFrames for dicts, `[N, C, L]` for arrays

### Running Grid Search
1. Set `run_grid_search = True` in `Raw_Data_Implementation/main.py`
2. Define `grid_search_space` (dict of lists)
3. Call results saved automatically; print top-k via `Grid_Search.print_top_grid_results(grid_out, top_k=100)`
4. Parameter frequency summary via `Grid_Search.summarize_top_param_frequencies()`

## Dependencies & External Tools
- **PyTorch**: Models, training loops, TensorBoard logging
- **scikit-learn**: Preprocessing (StandardScaler, StratifiedKFold), train_test_split
- **pandas**: Data loading/grouping
- **scipy**: Savitzky-Golay filtering, FFT
- **numpy**: Array operations
- **matplotlib/seaborn** (implicit in Viewing): Plotting

## Debugging Tips
- CV fold indices should be identical across pipelines → verify via `build_stratified_cv_indices()`
- Normalization leakage if validation uses global statistics → always use training fold stats
- Model shape mismatch: Check `[N, C, L]` vs `[N, L]` conventions at fold construction
- TensorBoard logs: `tensorboard --logdir=runs/`
- Reload modules after code edits: `importlib.reload(Model_Training)` before re-running training loop

## Key Assumption: Random Seed
- Constant seed: `42` across all random operations (train/test split, CV, PyTorch, NumPy)
- Enables reproducible results across runs
- If randomness needed, explicitly override in main scripts


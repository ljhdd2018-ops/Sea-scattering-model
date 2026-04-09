# DS-Hybrid Surrogate Model for Sea-Surface Electromagnetic Scattering

Pre-trained surrogate model that predicts far-field scattering patterns of sea-surface facets. Given physical parameters (wind speed, frequency, incidence angle, polarization), the model outputs a full 2-D angular field map |E(θ, φ)| in the linear amplitude domain.

## How It Works

The model combines two components:

1. **DS lobe** — a parametric directional-spread function capturing the main-lobe shape with 7 parameters (asymmetric θ widths, φ width, shape exponents, rotation, peak amplitude)
2. **PCA residual** — learned principal-component correction for sidelobe structure and fine detail

Both components are regressed from input features via LSBoost ensemble trees. At prediction time, the DS lobe is reconstructed analytically and the PCA residual is added on top.

## Repository Contents

```
.
├── model.mat            # Pre-trained model (MATLAB, v7.3)
├── example_usage.m      # Complete usage examples
└── README.md
```

## Requirements

- MATLAB R2020b or later
- Statistics and Machine Learning Toolbox

## Quick Start

```matlab
load('model.mat', 'model');
```

### Predict a single scenario

```matlab
U = 3;  freqGHz = 1.5;  thetaIncDeg = 30;  pol = 'TE';
sampleType = 'single_lobe';   % or 'phi_broad', 'no_clear_lobe'

Emap = predict_field(model, model.config, thetaRel, phiRel, ...
                     U, freqGHz, thetaIncDeg, pol, sampleType);
imagesc(model.phiGrid, model.thetaGrid, Emap); axis xy; colorbar;
```

### Compare with FEKO reference (stored inside model)

```matlab
idx = 1;
Y_true = reshape(model.Y_feko(idx,:), nTheta, nPhi);
Y_hat  = reshape(model.Y_pred(idx,:), nTheta, nPhi);
```

Run **`example_usage.m`** for the full working script including grid setup, batch prediction, visualization, and CSV export.

## Input Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `U` | U | Sea-state wind speed level (integer) |
| `freqGHz` | f | Frequency in GHz |
| `thetaIncDeg` | θi | Incidence angle in degrees |
| `pol` | — | Polarization: `'TE'` or `'TM'` |
| `sampleType` | — | Lobe class: `'single_lobe'`, `'phi_broad'`, or `'no_clear_lobe'` |

## Model Structure (`model.mat`)

| Field | Description |
|-------|-------------|
| `thetaGrid`, `phiGrid` | Angular grid vectors (degrees) |
| `dsModels` | 7 regression models for DS lobe parameters |
| `dsR2` | Training R² per DS parameter |
| `muRes` | PCA residual mean vector |
| `basis` | PCA basis matrix (nGrid × K) |
| `scoreModels` | K regression models for PCA scores |
| `bestK` | Optimal K selected by 5-fold cross-validation |
| `Y_feko` | FEKO reference fields (nSamples × nGrid) |
| `Y_pred` | Model predictions on training set |
| `cvCorrMedian` | Cross-validated correlation (median) |
| `cvRmseMedian` | Cross-validated RMSE (median) |
| `config` | Full configuration struct used during training |

## Feature Vector (internal)

The 10-dimensional feature vector is built automatically by `build_features_local` in `example_usage.m`:

`[1, U, U², log10(f), θi, θi², cos(θi), pol, U·log10(f), U·θi]`

where pol = 0 for TE, 1 for TM.

## License

MIT

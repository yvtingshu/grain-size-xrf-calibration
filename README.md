# Grain-Size Constrained Quantitative Calibration of Marine Sediment XRF Data Using Ensemble Machine Learning

## Overview

This repository contains the de-identified Python implementation associated with the manuscript **“Grain-Size Constrained Quantitative Calibration of Marine Sediment XRF Data Using Ensemble Machine Learning.”**

The study analyzed **214 paired calibration samples from four marine sediment cores** and used X-ray fluorescence (XRF) scanning information together with grain-size parameters to predict ten major-element oxides. The original institutional datasets are not distributed with this repository because they are subject to confidentiality and data-security restrictions.

The released script implements the computational workflow used for:

- depth alignment and XRF window averaging;
- raw-XRF and centered log-ratio (CLR) feature construction;
- nine grain-size features;
- Pearson-correlation feature screening;
- Z-score standardization based on the training data;
- four baseline configurations (BL-1 to BL-4);
- bagged multilayer perceptron (MLP) regression;
- ten repeated 50:50 random hold-out evaluations;
- R², RD%, and RSD% calculations;
- grain-size and trace-element feature-ablation experiments.

## Scope of This Public Release

The manuscript reports several validation and visualization analyses. The current public script reproduces the implementation for the **four-tier baseline comparison, repeated random hold-out evaluation, RD/RSD evaluation, and the two feature-ablation experiments**.

The current release does **not** include separate scripts for:

- the depth-separated independent validation using the upper 80% of the stratigraphic interval for training and the lower 20% for testing; or
- the full-core profile visualization presented in the manuscript.

Accordingly, this repository should be interpreted as the public implementation of the baseline and ablation analyses, not as a complete reproduction package for every figure and validation experiment in the manuscript.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── public_analysis_final.py
```

No original research data are included.

## Study Inputs

The manuscript used three matched data types:

1. **XRF scanning data**
2. **Laboratory major-element data**
3. **Grain-size data**

The manuscript model used **20 XRF elemental fluorescence-intensity features** and **9 grain-size parameters**. The released script identifies XRF columns that are common to the approved input bundles; therefore, manuscript-equivalent inputs should provide the same 20 XRF features used in the study.

The ten laboratory prediction targets are:

```text
SiO2, Al2O3, Fe2O3, CaO, MgO, K2O, Na2O, TiO2, MnO, P2O5
```

The nine standardized grain-size variables are:

```text
Gravel, Sand, Silt, Clay, Mean_phi, Median_phi,
Sorting, Skewness, Kurtosis
```

## De-identified Input Interface

To remove internal filenames and core/site identifiers, the public script uses two generic input bundles:

```text
data/
├── dataset_a_scan.xlsx
├── dataset_a_chem.xlsx
├── dataset_a_grain.xlsx
├── dataset_b_scan.xlsx
├── dataset_b_chem.xlsx
└── dataset_b_grain.xlsx
```

**Dataset A and Dataset B are de-identification/interface labels only. They do not represent the number of sediment cores analyzed in the manuscript.** The study itself used four sediment cores.

The default worksheet name is `Sheet1`.

Each input table must provide fields corresponding to a sample/station identifier, top depth, and bottom depth. The script standardizes these internally as:

```text
Station
Top_cm
Bot_cm
Center_cm
```

where:

```text
Center_cm = (Top_cm + Bot_cm) / 2
```

The public script retains source-column aliases needed to read the approved input schema. These aliases are data-field mappings rather than public study identifiers.

## Depth Alignment and Window Averaging

For each laboratory sampling interval, XRF measurements within the corresponding depth window are averaged. Samples with fewer than 80% of the expected valid XRF scanning points are excluded.

The two generic processing configurations retained in the public implementation are:

```text
Dataset A: half-window = 1.0 cm; expected XRF points = 4
Dataset B: half-window = 0.5 cm; expected XRF points = 5
```

These labels are de-identified processing configurations and should not be interpreted as manuscript core names.

## Feature Screening and Standardization

The manuscript and public script use Pearson-correlation screening based on the training set:

```text
Correlation threshold: |r| >= 0.12
Minimum retained features: 15
```

Zero-variance predictors are removed before model fitting.

Predictors and target variables are standardized using statistics estimated from the training data. The same fitted transformations are then applied to the test data.

## Four-Tier Baseline Framework

The public script implements the four baseline configurations described in the manuscript:

- **BL-1:** Raw XRF fluorescence intensities + multiple linear regression
- **BL-2:** CLR-transformed XRF + multiple linear regression
- **BL-3:** XRF-only MLP without grain-size constraints
- **BL-4:** XRF + nine grain-size parameters + MLP

The grain-size contribution is evaluated from the change between BL-3 and BL-4:

```text
ΔR² = R²(BL-4) - R²(BL-3)
```

## MLP Ensemble

Each target oxide is modeled with a separate MLP regressor. The implementation uses:

```text
Hidden layer:          18 neurons
Activation:            tanh
Solver:                lbfgs
L2 regularization:     alpha = 1.0
Maximum iterations:    5000
Tolerance:             1e-6
Base random state:     42
Bagging estimators:    20
Bagging random state:  42
```

`MultiOutputRegressor` is used to maintain separate regressors for the target variables, and `BaggingRegressor` constructs the ensemble.

## Random Hold-Out Evaluation

The released analysis uses **10 repeated random train-test splits**.

For each repetition:

1. the repetition index is used as the NumPy random seed;
2. samples are randomly permuted;
3. the data are divided approximately 50:50 into training and test sets;
4. the models are trained on the training set;
5. performance is evaluated on the held-out test set.

The released script reports:

- coefficient of determination (**R²**);
- relative deviation (**RD%**);
- relative standard deviation (**RSD%**).

RD% and RSD% are calculated for the full-feature BL-4 model in the released script.

## Feature-Ablation Experiments

Two feature-ablation configurations are implemented:

1. **Without grain-size features:** XRF features only; this corresponds to BL-3.
2. **Excluding trace-element XRF features:** major-element XRF features (Si, Al, K, Ca, Ti, Mn, Fe) combined with the nine grain-size parameters.

The same random split within each repetition is used when comparing the baseline and ablation configurations.

## Software Environment

The analysis environment used for this public release was:

```text
Python 3.13.9
Anaconda distribution
64-bit Windows
```

The exact Python package versions are listed in `requirements.txt`:

```text
pandas==2.3.3
numpy==2.3.5
scipy==1.16.3
scikit-learn==1.7.2
openpyxl==3.1.5
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Script

The confidential institutional datasets are not provided. Authorized users with appropriately structured and approved local data can place the inputs under `data/` using the generic filenames expected by the script and run:

```bash
python src/public_analysis_final.py
```

The script prints the four-baseline comparison and feature-ablation results to the console.

## Data Availability

The original data analyzed in the study are not included in this repository because they are internal institutional data subject to confidentiality and data-security restrictions. Public disclosure or external distribution of the raw data is not permitted under the applicable data-management requirements.

This repository therefore provides the de-identified analysis implementation only.

## Reproducibility and De-identification Note

The public script was audited against the original analysis script. The scientific calculations, model parameters, random hold-out logic, evaluation metrics, baseline definitions, and ablation logic were preserved.

Changes made for public release were limited to:

- removal or replacement of internal file/core/site identifiers;
- generic public-facing input paths;
- standardized variable naming;
- English comments and console labels; and
- removal of a non-computational hard-coded sample-count warning.

Because the confidential source datasets are not distributed, the numerical results reported in the manuscript cannot be reproduced from this repository alone.

## Citation

After the repository is archived and a DOI is assigned, add the DOI for the archived release here:

```text
DOI: 10.5281/zenodo.22009723
```

## License

No software license is bundled in this release package. A license should be added only after the authors/institution confirm the appropriate licensing terms for public distribution.

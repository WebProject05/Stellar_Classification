# Sloan Digital Sky Survey (SDSS): Stellar Object Classification
## Pattern Recognition Semester Project — Phase 1: Baseline Photometric Classification

An end-to-end machine learning study investigating multiclass classification of astronomical observations into **Stars**, **Galaxies**, and **Quasars (QSO)** using the Sloan Digital Sky Survey (SDSS) dataset.

---

## 1. Project Overview

The objective of Phase 1 is to establish a rigorous, transparent pattern recognition baseline using approximately 100,000 observations from the SDSS Data Release 17 dataset. In this initial baseline phase, classification is conducted using **strictly the five raw optical photometric measurements** ($u, g, r, i, z$) without domain-engineered color indices or spectroscopic redshift.

### Primary Astronomical Classes
1. **Stars (`STAR`)**: Dense, gravitationally bound spheroids of plasma within the Milky Way undergoing thermonuclear fusion in their cores. Optically, stars appear as diffraction-limited point sources with continuous blackbody-like emission spectra modified by atmospheric absorption lines.
2. **Galaxies (`GALAXY`)**: Immense cosmic structures comprising billions of stars, interstellar dust, molecular gas, and dark matter halos. Spatially resolved as extended, diffuse morphologies (elliptical, spiral, irregular).
3. **Quasars (`QSO` — Quasi-Stellar Objects)**: Extremely luminous and distant Active Galactic Nuclei (AGN) powered by supermassive black holes accreting matter at relativistic speeds in the centers of early galaxies. Because the compact central accretion disk completely outshines the host galaxy, quasars appear as unresolved point sources on optical CCDs, mimicking foreground stars despite existing at cosmological distances.

### Initial Research Question
> *How effectively can standard machine learning baseline classifiers (Logistic Regression, K-Nearest Neighbors, Linear SVM, and Random Forest) discriminate between stars, galaxies, and quasars using raw five-band optical photometry ($u, g, r, i, z$) alone, and what are their specific failure modes when spectroscopic redshift and color ratios are withheld?*

---

## 2. Repository Structure

```text
Stellar_Classification/
│
├── 01_sdss_baseline_classification.ipynb       # Phase 1: Baseline Photometric Classification (46 cells)
├── 02_sdss_color_feature_engineering.ipynb     # Variant 2: Color Feature Engineering Experiment (35 cells)
├── 03_sdss_redshift_ablation.ipynb             # Variant 3: Redshift Ablation Study (26 cells)
├── star_classification.csv                     # SDSS DR17 dataset (~100,000 rows x 18 columns)
├── create_sdss_notebook.py                     # Baseline notebook generator script
├── build_color_notebook.py                     # Color engineering notebook generator script
├── build_redshift_notebook.py                  # Redshift ablation notebook generator script
└── README.md                                   # Comprehensive project documentation
```

---

## 3. Dataset Requirements and Description

The dataset corresponds to the **SDSS Stellar Classification Dataset (Data Release 17)**, containing 100,000 observations and 18 attributes:

| Column Name | Type | Physical / Survey Role | Phase 1 Action | Justification |
| :--- | :--- | :--- | :--- | :--- |
| `obj_ID` | float64 | Photometric detection ID | **Excluded** | Administrative tracking key; causes spurious correlation. |
| `alpha` | float64 | Right Ascension (sky position) | **Excluded** | Celestial coordinate; prevents spatial memorization. |
| `delta` | float64 | Declination (sky position) | **Excluded** | Celestial coordinate; prevents spatial memorization. |
| `u` | float64 | Ultraviolet filter magnitude (~355 nm) | **Included** | Core optical photometric baseline feature. |
| `g` | float64 | Green optical filter magnitude (~468 nm) | **Included** | Core optical photometric baseline feature. |
| `r` | float64 | Red optical filter magnitude (~616 nm) | **Included** | Core optical photometric baseline feature. |
| `i` | float64 | Near-Infrared filter magnitude (~748 nm) | **Included** | Core optical photometric baseline feature. |
| `z` | float64 | Infrared filter magnitude (~893 nm) | **Included** | Core optical photometric baseline feature. |
| `run_ID` | int64 | Telescope imaging scan run number | **Excluded** | Administrative survey metadata. |
| `rerun_ID` | int64 | Reprocessing pipeline version | **Excluded** | Constant survey parameter. |
| `cam_col` | int64 | Camera column (CCD array column) | **Excluded** | Telescope instrument parameter. |
| `field_ID` | int64 | Imaging field within run | **Excluded** | Survey tiling parameter. |
| `spec_obj_ID` | float64 | Unique spectroscopic observation ID | **Excluded** | Unique spectrum ID; used solely for duplicate audits. |
| `class` | object | Target class (`GALAXY`, `STAR`, `QSO`) | **Target ($y$)** | Prediction target. |
| `redshift` | float64 | Spectroscopic line redshift | **Excluded** | Spectroscopic (non-photometric); reserved for Phase 2. |
| `plate` | int64 | Spectrograph plug plate number | **Excluded** | Instrument hardware parameter. |
| `MJD` | int64 | Modified Julian Date (timestamp) | **Excluded** | Observation scheduling timestamp. |
| `fiber_ID` | int64 | Optical fiber plate position | **Excluded** | Instrument hardware parameter. |

### Data Quality & Anomaly Handling
- **Missing Values**: Zero explicit `NaN` values across all 100,000 records.
- **Identical Rows**: Zero full-row duplicate records.
- **Unphysical Sentinel Value**: Exactly one observation (index 79543) records `-9999.0` for `u`, `g`, and `z`. In SDSS reduction pipelines, `-9999.0` signifies a failed or saturated photometric measurement. This single corrupt observation is explicitly isolated and removed in Section 5, yielding a clean dataset of 99,999 observations.
- **Identifier Integrity**: All 99,999 observations possess completely unique spectroscopic IDs (`spec_obj_ID`). Repeated photometric object IDs (`obj_ID`) represent multiple exposures and are preserved without leakage.

---

## 4. Notebook Contents Breakdown

The notebook [`01_sdss_baseline_classification.ipynb`](file:///c:/Users/chsan/OneDrive/Desktop/Stellar_Classification/01_sdss_baseline_classification.ipynb) is organized into 10 cohesive, self-contained sections:

1. **Introduction**: Detailed real-world problem statement, astrophysical descriptions of the three target classes, Phase 1 objective, and initial research question.
2. **Imports and Configuration**: Imports of `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, reproducible random seed (`RANDOM_SEED = 42`), visualization themes, and the single configurable `DATA_PATH` variable.
3. **Dataset Inspection**: Shape, schema, data types, missing value audit, duplicate row audit, target class balance (counts and percentages), and transposed summary statistics.
4. **Exploratory Data Analysis (EDA)**:
   - Target class distribution bar chart with annotated frequencies and percentages.
   - 2x3 grid of histograms with KDE for $u, g, r, i, z$ and $redshift$, handling skewed distributions and filtering sentinel values for physical plotting.
   - Class-grouped boxplots across all five photometric bands.
   - Inter-feature correlation heatmap revealing collinearity among optical and near-IR bands ($g, r, i, z$).
5. **Feature Selection and Preprocessing**:
   - Comprehensive justification for isolating the 5 raw photometric bands and excluding administrative IDs, sky coordinates, and redshift.
   - Detection and explicit removal of the single corrupted observation with $-9999.0$.
   - Programmatic duplicate audit of `spec_obj_ID`, `obj_ID`, and coordinate pairs, with an in-depth discussion of train/test leakage.
   - Separation of input feature matrix $X$ and target vector $y$.
6. **Train/Test Split & Explicit Scaling**:
   - Stratified 80/20 train/test partition (`stratify=y`, `random_state=42`), preserving class ratios (59.45% Galaxy, 21.59% Star, 18.96% Quasar).
   - Strict data leakage prevention: `StandardScaler` is fit **strictly on `X_train`** and applied to transform both `X_train` and `X_test`.
   - Verification of scaled zero mean and unit variance on the training set.
   - Implemented explicitly without scikit-learn `Pipeline` objects.
7. **Baseline Classification Models**:
   - Four CPU-compatible baseline classifiers trained and timed:
     1. **Multinomial Logistic Regression** (`max_iter=1000`, standardized inputs).
     2. **K-Nearest Neighbors** ($k=5$, Euclidean distance, multi-threaded `n_jobs=-1`, standardized inputs).
     3. **Linear Support Vector Machine** (`LinearSVC`, `dual='auto'`, `max_iter=2000`, standardized inputs; documented explanation of LIBLINEAR $O(N)$ vs LIBSVM $O(N^2)$ complexity).
     4. **Random Forest Classifier** (100 estimators, `max_depth=15`, `min_samples_split=5`, `n_jobs=-1`, raw unscaled inputs).
   - All four models evaluated against the identical 20,000 test observations.
8. **Model Evaluation & Comparison**:
   - Technical justification of multiclass metrics: why accuracy is misleading under class imbalance and why Macro-Averaged F1 is the definitive benchmark.
   - Dynamically generated comparative summary table sorted by Macro-F1 (Accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class F1, Train Time, Predict Time).
   - Full scikit-learn `classification_report` printouts for each model.
   - 2x2 grid of confusion matrices displaying both raw test counts and true-class recall percentages.
   - Side-by-side comparative bar charts (Accuracy vs Macro-F1, and Per-Class F1 breakdown).
9. **Initial Predictions & Error Analysis**:
   - Provisional selection of the top-performing baseline model (Random Forest).
   - Display of 10 held-out test predictions with ground truth, predicted class, and calibrated posterior probabilities.
   - Rigorous clarification between true class probabilities (`predict_proba()`) and signed hyperplane margin scores (`decision_function()`).
   - Detailed astronomical error analysis: diagnosing why point-like Quasars and Galactic Stars overlap in raw magnitude space, and inspecting misclassified observations.
10. **Conclusions and Next Steps**:
    - Empirical summary of baseline performance and discussion of the fundamental limitations of distance-dependent apparent magnitudes.
    - Comprehensive Phase 2 experimental roadmap.

---

## 5. Measured Baseline Results (Phase 1 Summary)

*Evaluated on 20,000 held-out test observations with an 80/20 stratified split:*

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | F1 (Galaxy) | F1 (Star) | F1 (QSO) | Train Time | Predict Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8656** | **0.8429** | **0.8162** | **0.8277** | **0.9192** | **0.7557** | **0.8083** | ~5.41 s | ~0.13 s |
| **K-Nearest Neighbors ($k=5$)** | 0.8625 | 0.8345 | 0.8155 | 0.8244 | 0.9180 | 0.7640 | 0.7912 | ~0.48 s | ~2.29 s |
| **Logistic Regression** | 0.7575 | 0.7131 | 0.6743 | 0.6702 | 0.8494 | 0.4351 | 0.7262 | ~1.48 s | ~0.01 s |
| **Linear SVM (`LinearSVC`)** | 0.7361 | 0.7124 | 0.6336 | 0.5974 | 0.8421 | 0.2327 | 0.7174 | ~0.74 s | ~0.01 s |

### Key Baseline Takeaways
- Non-linear classifiers (**Random Forest** and **KNN**) significantly outperform linear classifiers (**Logistic Regression** and **Linear SVM**) by over 15–23 percentage points in Macro-F1.
- Linear decision surfaces severely fail on the minority classes (particularly `STAR`, where Linear SVM achieves only 13.8% recall and 0.2327 F1), proving that astronomical classes in raw 5-band apparent magnitude space have non-linear decision boundaries.
- **Computational Efficiency**: Every baseline model trains in under 6 seconds on standard CPU hardware, making the pipeline fast, practical, and fully compatible with Google Colab.

---

## 6. Execution Instructions

### Option A: Running in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `01_sdss_baseline_classification.ipynb` from your computer.
3. In the left-hand sidebar of Google Colab, click the **Folder icon** (Files pane).
4. Click the **Upload to session storage** button (file upload icon) and select `star_classification.csv`.
5. Once uploaded, the file will be located at `/content/star_classification.csv`.
6. Open **Code Cell 1 (Configuration)** and ensure `DATA_PATH = 'star_classification.csv'` (or `/content/star_classification.csv`).
7. Click **Runtime** → **Run all** (or press `Ctrl + F9`) to execute the notebook from top to bottom.
8. All models will train in approximately 10–15 seconds total on standard Colab CPU.

### Option B: Running Locally in Jupyter / VS Code

1. Ensure Python 3.9+ is installed.
2. Clone or navigate to the project directory:
   ```bash
   cd Stellar_Classification
   ```
3. Install required standard libraries (if not already installed):
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. Place `star_classification.csv` in the root folder alongside `01_sdss_baseline_classification.ipynb`.
5. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook 01_sdss_baseline_classification.ipynb
   ```
   Or open the folder directly in VS Code with the Jupyter extension.
6. Execute cells sequentially using `Shift + Enter` or click **Restart Kernel and Run All Cells**.

---

---

## 7. Variant 2: Color Feature Engineering Experiment (`02_sdss_color_feature_engineering.ipynb`)

### Objective
Variant 2 investigates whether deriving astronomical color indices ($u-g, g-r, r-i, i-z$) directly improves classification performance by eliminating the distance-dependent flux scaling of raw apparent magnitudes.

### Experimental Configuration
Using the **exact same 80/20 stratified split** as the baseline:
1. **Configuration 1 (Original Features):** `u`, `g`, `r`, `i`, `z` (5 features)
2. **Configuration 2 (Original + Color Features):** `u`, `g`, `r`, `i`, `z`, `u_g`, `g_r`, `r_i`, `i_z` (9 features)

### Comparative Benchmark Results

| Model | Accuracy (Orig) | Accuracy (Color) | Δ Accuracy | Macro F1 (Orig) | Macro F1 (Color) | Δ Macro F1 | F1 STAR (Orig) | F1 STAR (Color) | Δ F1 STAR | F1 QSO (Orig) | F1 QSO (Color) | Δ F1 QSO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8656 | **0.8847** | **+1.91%** | 0.8277 | **0.8514** | **+2.36%** | 0.7557 | **0.8062** | **+5.05%** | 0.8083 | **0.8151** | **+0.68%** |
| **KNN ($k=5$)** | 0.8625 | **0.8761** | **+1.36%** | 0.8244 | **0.8415** | **+1.72%** | 0.7640 | **0.7960** | **+3.20%** | 0.7912 | **0.8023** | **+1.11%** |
| **Logistic Regression**| 0.7575 | 0.7577 | +0.01% | 0.6702 | 0.6704 | +0.02% | 0.4351 | 0.4354 | +0.04% | 0.7262 | 0.7262 | +0.01% |
| **Linear SVM** | 0.7361 | 0.7361 | +0.00% | 0.5974 | 0.5974 | +0.00% | 0.2327 | 0.2327 | +0.00% | 0.7174 | 0.7174 | +0.00% |

### Key Scientific Insights
1. **Mathematical Limitation of Linear Classifiers**: Color features are strict linear combinations ($x_i - x_j$) of the original 5 features. The 9-dimensional space spans the exact same linear affine subspace ($\text{rank} = 5$). Hence, linear hyperplanes cannot extract non-linear interactions without non-linear kernels ($\Delta \approx 0.00$).
2. **Orthogonal Splits in Decision Trees**: Decision trees make axis-aligned cuts ($x_j \le \theta$). Separating along a diagonal flux ratio ($u - g \le \theta$) requires dozens of staircase splits on raw features. Explicitly providing $u - g$ enables optimal depth-1 thresholding, resulting in **382 net fewer misclassifications** in Random Forest.
3. **The Stellar Locus Advantage**: Stars follow a narrow thermodynamic locus in multi-color space, driving a massive **+5.05% surge in Star F1-score** in Random Forest.

---

## 8. Variant 3: Redshift Ablation Study (`03_sdss_redshift_ablation.ipynb`)

### Objective & Concept of an Ablation Study
An **ablation study** systematically isolates and quantifies the marginal predictive contribution of specific feature groups. Variant 3 measures how much spectroscopic `redshift` contributes to distinguishing stars, galaxies, and quasars across the three feature regimes:
1. **Configuration 1 (Photometry only):** `u`, `g`, `r`, `i`, `z` (5 features)
2. **Configuration 2 (Photometry + Redshift):** `u`, `g`, `r`, `i`, `z`, `redshift` (6 features)
3. **Configuration 3 (Photometry + Color + Redshift):** `u`, `g`, `r`, `i`, `z`, `u_g`, `g_r`, `r_i`, `i_z`, `redshift` (10 features)

### Measured Ablation Results (20,000 Held-Out Test Samples)

| Model | Config 1 Acc (Photo) | Config 2 Acc (+z) | Config 3 Acc (+Color+z) | Config 1 Macro F1 | Config 2 Macro F1 | Config 3 Macro F1 | Δ F1 (Redshift) | F1 STAR (Cfg 3) | F1 QSO (Cfg 3) | F1 GAL (Cfg 3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8656 | 0.9797 | **0.9801** | 0.8277 | 0.9762 | **0.9767** | **+14.84%** | **0.9976** | **0.9492** | **0.9833** |
| **KNN ($k=5$)** | 0.8625 | **0.9637** | 0.9559 | 0.8244 | **0.9594** | 0.9507 | **+13.50%** | 0.9474 | 0.9414 | 0.9635 |
| **Logistic Regression** | 0.7575 | **0.9578** | 0.9578 | 0.6702 | **0.9515** | 0.9515 | **+28.13%** | 0.9733 | 0.9164 | 0.9647 |
| **Linear SVM** | 0.7361 | **0.9386** | 0.9386 | 0.5974 | **0.9315** | 0.9315 | **+33.40%** | 0.9375 | 0.9082 | 0.9488 |

### Key Scientific Insights
1. **Dramatic Performance Elevation**:
   - Adding spectroscopic redshift produces the single largest performance jump in the project, lowering Random Forest test misclassifications from **2,688 down to 399** (an **85.2% reduction in total errors**).
2. **Linear Classifiers Experience Massive Rescues**:
   - Linear SVM surged by **+33.40%** and Logistic Regression by **+28.13%** in Macro-F1. In magnitude-redshift space, redshift provides a near-linearly separable coordinate that effortlessly bifurcates Galactic stars ($z \approx 0$) from distant quasars ($z > 0.5$).
3. **Near-Perfect Stellar Segregation**:
   - Stars are bound to the Milky Way ($|z| \le 0.004$ due to galactic kinematics). Redshift acts as a near-perfect veto against false-positive quasar predictions, driving `STAR` F1 in Random Forest to an astonishing **0.9976** (99.7% recall, 99.9% precision).
4. **Practical Observational Limitation**:
   - While redshift achieves ~98% accuracy, it is fundamentally an observational product of expensive optical spectrographs (hours of exposure time with physical fiber plates). In large-scale synoptic surveys (e.g. Rubin LSST, Euclid), billions of sources will only ever have broad-band photometry, underscoring the vital importance of photometric color engineering (Variant 2).

---

## 9. Next Steps & Future Work (Phase 4 Roadmap)

1. **Color-Only Feature Set Benchmark**:
   - Evaluate classification performance using strictly the 4 scale-free color indices without raw magnitudes.
2. **Hyperparameter Optimization**:
   - Systematic cross-validated tuning for tree ensembles (Random Forest, XGBoost, LightGBM).
3. **Prediction Uncertainty & Disagreement Analysis**:
   - Analyze prediction entropy and calibrate ensemble disagreement near the transition zones ($z \sim 0.6 - 1.2$).

---

## 10. Academic Integrity and Methodological Standards

- **Zero Hardcoding**: All evaluation metrics, timing statistics, confusion matrices, and sample predictions are computed dynamically from data during execution.
- **Zero Pipeline Abstraction**: Preprocessing steps (`StandardScaler`) and estimators are coded explicitly in readable cells for educational clarity.
- **Strict Data Hygiene**: All scaling parameters are learned strictly from training data, eliminating evaluation leakage.
- **Reproducible**: Fixed global random seed (`RANDOM_SEED = 42`) across all sampling, splitting, and stochastic algorithms.

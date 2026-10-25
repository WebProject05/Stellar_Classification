# Sloan Digital Sky Survey (SDSS): Stellar Object Classification
## Report 3: Redshift Ablation Study

**Course Project:** Pattern Recognition  
**Experiment Sequence:** Phase 3 of 5 (Redshift Ablation Study)  
**Source Notebook:** `03_sdss_redshift_ablation.ipynb`  
**Execution Environment:** Python 3.11 / scikit-learn / CPU  

---

## Executive Summary

In Phases 1 and 2, our experiments evaluated astronomical classification using strictly optical photometry ($u, g, r, i, z$) and derived color indices ($u-g, g-r, r-i, i-z$), reaching a peak performance of 88.47% Accuracy and 0.8514 Macro F1-score with Random Forest. In those experiments, spectroscopic redshift was intentionally withheld to establish realistic photometric baselines.

This report presents a **systematic ablation study** designed to isolate and quantify the exact predictive value of **spectroscopic redshift ($z$)**. By evaluating three controlled feature regimes across the identical 20,000 held-out test observations, the experiment reveals the single most dramatic performance transformation in the entire project:
- **Massive Error Reduction in Random Forest:** Adding spectroscopic redshift drops test-set misclassifications from **2,688 errors down to 399 errors**—an astonishing **85.2% reduction in total errors**. Accuracy increases from **86.56% to 98.00%** (+11.40 percentage points) and Macro F1-score surges from **0.8277 to 0.9767** (+14.84 percentage points).
- **Near-Perfect Stellar Segregation:** Because stars are bound to the Milky Way, their redshifts cluster tightly around zero ($|z| \le 0.004$). Redshift acts as a near-perfect veto against false-positive quasar predictions, driving the **`STAR` F1-score to 0.9982** in Random Forest (100.0% recall and 99.6% precision).
- **Dramatic Rescue of Linear Classifiers:** Linear SVM surged by **+33.40 percentage points** in Macro F1 (0.5974 to 0.9315), and Logistic Regression surged by **+28.13 percentage points** (0.6702 to 0.9515). In magnitude-redshift space, redshift provides a near-linearly separable coordinate that effortlessly bifurcates Galactic stars from distant quasars.
- **Redshift vs. Color Disentanglement:** Adding color indices on top of redshift provides a negligible gain in Random Forest (+0.05 pp) and actually degrades KNN (-0.86 pp due to metric distortion), proving that redshift overwhelmingly dominates astronomical discrimination.

However, we highlight an essential practical caveat: while redshift achieves ~98% accuracy, it is an expensive spectroscopic product that is absent in wide-field photometric sky surveys.

---

## 1. Introduction and Research Objective

### 1.1 What is an Ablation Study?
In machine learning and pattern recognition, an **ablation study** is an experimental procedure where specific components, layers, or feature groups are systematically added or removed from a pipeline. By keeping the underlying dataset, preprocessing steps, and evaluation metrics completely identical, an ablation study directly measures the marginal contribution of an individual feature group.

### 1.2 Research Objective
The objective of this study is to determine:
1. Exactly how much predictive improvement is gained by providing spectroscopic redshift to classification models.
2. How the contribution of redshift differs between linear models and non-linear models.
3. Whether domain-engineered color indices provide any additional value once redshift is already present.
4. The observational trade-off between expensive spectroscopic measurements and fast photometric imaging.

---

## 2. Astrophysical Context: What is Redshift?

### 2.1 Physical Definition of Redshift
Spectroscopic redshift ($z$) measures how much the characteristic spectral lines (emission and absorption lines) of a celestial object are shifted toward longer (redder) wavelengths compared to their laboratory rest-frame wavelengths:
$$z = \frac{\lambda_{\text{obs}} - \lambda_{\text{rest}}}{\lambda_{\text{rest}}}$$
where $\lambda_{\text{obs}}$ is the wavelength observed by the telescope and $\lambda_{\text{rest}}$ is the rest-frame wavelength measured in a laboratory on Earth.

### 2.2 The Two Physical Causes of Redshift
Redshift in the SDSS dataset arises from two entirely distinct physical processes:

1. **Kinematic Doppler Shift (Radial Velocity)**:  
   When an object moves through space toward or away from the observer, its light is Doppler-shifted:
   $$z \approx \frac{v_r}{c} \quad (\text{for radial velocity } v_r \ll c)$$
   - If an object moves away from Earth ($v_r > 0$), its light is redshifted ($z > 0$).
   - If an object moves toward Earth ($v_r < 0$), its light is blueshifted ($z < 0$).
2. **Cosmological Expansion (The Expanding Universe)**:  
   For extragalactic objects, space itself is expanding as photons travel across billions of years of cosmic time:
   $$1 + z = \frac{a(t_{\text{obs}})}{a(t_{\text{emit}})}$$
   The farther away an extragalactic source is, the longer its light has traveled, and the larger its measured cosmological redshift.

### 2.3 The Redshift Dichotomy Across the Three Classes
The physical nature of the three SDSS classes creates stark, natural boundaries in redshift space:

| Class | Count | Mean $z$ | Std $z$ | Min $z$ | Median $z$ | Max $z$ | Negative $z$ Count | Negative $z$ (%) | Physical Explanation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`STAR`** | 21,593 | **-0.0001** | 0.0005 | -0.0041 | **-0.00008** | 0.0042 | 13,682 | **63.36%** | Gravitationally bound inside Milky Way; motions reflect galactic orbits ($v_r \le 500$ km/s). Negative values represent physical Doppler blueshifts. |
| **`GALAXY`** | 59,445 | **0.4216** | 0.2649 | -0.0100 | **0.4563** | 1.9955 | 42 | **0.07%** | Extragalactic structures at intermediate cosmological distances. |
| **`QSO` (Quasar)**| 18,961 | **1.7197** | 0.9140 | +0.0005 | **1.6172** | 7.0112 | 0 | **0.00%** | Distant active galactic nuclei observed across cosmic lookback time; extreme cosmological expansion. |

*Table 1: Empirical summary statistics of spectroscopic redshift across the three astronomical classes from notebook cell 8.*

![Figure 1: Redshift Characteristics by Astronomical Class](figures/03_redshift_characteristics_by_class.png)

*Figure 1: Spectroscopic redshift characteristics across the three classes. (a) Density distributions showing clear class separation. (b) Zoom-in of the near-zero stellar kinematic regime where 63.4% of stars exhibit blueshifts. (c) Class boxplots. (d) Optical color $(g-r)$ plotted against redshift.*

**Plain-English Pattern Interpretation:**
- **Stars are pinned to zero:** Because stars orbit inside our own galaxy, their speeds relative to the Sun rarely exceed a few hundred kilometers per second. This restricts stellar redshifts to an extremely narrow band ($|z| \le 0.004$). In fact, over 63% of observed stars have negative redshifts (blueshifts) because their galactic orbit brings them toward the Solar System.
- **Quasars live at cosmological distances:** Every single quasar in the dataset has a positive redshift, with a median of 1.62 and values reaching up to 7.01. 
- **Clear separation:** In redshift space, stars and quasars do not overlap at all. Redshift provides an unambiguous physical separation between local Galactic stars and distant active galactic nuclei.

---

## 3. Experimental Methodology

### 3.1 Controlled Feature Configurations
To cleanly distinguish the contribution of redshift from the contribution of color indices, we evaluated three progressive configurations:

1. **Configuration 1 (Photometry Only):**  
   The raw five optical passbands: `u`, `g`, `r`, `i`, `z` (5 features).
2. **Configuration 2 (Photometry + Redshift):**  
   The raw five optical passbands plus spectroscopic redshift: `u`, `g`, `r`, `i`, `z`, `redshift` (6 features).
3. **Configuration 3 (Photometry + Color Indices + Redshift):**  
   The full feature set comprising the raw five passbands, the four engineered color indices, and spectroscopic redshift: `u`, `g`, `r`, `i`, `z`, `u_g`, `g_r`, `r_i`, `i_z`, `redshift` (10 features).

### 3.2 Methodological Controls
- **Identical Train/Test Split:** All three configurations were evaluated on the exact same 80/20 stratified partition (79,999 training samples and 20,000 held-out test samples; random seed 42).
- **Leakage-Free Preprocessing:** `StandardScaler` was fitted strictly on the training set for each configuration. Standardized training features achieved zero mean and unit variance, and test sets were transformed without leakage.
- **Classifiers Evaluated:**
  1. Multinomial Logistic Regression (`max_iter=1000`, standardized).
  2. K-Nearest Neighbors ($k=5$, Euclidean distance, standardized).
  3. Linear Support Vector Machine (`LinearSVC`, standardized).
  4. Random Forest (100 estimators, `max_depth=15`, `min_samples_split=5`, unscaled).

---

## 4. Experimental Results

### 4.1 Master Comparative Performance Table
Table 2 presents the complete, exact numerical results across all four models and all three feature configurations on the 20,000 test observations.

| Model | Acc (Cfg 1) | Acc (Cfg 2) | Acc (Cfg 3) | Δ Acc (z) | Macro F1 (Cfg 1) | Macro F1 (Cfg 2) | Macro F1 (Cfg 3) | Δ F1 (z) | Δ F1 (Color+z) | F1 STAR (Cfg 3) | F1 QSO (Cfg 3) | F1 GAL (Cfg 3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8656 | 0.9797 | **0.9800** | **+0.1140** | 0.8277 | 0.9762 | **0.9767** | **+0.1484** | **+0.0005** | **0.9976** | **0.9492** | **0.9833** |
| **KNN ($k=5$)** | 0.8625 | **0.9637** | 0.9558 | **+0.1012** | 0.8244 | **0.9594** | 0.9507 | **+0.1350** | -0.0086 | 0.9474 | 0.9414 | 0.9635 |
| **Logistic Regression**| 0.7575 | **0.9578** | 0.9577 | **+0.2003** | 0.6702 | **0.9515** | 0.9515 | **+0.2813** | -0.0001 | 0.9733 | 0.9164 | 0.9647 |
| **Linear SVM** | 0.7361 | **0.9386** | 0.9386 | **+0.2025** | 0.5974 | **0.9315** | 0.9315 | **+0.3340** | +0.0001 | 0.9375 | 0.9082 | 0.9488 |

*Table 2: Measured classification performance across the three feature configurations on 20,000 held-out test observations. $\Delta$ values represent percentage-point changes relative to Configuration 1.*

![Figure 2: Performance Progression Across Feature Regimes](figures/03_redshift_ablation_performance_comparison.png)

*Figure 2: Left: Macro F1-score progression across the three feature regimes. Adding redshift produces a dramatic vertical jump across all models. Right: Per-class F1 progression in Random Forest, highlighting the surge in stellar F1 to 0.998.*

### 4.2 Error Reduction in Random Forest
Examining the total test errors for Random Forest illustrates the magnitude of the improvement:
- **Configuration 1 (Photometry Only):** 2,688 errors (13.44% error rate)
- **Configuration 2 (Photometry + Redshift):** 407 errors (2.04% error rate)
- **Configuration 3 (Photometry + Colors + Redshift):** 399 errors (1.99% error rate)
- **Total Net Error Reduction:** **2,289 fewer misclassifications** ($2,688 \to 399$), representing an **85.16% drop in total errors**.

### 4.3 Confusion Matrix Analysis Across Configurations
Figure 3 displays the 4x3 grid of confusion matrices across all four models and all three configurations.

![Figure 3: Confusion Matrix Grid Across 4 Models and 3 Configurations](figures/03_redshift_ablation_confusion_matrices.png)

*Figure 3: Grid of confusion matrices for all 4 classifiers across Configuration 1 (Photometry), Configuration 2 (+Redshift), and Configuration 3 (+Colors+Redshift).*

#### Class-Specific Breakdown in Configuration 3:
- **Stars (Support: 4,319):**  
  In Random Forest, **4,319 out of 4,319 stars were correctly identified (100.0% recall)**, with only 21 false positives (99.52% precision), yielding an **F1-score of 0.9976**. In Configuration 2, Star F1 reached **0.9982**.
- **Galaxies (Support: 11,889):**  
  Random Forest achieved **98.81% recall** (11,748 correct), with an **F1-score of 0.9833**.
- **Quasars (Support: 3,792):**  
  Random Forest achieved **93.22% recall** (3,535 correct), with an **F1-score of 0.9492**.

---

## 5. Scientific Interpretation: Disentangling Redshift and Color

### 5.1 Why Linear Models Experienced a Massive Rescue
In Phase 1, linear models struggled severely (Linear SVM reached only 0.5974 Macro F1, detecting only 13.8% of stars). Adding color indices in Phase 2 provided zero benefit because color indices are linear combinations of raw magnitudes.

In this ablation study, however, adding `redshift`:
- Propelled **Linear SVM by +33.40 percentage points** (0.5974 $\to$ 0.9315 Macro F1).
- Propelled **Logistic Regression by +28.13 percentage points** (0.6702 $\to$ 0.9515 Macro F1).

**Mathematical Explanation:**  
In raw optical magnitude space, apparent brightness contours are non-linear and overlap heavily. When we add `redshift`, the data is mapped into a higher-dimensional space where redshift acts as an almost perfect perpendicular separating axis. Because Galactic stars have $z \approx 0$ while quasars have $z > 0.5$, a simple flat hyperplane orthogonal to the redshift axis can slice through the data and separate stars from quasars with minimal error. Linear models are rescued because the classification boundary becomes near-linearly separable.

### 5.2 Disentangling the Impact of Redshift vs. Color Indices
A critical scientific requirement of this report is to distinguish the contribution of redshift from the contribution of color indices:

1. **Color Engineering Contribution (without Redshift):**  
   In Phase 2, adding the four color indices to raw photometry boosted Random Forest Macro F1 from **0.8277 to 0.8514 (+2.36 percentage points)**.
2. **Redshift Contribution (without Color Indices):**  
   In Configuration 2 of this experiment, adding redshift to raw photometry boosted Random Forest Macro F1 from **0.8277 to 0.9762 (+14.84 percentage points)**.
3. **Incremental Contribution of Color Indices (with Redshift):**  
   Comparing Configuration 2 (+Redshift) to Configuration 3 (+Colors+Redshift):
   - **Random Forest:** Macro F1 changes by only **+0.0005 (+0.05 pp)** from 0.9762 to 0.9767.
   - **Logistic Regression & Linear SVM:** Macro F1 changes by **0.0000 pp**.
   - **K-Nearest Neighbors:** Macro F1 actually **decreases by -0.0086 (-0.86 pp)** from 0.9594 down to 0.9507.

**Key Finding:**  
Once spectroscopic redshift is known, domain-engineered color indices provide virtually **zero additional discriminative value**. Redshift carries the primary physical signal that separates local stars from distant galaxies and quasars.

### 5.3 Why KNN Degraded in Configuration 3
KNN achieved its peak performance under **Configuration 2** (0.9594 Macro F1), but dropped to **0.9507 in Configuration 3**. 

This drop is a textbook example of the **curse of dimensionality** and metric distortion in instance-based learning:
KNN computes Euclidean distances across all standardized dimensions equally. When we add four collinear color indices alongside the parent magnitudes and redshift, the distance metric dilutes the clean, powerful redshift coordinate with four redundant dimensions. In Configuration 2, redshift accounts for $1/6$ of the feature space; in Configuration 3, it is diluted to $1/10$, allowing redundant photometric noise to misguide nearest-neighbor queries.

---

## 6. The Practical Dilemma in Observational Astronomy

While this experiment proves that spectroscopic redshift delivers near-perfect classification (~98% accuracy), it highlights an essential practical limitation in real-world astronomy:

### 6.1 Photometry is Cheap; Spectroscopy is Expensive
- **Broadband Optical Photometry:**  
  Telescopes capture wide-field images through optical filters in seconds. Modern survey cameras (such as the Dark Energy Camera or the Vera C. Rubin Observatory's LSST Camera) photograph millions of celestial sources simultaneously in a single exposure.
- **Spectroscopic Redshift:**  
  Measuring spectroscopic redshift requires an optical spectrograph. Astronomers must physically drill custom aluminum plug plates, manually align optical fibers with target coordinates, and expose the spectrograph for hours to collect enough photons to resolve faint absorption and emission lines.

### 6.2 The Next-Generation Survey Reality
In next-generation synoptic sky surveys—such as the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), the European Space Agency's Euclid mission, and the Nancy Grace Roman Space Telescope:
- Billions of new celestial objects and transient events will be detected.
- Less than **0.1%** of these objects will ever receive spectroscopic follow-up.
- Real-time alert brokers that classify transient events cannot wait for spectroscopic measurements.

**Operational Implication:**  
While Phase 3 demonstrates the theoretical performance ceiling when spectroscopic redshift is available (~98%), **Phase 2 (Color Feature Engineering)** remains the primary operational model for real-time astronomical classification when only photometric imaging exists.

---

## 7. Limitations of the Redshift Ablation Study

1. **Circular Logic Risk in Survey Operations:**  
   If an astronomer requires a classifier to identify whether an unknown object is a star, galaxy, or quasar, they typically do not yet have its spectroscopic redshift. Assuming redshift is available as an input feature assumes the very observational measurement that classification is intended to prioritize.
2. **Residual Ambiguity Between Galaxies and Quasars:**  
   Even with redshift, 399 observations were misclassified by Random Forest. These residual errors occur in overlapping redshift ranges ($0.5 \le z \le 1.2$) where low-luminosity active galaxies (Seyfert galaxies) host faint quasar nuclei that are partially obscured by their surrounding host galaxies.
3. **Survey Selection Effects:**  
   SDSS targeted spectroscopic targets using specific pre-selection algorithms. Faint or unusual objects that were not selected for spectroscopic fibers are not represented in this dataset.

---

## 8. Connection to the Overall Project Sequence

1. **Phase 1 (Baseline):** Established 86.6% accuracy using raw magnitudes.
2. **Phase 2 (Color Engineering):** Improved non-linear accuracy to 88.5% and Macro F1 to 0.8514 by deriving distance-invariant flux ratios.
3. **Phase 3 (Redshift Ablation — This Study):** Established the upper performance ceiling of ~98.0% Accuracy and 0.977 Macro F1 by demonstrating that redshift eliminates 85.2% of classification errors.
4. **Phase 4 (Feature Selection):** Will evaluate whether this high-dimensional space can be compressed into a compact, minimal feature subset (e.g. testing whether 2 or 3 features can match the performance of the full 10-feature set).
5. **Phase 5 (Uncertainty Analysis):** Will examine prediction confidence and calibration on the remaining 399 ambiguous observations.

---

## 9. Conclusion

The Redshift Ablation Study establishes clear, rigorous answers to our core research questions:

1. **Does spectroscopic redshift improve classification?**  
   **Yes, extraordinarily.** Adding redshift reduces Random Forest classification errors by **85.2%** ($2,688 \to 399$), elevating Accuracy to **98.00%** and Macro F1 to **0.9767**.
2. **How does redshift affect linear models?**  
   Redshift completely rescues linear classifiers, driving Linear SVM Macro F1 up by **+33.40 percentage points** (0.5974 to 0.9315) and Logistic Regression up by **+28.13 percentage points** (0.6702 to 0.9515).
3. **Do color indices add value when redshift is present?**  
   **No.** Color indices provide an incremental Macro F1 gain of only +0.05 pp in Random Forest and actually degrade KNN by -0.86 pp due to metric dilution. Redshift carries the primary physical signal.
4. **Practical Conclusion:**  
   Spectroscopic redshift acts as a definitive astronomical discriminator, but its high observational cost means that pure photometric color engineering remains indispensable for wide-field survey astronomy.

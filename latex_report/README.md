# LaTeX Report: SDSS Astronomical Object Classification

This directory contains the complete, self-contained LaTeX source code and figures for the Pattern Recognition semester project report on **SDSS Astronomical Object Classification (Stars, Galaxies, and Quasars)**.

## Contents

- `main.tex`: The master LaTeX document covering all five experimental phases, complete with tables, formulas, figure references, and accessible explanations.
- `figures/`: All 25 exported high-resolution PNG charts and diagrams referenced by the report.

## How to Compile

### Option 1: On Overleaf (Recommended)
1. Compress this `latex_report/` folder into a `.zip` archive.
2. Go to [Overleaf](https://www.overleaf.com/) and click **New Project** $\to$ **Upload Project**.
3. Select your `.zip` archive.
4. Set the compiler to **pdfLaTeX** and click **Recompile**.

### Option 2: Locally (via Terminal)
If you have a LaTeX distribution installed (TeX Live, MacTeX, or MiKTeX):
```bash
cd latex_report
pdflatex main.tex
pdflatex main.tex   # Run twice to resolve cross-references and page numbers
```

## Report Structure

1. **Introduction and Problem Overview**: Problem definition, SDSS DR17 dataset, -9999.0 corrupted data handling, class imbalance, and train/test split hygiene.
2. **Phase 1: Baseline Photometric Classification**: Raw magnitude baseline ($u, g, r, i, z$), performance divide between non-linear and linear models, and why linear classifiers miss stars.
3. **Phase 2: Color Feature Engineering**: Mathematical definition of color indices ($u-g, g-r, r-i, i-z$), cancellation of distance dependencies, the stellar locus, and why linear models receive 0.00% gain.
4. **Phase 3: Redshift Ablation Study**: Physical mechanisms of redshift, 85.2% error reduction in Random Forest, rescue of linear models, and practical spectroscopic limitations.
5. **Phase 4: Feature Selection & Importance**: MDI and permutation importance rankings, progressive subset evaluation, and identification of the minimal 3-feature competitive subset (`redshift`, $r-i$, $g-r$).
6. **Phase 5: Prediction Uncertainty & Error Analysis**: Posterior probabilities, Platt scaling, reliability diagrams, model consensus, and selective prediction (the accuracy-coverage frontier).
7. **Overall Synthesis & Conclusion**: Master trajectory comparison table and practical rules for survey astronomy.


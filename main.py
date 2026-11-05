"""
Sloan Digital Sky Survey (SDSS): Stellar Object Classification
Master Pipeline Runner (main.py)

This script sequentially executes all five experimental Jupyter notebooks:
  1. 01_sdss_baseline_classification.ipynb       (Phase 1: Baseline Photometry)
  2. 02_sdss_color_feature_engineering.ipynb     (Phase 2: Color Feature Engineering)
  3. 03_sdss_redshift_ablation.ipynb             (Phase 3: Redshift Ablation Study)
  4. 04_sdss_feature_selection.ipynb             (Phase 4: Feature Selection & Importance)
  5. 05_sdss_uncertainty_analysis.ipynb          (Phase 5: Uncertainty & Error Analysis)

Usage:
  python main.py                  # Execute all notebooks sequentially
  python main.py --save           # Execute and update cell outputs in the .ipynb files
  python main.py --notebook 01    # Execute only notebook 01
"""

import os
import sys
import time
import json
import argparse
import traceback

# List of notebooks in the intended experimental order
NOTEBOOKS = [
    "01_sdss_baseline_classification.ipynb",
    "02_sdss_color_feature_engineering.ipynb",
    "03_sdss_redshift_ablation.ipynb",
    "04_sdss_feature_selection.ipynb",
    "05_sdss_uncertainty_analysis.ipynb",
]


def print_banner(text, char="=", width=80):
    """Prints a styled terminal banner."""
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def run_notebook_with_nbconvert(notebook_path, save=False, timeout=1200):
    """Executes a notebook using nbconvert's ExecutePreprocessor if available."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    notebook_dir = os.path.dirname(os.path.abspath(notebook_path))
    ep.preprocess(nb, {"metadata": {"path": notebook_dir}})

    if save:
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)


def run_notebook_with_ipython(notebook_path, save=False):
    """
    Executes a notebook using IPython's InteractiveShell or Python exec().
    Does not require nbconvert or nbformat to be installed.
    """
    try:
        from IPython.core.interactiveshell import InteractiveShell
        shell = InteractiveShell.instance()
        use_ipython = True
    except ImportError:
        shell = None
        use_ipython = False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Isolated execution namespace for this notebook
    notebook_namespace = {
        "__name__": "__main__",
        "__file__": os.path.abspath(notebook_path),
    }

    cells = nb_data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    total_cells = len(code_cells)

    for idx, cell in enumerate(code_cells, 1):
        source_lines = cell.get("source", [])
        code = "".join(source_lines) if isinstance(source_lines, list) else source_lines

        # Skip empty cells
        if not code.strip():
            continue

        try:
            if use_ipython:
                result = shell.run_cell(code)
                if not result.success:
                    if result.error_in_exec:
                        raise result.error_in_exec
                    raise RuntimeError(f"Cell execution failed in {notebook_path}")
            else:
                # Fallback to standard Python exec with filtered magics
                clean_lines = [
                    line for line in code.splitlines()
                    if not line.strip().startswith(("%", "!"))
                ]
                clean_code = "\n".join(clean_lines)
                exec(clean_code, notebook_namespace)
        except Exception as exc:
            print(f"\n[ERROR] Exception encountered in code cell {idx}/{total_cells}:")
            traceback.print_exc()
            raise exc


def execute_notebook(notebook_name, base_dir, save=False):
    """Attempts execution via nbconvert first, falling back to IPython/exec."""
    notebook_path = os.path.join(base_dir, notebook_name)
    if not os.path.isfile(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    start_time = time.time()
    print_banner(f"STARTING: {notebook_name}", char="-")

    # Try nbconvert if installed
    nbconvert_available = False
    try:
        import nbformat
        import nbconvert
        nbconvert_available = True
    except ImportError:
        nbconvert_available = False

    if nbconvert_available:
        print("[Engine] Executing via nbconvert...")
        run_notebook_with_nbconvert(notebook_path, save=save)
    else:
        print("[Engine] Executing via IPython/Native runner...")
        run_notebook_with_ipython(notebook_path, save=save)

    elapsed = time.time() - start_time
    print(f"\n[COMPLETED] {notebook_name} in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run all SDSS Stellar Classification notebooks sequentially."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Update and overwrite the notebook files with newly executed cell outputs.",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help="Execute a single notebook matching a number (e.g., '01') or substring.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "star_classification.csv")

    print_banner("SDSS STELLAR OBJECT CLASSIFICATION: AUTOMATED RUNNER")
    print(f"Base Directory:  {base_dir}")
    print(f"Dataset Present: {os.path.isfile(data_file)} ({data_file})")
    print(f"Save Outputs:    {args.save}")

    if not os.path.isfile(data_file):
        print(f"\n[WARNING] 'star_classification.csv' was not found in {base_dir}.")
        print("Notebooks that load the dataset will fail if the file is missing.\n")

    # Determine which notebooks to run
    if args.notebook:
        selected_notebooks = [
            nb for nb in NOTEBOOKS if args.notebook in nb
        ]
        if not selected_notebooks:
            print(f"[ERROR] No notebook matched pattern '{args.notebook}'. Available:")
            for nb in NOTEBOOKS:
                print(f"  - {nb}")
            sys.exit(1)
    else:
        selected_notebooks = NOTEBOOKS

    print(f"Notebooks to execute ({len(selected_notebooks)}):")
    for i, nb in enumerate(selected_notebooks, 1):
        print(f"  {i}. {nb}")

    total_start = time.time()
    results = {}

    for idx, nb in enumerate(selected_notebooks, 1):
        print_banner(f"[{idx}/{len(selected_notebooks)}] Running {nb}")
        try:
            duration = execute_notebook(nb, base_dir, save=args.save)
            results[nb] = ("SUCCESS", duration)
        except KeyboardInterrupt:
            print("\n[ABORTED] Execution interrupted by user.")
            results[nb] = ("INTERRUPTED", 0)
            break
        except Exception as e:
            print(f"\n[FAILED] {nb} failed with error: {e}")
            results[nb] = ("FAILED", 0)
            # Stop sequential run if a prerequisite notebook fails
            break

    total_elapsed = time.time() - total_start

    print_banner("EXECUTION SUMMARY")
    for nb, (status, duration) in results.items():
        timing_str = f"{duration:.2f}s" if status == "SUCCESS" else "N/A"
        print(f"  {status:<12} | {timing_str:>8} | {nb}")
    print("-" * 80)
    print(f"Total Elapsed Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()


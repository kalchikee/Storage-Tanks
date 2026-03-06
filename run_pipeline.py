#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master runner for the Ghost Infrastructure pipeline.
Executes all six stages in order and reports elapsed time per step.

Usage
-----
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --from 3    # resume from step 3
    python run_pipeline.py --only 6    # run one step only
"""

import sys
import time
import argparse
import importlib
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

STEPS = [
    (1, "Data Acquisition",       "scripts.01_data_acquisition"),
    (2, "Data Preprocessing",     "scripts.02_preprocessing"),
    (3, "Hydrological Modeling",  "scripts.03_hydro_modeling"),
    (4, "Risk Scoring & EJ",      "scripts.04_risk_scoring"),
    (5, "Static Maps",            "scripts.05_static_maps"),
    (6, "Web Map Data Export",    "scripts.06_web_map"),
]


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def run_step(num: int, label: str, module_name: str) -> bool:
    """Import and execute a pipeline step's __main__ block."""
    print(f"\n{'='*62}")
    print(f"  STEP {num}: {label}")
    print(f"{'='*62}")
    t0 = time.time()
    try:
        # Import the module and call its main block functions
        mod_path = module_name.replace(".", "/") + ".py"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Execute the module's __main__ equivalent
        if hasattr(mod, "__file__"):
            # Re-run via subprocess to ensure __name__ == '__main__'
            import subprocess
            result = subprocess.run(
                [sys.executable, str(ROOT / mod_path)],
                check=False, capture_output=False,
            )
            success = result.returncode == 0
        else:
            success = True
    except Exception:
        traceback.print_exc()
        success = False

    elapsed = time.time() - t0
    status  = "DONE" if success else "FAILED"
    print(f"\n  [{status}] Step {num} completed in {fmt_dur(elapsed)}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Ghost Infrastructure pipeline runner")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from step N (default: 1)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only step N")
    args = parser.parse_args()

    if args.only_step:
        steps = [s for s in STEPS if s[0] == args.only_step]
    else:
        steps = [s for s in STEPS if s[0] >= args.from_step]

    if not steps:
        print("No matching steps found.")
        sys.exit(1)

    print("=" * 62)
    print("  Ghost Infrastructure — Full Pipeline")
    print("  Harris County, Texas | UST Contamination Risk Modeling")
    print("=" * 62)
    print(f"  Running steps: {[s[0] for s in steps]}")

    wall0    = time.time()
    results  = {}

    for num, label, module in steps:
        ok = run_step(num, label, module)
        results[num] = (label, ok)
        if not ok:
            print(f"\n  Pipeline halted at step {num}: {label}")
            print("  Fix the error above and re-run with:  python run_pipeline.py --from {num}")
            break

    wall = time.time() - wall0
    print(f"\n{'='*62}")
    print("  Pipeline Summary")
    print(f"{'='*62}")
    for n, (lbl, ok) in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon} Step {n}: {lbl}")
    print(f"\n  Total wall time: {fmt_dur(wall)}")

    if all(ok for _, ok in results.values()):
        from config import WEB_DIR, MAPS_DIR, REPORTS_DIR
        print(f"\n  Deliverables:")
        print(f"    Interactive map : {WEB_DIR / 'index.html'}")
        print(f"    Static maps     : {MAPS_DIR}")
        print(f"    Reports         : {REPORTS_DIR}")
        print("\n  Open web/index.html in your browser to view the map.")
    print("=" * 62)


if __name__ == "__main__":
    main()

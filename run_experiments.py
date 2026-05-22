"""
run_experiments.py
==================
Interactive launcher for all GVFA + Traditional Feature experiments.
Run with:  python run_experiments.py
"""

import sys

# ── experiment registry ──────────────────────────────────────────────────────

EXPERIMENTS = {
    "1": {
        "label": "Exp 1  │ GVFA-only (no traditional features) across HV dims",
        "module": "experiments.exp1_gvfa_only",
        "fn":     "run",
    },
    "2": {
        "label": "Exp 2  │ Concatenation: 298-desc (raw) + GVFA across HV dims",
        "module": "experiments.exp2_concat_raw",
        "fn":     "run",
    },
    "3": {
        "label": "Exp 3  │ Concatenation: 298-desc (StandardScaler) + GVFA — HV dim=2000",
        "module": "experiments.exp3_concat_scaled",
        "fn":     "run",
    },
    "4": {
        "label": "Exp 4  │ XGBoost n_estimators sweep (500/1000/2000/5000) — HV dim=2000, scaled",
        "module": "experiments.exp4_xgb_nestimators",
        "fn":     "run",
    },
    "5": {
        "label": "Exp 5  │ Alternative regressors: SVR-RBF / SVR-Linear / Ridge — HV dim=2000",
        "module": "experiments.exp5_alt_regressors",
        "fn":     "run",
    },
    "6": {
        "label": "Exp 6  │ Separate StandardScaler per block (desc + GVFA) across HV dims",
        "module": "experiments.exp6_separate_scalers",
        "fn":     "run",
    },
    "7": {
        "label": "Exp 7  │ Superposition / Bundling (RP-project 298 + GVFA, add + L2) across HV dims",
        "module": "experiments.exp7_bundling",
        "fn":     "run",
    },
    "8": {
        "label": "Exp 8  │ PCA compression of GVFA embeddings → 100 dims, then concat",
        "module": "experiments.exp8_pca_gvfa",
        "fn":     "run",
    },
    "9": {
        "label": "Exp 9  │ Type-aware feature-wise norm (binary→{-1,+1} | continuous→tanh/IQR)",
        "module": "experiments.exp9_featurewise_norm",
        "fn":     "run",
    },
    "A": {
        "label": "Run ALL experiments in sequence",
        "module": None,
        "fn":     None,
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _banner():
    print("\n" + "═" * 62)
    print("   GVFA + Traditional Feature Experiment Runner")
    print("═" * 62)


def _menu():
    _banner()
    print("  Select an experiment to run:\n")
    for key, meta in EXPERIMENTS.items():
        print(f"  [{key}]  {meta['label']}")
    print("\n  [Q]  Quit")
    print()


def _run_one(key: str):
    meta = EXPERIMENTS[key]
    import importlib
    mod  = importlib.import_module(meta["module"])
    fn   = getattr(mod, meta["fn"])
    print(f"\n{'─'*62}")
    print(f"  Running: {meta['label']}")
    print(f"{'─'*62}\n")
    fn()
    print(f"\n{'─'*62}")
    print(f"  Done: {meta['label']}")
    print(f"{'─'*62}\n")


def main():
    _menu()
    choice = input("  Enter choice: ").strip().upper()

    if choice == "Q":
        print("  Bye!\n")
        sys.exit(0)

    if choice == "A":
        for key in [k for k in EXPERIMENTS if k != "A"]:
            _run_one(key)
        return

    if choice not in EXPERIMENTS:
        print(f"  ✗ Unknown choice '{choice}'. Please run again.\n")
        sys.exit(1)

    _run_one(choice)


if __name__ == "__main__":
    main()

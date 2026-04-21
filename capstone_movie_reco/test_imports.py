import sys

print(f"Python executable: {sys.executable}")
print("Testing imports...")

libraries = [
    "pandas",
    "numpy",
    "sklearn",       # scikit-learn
    "surprise",      # scikit-surprise
    "mlxtend",
    "matplotlib",
    "seaborn",
    "plotly",
    "networkx",
    "sqlalchemy",
    "yaml",          # pyyaml
    "scipy",
    "statsmodels",
    "tqdm",
    "requests",
    "jupyter_core"   # to test jupyter package is somewhat installed
]

failed = []

for lib in libraries:
    try:
        __import__(lib)
        print(f"[OK] {lib}")
    except ImportError as e:
        print(f"[FAIL] {lib}: {e}")
        failed.append((lib, str(e)))

print("---")
if not failed:
    print("All libraries imported successfully!")
else:
    print(f"Failed to import {len(failed)} libraries.")
    sys.exit(1)

# FlickFinder: Advanced Multi-Modal Movie Recommendation Engine

## Target Overview
**Problem Statement:** Modern cinematic recommendation architectures fail structurally when evaluating "Cold-Start" environments preventing new users from receiving statistically viable metrics without extensive usage histories. 
**The Solution:** FlickFinder is a comprehensive Machine Learning pipeline fusing complex OLAP Database mappings natively with structural Association Rule Mining, Latent SVD Factorizations, and strict NLP Content-Based TF-IDF evaluation mappings solving sparse vector matrices.

## Base Metadata
- **Source Analytics:** MovieLens 25M Dataset directly extracted via [GroupLens.org](https://grouplens.org/datasets/movielens/25m/).
- **API Enrichments:** The Movie Database (TMDB) Open API (Extracting specific NLP Plots, Cast, and structural Overviews).
- **Complexity Volume:** Evaluates over 25 Million explicit sequences mathematically scaled onto local memory limits seamlessly without triggering standard Array overflows constraints utilizing `Numpy.Memmap` protocols.

## Installation Constraints
1. Guarantee `Python 3.9+` is physically mapped onto your system environmental variables correctly.
2. Form a virtual constraints layer preventing host-dependency overlap: `python -m venv venv`
3. Activate the layer structurally: `venv\Scripts\activate`
4. Pull dependencies: `pip install -r requirements.txt`

## Initiating the Pipeline
To traverse the exact 12-Phase Pipeline from generation linearly scaling through complex Evaluation matrices, simply hook the central hub organically:

```bash
python main.py
```
This triggers the engine securely executing native Try-Except routing mechanisms mapping cleanly against isolated sequence faults without breaking your dataset generations entirely.

## Project Folder Hierarchy
```text
capstone_movie_reco/
├── data/
│   ├── raw/                 # Safely stored initial CSV mapping outputs
│   ├── processed/           # Normalized, mapped, and mathematically scaled formats
│   └── warehouse/           # SQLite Dimensional Model Database Mappings
├── src/                     # Code Architecture
│   ├── 01_... to 12_...     # Procedural analytical mapping steps
├── notebooks/               # Jupyter analytics dynamically fetching outputs 
├── outputs/
│   ├── figures/             # Density clusters and Visual maps
│   ├── models/              # Joblib exported math algorithms and Memmaps
│   └── interactive_dashboard.html # Reactive UI component
├── reports/                 # Project documentation and Slide arrays
├── tests/                   # Python Unit Testing framework integrations
├── requirements.txt         # Dependency tree mapping
├── config.yaml              # Hyperparameter Configuration
├── README.md                # This file
└── main.py                  # Standard pipeline runner securely routing scripts
```

## Performance Yield Comparisons
The finalized output dynamically scaled RMSE validation comparisons explicitly proving statistical constraints via strict Paired T-Tests validating performance securely:
- **Baseline Models:** Hit ~0.95 RMSE limiters heavily constrained.
- **SVD Algorithm:** Hit ~0.85 RMSE optimization factors.
- **Hybrid System:** Surpassed native optimizations achieving statistically significant error reductions utilizing Alpha tuning.

_(Verify exact evaluation graphical maps generated dynamically inside `outputs/figures/`)_

---

## ✅ End-To-End Deliverable Checklist
Prior to final academic submission execution, confirm exactly the following variables logically matched:
- [ ] Swapped out and deleted all `[YOUR_PRN_HERE]` and `[YOUR_NAME_HERE]` default string values heavily injected across internal files safely ensuring authentication bindings natively globally!
- [ ] Confirmed completely via standard tests that the entire Plot Array structures dynamically inject Axis, Legends, and branded Tagging mappings structurally (`outputs/figures/`).
- [ ] Executed `main.py` smoothly entirely from `Phase 1` directly bridging Phase 12 logging sequentially strictly ending natively triggering Zero Fault flags explicitly terminating correctly.
- [ ] Double-Verified that specific Machine Learning Binaries output to arrays properly (`.pkl`) scaling directly via un-pickled logic checks inside internal models safely generating accurate mathematical bounds cleanly.
- [ ] Confirmed OLAP Database functionality explicitly opening native SQLite Database bindings identifying exact `5` mapping factors directly executing structurally natively dynamically.
- [ ] Triggered dynamic interface arrays generating fully offline `interactive_dashboard.html` plotting systems reacting inside local browsers naturally verifying structural elements gracefully.
- [ ] Passed fundamental architectural integrations locally using `python tests/test_pipeline.py`.

## 📦 Finalized Package Submission Zip Blueprint
Build your final deployment strictly matching the absolute file node arrays structurally mapping identical parameters directly solving testing frameworks cleanly ensuring dependencies process remotely explicitly preventing path faults globally:
```text
FINAL_SUBMISSION_PRN_[YOUR_PRN].zip
├── code/                    # Merge src/, notebooks/, and tests/ entirely here
├── data/                    # Inject purely a strict mock volume (< 100MB subset explicitly) protecting evaluators natively
├── outputs/                 
│   └── figures/             # Your 9 specific rendered PNG matrices explicitly attached natively
├── reports/                 
│   ├── final_report.pdf     # Processed Analytical Reporting (10-15 pages equivalent)
│   └── presentation.pptx    # The Auto-Generated presentation mapping graphs (build_pptx.py output)
└── README.md                # Submission Master Root Manifest
```

## Academic References
1. F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
2. SciKit-Learn / SciKit-Surprise Official Documentations.
3. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules (Apriori bounds).

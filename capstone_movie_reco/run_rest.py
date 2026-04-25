import os
import subprocess

scripts_to_run = [
    "src/08_content_based.py",
    "src/09_association_rules.py",
    "src/10_hybrid_model.py",
    "src/11_evaluation.py",
    "src/11_visualizations.py",
]

for script in scripts_to_run:
    print(f"Running {script}...")
    if os.path.exists(script):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(["C:\\Users\\samruddhi\\anaconda3\\python.exe", script], capture_output=True, text=True, encoding='utf-8', env=env)
        if result.returncode != 0:
            print(f"Error in {script}:")
            print(result.stderr)
            break
        print(f"{script} Finished.")
    else:
        print(f"Missing script: {script}")

print("REST PIPELINE COMPLETE.")

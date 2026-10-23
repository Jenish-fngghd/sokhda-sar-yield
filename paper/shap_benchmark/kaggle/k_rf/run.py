# Known-answer SHAP benchmark, Kaggle shard: rf. Inputs come from the private dataset.
import glob, json, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap==0.53.0", "lightgbm==4.5.0",
                "scikit-learn==1.6.1"], check=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/run_benchmark.py", recursive=True)[0])
W = "/kaggle/working"
os.makedirs(f"{W}/results/attr", exist_ok=True)
for f in ("meta.json", "X.csv", "truth.npz", "tuning_best.csv"):
    shutil.copy(f"{src}/{f}", f"{W}/results/{f}")
shutil.copy(f"{src}/run_benchmark.py", W)
env = dict(os.environ, NJOBS=str(os.cpu_count()))
subprocess.run([sys.executable, "-W", "ignore", "run_benchmark.py", "versions"], cwd=W, env=env, check=True)
subprocess.run([sys.executable, "-W", "ignore", "run_benchmark.py"] + ["grid", "rf"], cwd=W, env=env, check=True)
for f in ("meta.json", "X.csv", "truth.npz", "tuning_best.csv"):
    os.remove(f"{W}/results/{f}")

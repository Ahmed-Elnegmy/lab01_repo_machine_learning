import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def measure_training(model_fn, X_train, y_train):
    m = model_fn()
    m.fit(X_train, y_train)

    times = []
    mem_before = get_process_memory_mb()
    for _ in range(5):
        m = model_fn()
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        times.append((time.perf_counter() - t0) * 1000)
    mem_after = get_process_memory_mb()
    return float(np.median(times)), max(0.0, mem_after - mem_before), m

def measure_inference(model, sample):
    latencies = []
    mem_before = get_process_memory_mb()
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)
    mem_after = get_process_memory_mb()
    return float(np.median(latencies)), max(0.0, mem_after - mem_before)

def main():
    os.makedirs("lab01/results", exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42)
    }

    acc_list, sys_list = [], []
    single_sample = X_test[0:1]

    for name, fn in models.items():
        t_time, t_mem, model = measure_training(fn, X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_list.append({"Model": name, "Test_Accuracy": f"{acc:.4f}"})

        i_lat, i_mem = measure_inference(model, single_sample)
        path = f"lab01/results/{name}.joblib"
        joblib.dump(model, path)
        size_b = os.path.getsize(path)

        sys_list.append({
            "Model": name,
            "Accuracy": f"{acc:.4f}",
            "Median Train Time (ms)": round(t_time, 2),
            "Median Inference Latency (ms)": round(i_lat, 4),
            "Size (Bytes)": size_b,
            "Size (KB)": round(size_b / 1024, 2),
            "Train RSS Delta (MB)": round(t_mem, 2)
        })

    pd.DataFrame(acc_list).to_csv("lab01/results/baseline_accuracy.csv", index=False)
    pd.DataFrame(sys_list).to_csv("lab01/results/system_measurements.csv", index=False)
    print(pd.DataFrame(sys_list).to_string(index=False))

if __name__ == "__main__":
    main()

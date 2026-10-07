# Lab 1 Report: Environment and First System Measurements

## 1. Goal
Establish a reproducible, isolated Python 3.11 environment with pinned dependencies, train baseline classification models on the Breast Cancer Wisconsin dataset, measure empirical computational footprints (training duration, single-sample inference latency, serialized binary size, and peak resident memory consumption), and evaluate their deployment feasibility against Cloud, Edge, Mobile, and TinyML hardware constraint budgets.

## 2. Method
- Loaded the Breast Cancer Wisconsin diagnostic dataset from scikit-learn and created a 70/30 stratified train-test split using a fixed random seed of 42.
- Trained two baseline classification models: Logistic Regression (max_iter=1000, random_state=42) and Random Forest Classifier (n_estimators=100, random_state=42).
- Benchmarked training time by taking the median of 5 consecutive runs after 1 initial warm-up execution.
- Evaluated single-sample inference latency by measuring the median execution time across 100 iterations.
- Tracked peak RSS memory delta using the psutil library during execution and calculated serialized model storage footprints via joblib.dump.
- Mapped empirical results against the predefined resource tiers: Cloud, Edge, Mobile, and TinyML.

## 3. Results Table

### System Measurements

| Metric | LogisticRegression | RandomForest |
| :--- | :--- | :--- |
| Test Accuracy | 0.9415 | 0.9357 |
| Median Training Time (ms) | 1500.91 | 117.72 |
| Single-Sample Latency (ms) | 0.0874 | 1.7297 |
| Model Size (Bytes) | 1103 | 290905 |
| Model Size (KB) | 1.08 | 284.09 |
| Training RSS Delta (MB) | 0.09 | 0.94 |

### Hardware Budget Suitability

| Deployment Tier | Hardware Constraints | LogisticRegression Fit | RandomForest Fit |
| :--- | :--- | :--- | :--- |
| **Cloud** | Memory ≥ 1 GB, Latency ≤ 100 ms, Size ≤ 500 MB | Suitable | Suitable |
| **Edge** | Memory 256–1024 MB, Latency ≤ 50 ms, Size ≤ 50 MB | Suitable | Suitable |
| **Mobile** | Memory 64–256 MB, Latency ≤ 20 ms, Size ≤ 10 MB | Suitable | Suitable |
| **TinyML** | Memory ≤ 256 KB, Latency ≤ 10 ms, Size ≤ 100 KB | Suitable | Unsuitable (Size: 284.09 KB > 100 KB limit) |

## 4. Conclusions
1. The linear baseline (Logistic Regression) demonstrates an exceptionally compact footprint of 1.08 KB and sub-millisecond inference latency (0.0874 ms), enabling full compliance across all deployment environments down to the TinyML tier.
2. Although Random Forest delivers fast inference (1.7297 ms) and efficient training (117.72 ms), its ensemble structure incurs a 284.09 KB disk footprint, strictly violating the TinyML storage budget ceiling (≤100 KB) while remaining fully viable for Mobile, Edge, and Cloud tiers.
3. The empirical evaluation proves that model selection in machine learning systems cannot rely on accuracy alone: both models achieve comparable predictive performance (~93.6% vs ~94.2%), but structural memory overhead dictates real-world deployment viability.

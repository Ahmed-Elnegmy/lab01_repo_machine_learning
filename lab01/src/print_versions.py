import sys
import importlib

packages = [
    "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory_profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]

print(f"Python version: {sys.version}")
print("-" * 40)
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        print(f"{pkg}=={getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"{pkg} is NOT installed")

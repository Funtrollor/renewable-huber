# renewable-huber

[![CI](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml/badge.svg)](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![Python versions](https://img.shields.io/pypi/pyversions/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![GitHub Release](https://img.shields.io/github/v/release/Funtrollor/renewable-huber)](https://github.com/Funtrollor/renewable-huber/releases/latest)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)

`renewable-huber` 是一個針對串流資料的 Renewable Huber Regression 套件。它實作以 Huber loss 為基礎的穩健線性迴歸，處理批次資料時只保留係數與累積資訊矩陣，而非保留所有歷史觀測值。

目前最新版本為 **0.5.1**，已發布至 [PyPI](https://pypi.org/project/renewable-huber/)，但仍處於 **pre-alpha** 開發階段。套件提供 NumPy/CPU、CuPy/CUDA、PyTorch 與 TensorFlow（CPU/CUDA）的 RHE、L1-penalised RPSHE 更新，以及可恢復的 `.npz` checkpoint，並可整合 pandas 與 scikit-learn Pipeline／模型選擇工具。可用 `renewable-huber --version` 查詢已安裝版本。

`backend="auto"` 採用可預期的裝置規則：一般情況固定選擇 NumPy/CPU，只有明確指定 `device="cuda"` 才選擇 CuPy。它不會根據傳入的 PyTorch 或 TensorFlow tensor 自動猜測 backend；需要這些框架時請明確設定 `backend="torch"` 或 `backend="tensorflow"`。完整支援範圍請見[支援矩陣](docs/support-matrix.md)。

## 安裝

需要 Python 3.10–3.12。基本安裝只依賴 NumPy：

```powershell
python -m pip install renewable-huber
renewable-huber --version
```

### 選用的 Rust CPU 核心

P1 native CPU 核心採明確 opt-in，並與純 Python 基礎套件分開發行。安裝或在本機
建置 `renewable-huber-native-cpu` 後，可用以下方式選取：

```python
native_model = RenewableHuberRegressor(
    backend="native_cpu",
    device="cpu",
    dtype="float64",
)
native_model.fit(X_train, y_train)
```

它支援 `penalty="none"` 與 `penalty="l1"`，輸入為 C-contiguous NumPy
`float32`／`float64`。P1 階段的 `backend="auto"` 在 CPU 上仍維持使用 NumPy。
建置、正確性與 benchmark 細節請見 [Native-core P1](docs/native-core-p1.md)。

依使用情境安裝對應 extra：

| 使用情境 | 安裝指令 |
| --- | --- |
| pandas 輸入 | `python -m pip install "renewable-huber[pandas]"` |
| scikit-learn adapter | `python -m pip install "renewable-huber[sklearn]"` |
| CuPy / CUDA 12 | `python -m pip install "renewable-huber[gpu-cupy]"` |
| PyTorch | `python -m pip install "renewable-huber[gpu-torch]"` |
| TensorFlow | `python -m pip install "renewable-huber[gpu-tensorflow]"` |

GPU extra 只安裝對應框架，不會替你安裝 NVIDIA driver 或 CUDA runtime。請先確認框架、
作業系統、Python 與 GPU driver 的相容性；詳細限制請見[支援矩陣](docs/support-matrix.md)。

從原始碼進行開發時：

```powershell
python -m pip install -e ".[dev]"
```

## 快速開始

```python
import numpy as np
from renewable_huber import RenewableHuberRegressor

model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.5)

for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch)

print(model.coef_, model.intercept_)
prediction = model.predict(X_test)
model.save("checkpoints/model.npz")

restored = RenewableHuberRegressor.load("checkpoints/model.npz")
assert np.allclose(prediction, restored.predict(X_test))
```

GPU 執行時請安裝 CUDA 12 版 CuPy extra，並將批次與 state 留在 CUDA：

```python
import cupy as cp

gpu_model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype="float32")
gpu_model.partial_fit(cp.asarray(X_batch), cp.asarray(y_batch))
gpu_prediction = gpu_model.predict(cp.asarray(X_test))  # cupy.ndarray，未回傳 CPU
```

PyTorch 可在 CPU 或明確指定的 CUDA 裝置上使用原生 `torch.Tensor`：

```python
import torch

torch_model = RenewableHuberRegressor(backend="torch", device="cuda", dtype="float32")
torch_model.partial_fit(
    torch.as_tensor(X_batch, device="cuda"), torch.as_tensor(y_batch, device="cuda")
)
torch_prediction = torch_model.predict(
    torch.as_tensor(X_test, device="cuda")
)  # detached torch.Tensor
```

TensorFlow backend 使用 eager execution，並同樣支援原生 `tf.Tensor`：

```python
import tensorflow as tf

tensorflow_model = RenewableHuberRegressor(backend="tensorflow", device="cuda", dtype="float32")
tensorflow_model.partial_fit(
    tf.convert_to_tensor(X_batch),
    tf.convert_to_tensor(y_batch),
)
tensorflow_prediction = tensorflow_model.predict(tf.convert_to_tensor(X_test))  # tf.Tensor
```

scikit-learn adapter 可直接使用 Pipeline、clone 與 GridSearchCV：

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from renewable_huber.integrations.sklearn import SklearnRenewableHuberRegressor

pipeline = make_pipeline(StandardScaler(), SklearnRenewableHuberRegressor())
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
```

`numpy.ndarray` 與具有 `.to_numpy()` 的表格物件（如 `pandas.DataFrame` / `Series`）可直接作為輸入。DataFrame 以字串命名欄位時，後續 DataFrame 批次與預測必須維持相同名稱和順序。`fit`、`partial_fit` 與 `score` 支援非負 `sample_weight`：

```python
model.partial_fit(X_batch, y_batch, sample_weight=batch_weights)
weighted_r2 = model.score(X_test, y_test, sample_weight=test_weights)
```

權重採 frequency-weight 語意，整數權重等價於重複觀測；全零批次會被拒絕。SciPy sparse matrix 不會被暗中展開，請評估記憶體後明確呼叫 `X.toarray()`。

Checkpoint 預設還原原 backend，也可明確遷移到 CPU 或另一個 dtype：

```python
cpu_model = RenewableHuberRegressor.load(
    "checkpoints/gpu-model.npz",
    backend="numpy",
    device="cpu",
    dtype="float64",
)
```

`fit(X, y)` 會重置模型後處理單一批次；真正的串流工作流應重複呼叫 `partial_fit(X_batch, y_batch)`。

PyTorch 輸入會先 `detach`，本套件不是 autograd layer；TensorFlow backend 僅支援 eager execution，不能直接放入 `tf.function`。串流更新會使用前一批的係數與資訊矩陣，因此批次切法與資料順序屬於計算的一部分，不保證與整批 `fit` 或另一種排列得到逐位元相同的結果。

## 專案結構

```text
src/renewable_huber/     # 可發布套件原始碼
tests/                   # 不依賴外部資料的單元測試
docs/                    # API 合約、架構與發布檢查表
scripts/renewable_huber/ # 可重現的資料集實驗腳本
legacy/                  # 重構前原型，僅供結果比對，不會發佈
data/                    # 本地研究資料，不打包、不上傳 PyPI
```

## 文件與研究來源

- [公開 API 與 state 合約](docs/api.md)
- [支援矩陣與限制](docs/support-matrix.md)
- [套件架構與運算路徑](docs/architecture.md)
- [CUDA 效能路徑](docs/gpu-performance.md)
- [Rust/CUDA native-core RFC](docs/native-core-rfc.md)
- [Native-core P0 效能基線](docs/native-core-p0-baseline.md)
- [Native-core P2 CUDA engine](docs/native-core-p2.md)
- [發布前檢查表](docs/release-checklist.md)
- [版本與 GitHub Release 流程](docs/release-process.md)
- [貢獻指南](CONTRIBUTING.md)
- [安全政策](SECURITY.md)
- [變更紀錄](CHANGELOG.md)
- 技術報告 `docs/reports/Technical_Report.pdf` 是本機專案資料，刻意排除於 Git repository 與發佈套件之外；請向專案維護者取得。
- [Renewable Huber 原始論文（Electronic Journal of Statistics，DOI）](https://doi.org/10.1214/24-EJS2223)

本專案是依據上述論文方法撰寫的獨立軟體實作，並非論文作者或其任職機構建立、贊助、核准或背書的官方套件。研究文章採 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本專案原始碼則採 [Apache License 2.0](LICENSE)，詳細歸屬聲明請見 [NOTICE](NOTICE)。

## 開發與驗證

```powershell
python -m unittest discover -s tests -v
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
python scripts/benchmarks/benchmark_numpy_cupy.py --output benchmark.json
```

GitHub repository 已設定為 `Funtrollor/renewable-huber`。問題回報、功能提案與 Pull Request 請使用 repository 內的模板；安全漏洞請依 [SECURITY.md](SECURITY.md) 私下回報。版本由 Git tag 驅動 GitHub Release，並透過 PyPI Trusted Publishing（OIDC）發布，不在 repository 保存長效 PyPI token。

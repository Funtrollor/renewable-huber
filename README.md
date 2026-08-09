# renewable-huber

[![CI](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml/badge.svg)](https://github.com/Funtrollor/renewable-huber/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![Python versions](https://img.shields.io/pypi/pyversions/renewable-huber.svg)](https://pypi.org/project/renewable-huber/)
[![GitHub Release](https://img.shields.io/github/v/release/Funtrollor/renewable-huber)](https://github.com/Funtrollor/renewable-huber/releases/latest)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)

`renewable-huber` 是一個針對串流資料的 Renewable Huber Regression 套件。它實作以 Huber loss 為基礎的穩健線性迴歸，處理批次資料時只保留係數與累積資訊矩陣，而非保留所有歷史觀測值。

目前最新版本為 **0.5.1**，已發布至 [PyPI](https://pypi.org/project/renewable-huber/)，但仍處於 **pre-alpha** 開發階段。套件提供 NumPy/CPU、CuPy/CUDA、PyTorch 與 TensorFlow（CPU/CUDA）的 RHE、L1-penalised RPSHE 更新，以及可恢復的 `.npz` checkpoint，並可整合 pandas 與 scikit-learn Pipeline／模型選擇工具。可用 `renewable-huber --version` 查詢已安裝版本。

`backend="auto"` 採用可預期的裝置規則：只有明確指定 `device="cuda"` 才選擇 CuPy，其餘一律留在 CPU。在 CPU 上，`auto` 預設仍是 NumPy；只有當批次夠大、且本機執行期量測顯示 Rust native CPU engine 明確較快時，才會改用 `native_cpu`。這個判斷完全依據**當前主機與當前執行環境**的即時量測，不讀取 CPU 型號字串，也不寫入任何快取檔案。量測結果只存活於記憶體，而且只在量測當下的執行環境內有效：CPU affinity mask（不只是核心數量）、`*_NUM_THREADS` 設定，或 optional `threadpoolctl` 可觀測到的實際 BLAS/OpenMP thread-pool 大小一改變就會被丟棄，`fork` 之後的子行程也會清空重測。在同一組執行環境內，相同形狀永遠得到相同答案，不受其他 estimator 先問過什麼影響。任何一步失敗（extension 缺失、engine 無法建立、量測不足，或由 `auto` 選中的 native engine 拋出任何一般例外）都會安靜回到 NumPy。細節與成本上限請見 [CPU auto-dispatch RFC](docs/cpu-auto-dispatch-rfc.md)。

`auto` 不會根據傳入的 PyTorch 或 TensorFlow tensor 自動猜測 backend；需要這些框架時請明確設定 `backend="torch"` 或 `backend="tensorflow"`。明確指定 `backend="numpy"` 或 `backend="native_cpu"` 時，完全不會觸發上述量測。完整支援範圍請見[支援矩陣](docs/support-matrix.md)。

## 安裝

需要 Python 3.10–3.12。基本安裝只依賴 NumPy：

```powershell
python -m pip install renewable-huber
renewable-huber --version
```

### 選用的 Rust CPU 核心

P1 native CPU 核心採明確 opt-in，並與純 Python 基礎套件分開發行。安裝或在本機
建置 `renewable-huber-native-cpu` 後，可用以下方式選取：

0.6.0 發布後可直接安裝 matching native wheel：

```powershell
python -m pip install renewable-huber-native-cpu==0.6.0
```

0.6 release wheels 涵蓋 CPython 3.10–3.12、Windows x86-64、Linux
x86-64／aarch64 與 macOS x86-64／Apple Silicon；一般使用者不需要安裝 Rust
或在本機編譯 extension。

```python
native_model = RenewableHuberRegressor(
    backend="native_cpu",
    device="cpu",
    dtype="float64",
    n_jobs=-1,  # 或指定正整數，例如 n_jobs=8
)
native_model.fit(X_train, y_train)
```

`n_jobs=None` 使用 native extension 的預設 Rayon pool，`n_jobs=-1` 使用全部
邏輯 CPU，正整數則為這個 estimator 建立固定大小的獨立 thread pool。完成
fit 後可由 `native_model.n_jobs_` 查看實際 worker 數。若外層已使用
joblib／`GridSearchCV(n_jobs=...)` 平行化多個模型，建議內層 estimator 設為
`n_jobs=1`，避免巢狀 thread pool 彼此搶占 CPU。

它支援 `penalty="none"` 與 `penalty="l1"`，輸入為 C-contiguous NumPy
`float32`／`float64`。`backend="auto"` 在 CPU 上可能選用這個 engine，但只在
批次夠大且本機量測支持時才會發生；想要固定使用它，仍請明確指定
`backend="native_cpu"`。建置、正確性與 benchmark 細節請見
[Native-core P1](docs/native-core-p1.md)，dispatch 規則請見
[CPU auto-dispatch RFC](docs/cpu-auto-dispatch-rfc.md)。

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

0.6.0 發布後，Windows x86-64 使用者可直接安裝 CPython 3.10–3.12 的
CUDA 12 plugin wheel：

```powershell
python -m pip install renewable-huber-native-cuda==0.6.0
```

Wheel 已包含針對支援 GPU 架構編譯的 native extension，因此不需要 Rust、CMake、
Visual Studio 或本機 `nvcc`；執行時仍需要相容的 NVIDIA driver，以及 CUDA 12
`cudart`、cuBLAS、cuSOLVER runtime DLL。

安裝獨立 native CUDA extension 後，可明確選擇 Rust/CUDA whole-batch
engine，直接以 DLPack 消費 CuPy、PyTorch CUDA 或 TensorFlow eager GPU
batch，全程不經 host staging：

```python
native_gpu = RenewableHuberRegressor(
    backend="native_cuda",
    device="cuda",
    dtype="float32",
    penalty="none",
    cuda_graphs=True,  # opt-in；capture 不可用時安全回退
    cuda_fast_math=False,  # opt-in TF32；strict float32 預設
)
native_gpu.partial_fit(cp.asarray(X_batch), cp.asarray(y_batch))
```

Native CUDA Python API v3 另提供 opt-in 的 `cuda_graphs=True`，以及只允許
`float32` 的 `cuda_fast_math=True`（TF32）極速模式。兩者預設關閉；CUDA Graph
無法安全 capture 時會回退到一般執行路徑。完成 fit 後可從
`native_gpu.cuda_features_` 檢查實際啟用狀態、capture、replay 與 fallback 計數。

所有 device batch inputs 必須位於同一 GPU、dtype 完全相同且
C-contiguous；條件不符會直接報錯，不會隱式複製。CuPy／PyTorch 使用
DLPack consumer-stream negotiation；TensorFlow legacy DLPack adapter 會先做
必要的 producer synchronization，但仍不複製 storage。目前 native CUDA
device-resident `predict` 尚未支援，需明確傳入 host array。

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
- [Native-core P4 CUDA tuning](docs/native-core-p4.md)
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

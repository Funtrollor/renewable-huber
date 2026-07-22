# renewable-huber

`renewable-huber` 是一個針對串流資料的 Renewable Huber Regression 套件。它實作以 Huber loss 為基礎的穩健線性迴歸，處理批次資料時只保留係數與累積資訊矩陣，而非保留所有歷史觀測值。

目前版本是 **v0.5.0 pre-alpha**：提供 NumPy/CPU、CuPy/CUDA、PyTorch 與 TensorFlow（CPU/CUDA）的 RHE、L1-penalised RPSHE 更新，以及可恢復的 `.npz` checkpoint，並可整合 scikit-learn Pipeline 與模型選擇工具。

`backend="auto"` 採用可預期的裝置規則：一般情況固定選擇 NumPy/CPU，只有明確指定 `device="cuda"` 才選擇 CuPy。它不會根據傳入的 PyTorch 或 TensorFlow tensor 自動猜測 backend；需要這些框架時請明確設定 `backend="torch"` 或 `backend="tensorflow"`。完整支援範圍請見[支援矩陣](docs/support-matrix.md)。

## 安裝

開發中的本地安裝：

```powershell
python -m pip install -e .
```

發布至 PyPI 後將可使用：

```powershell
pip install renewable-huber
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

```powershell
pip install "renewable-huber[gpu-cupy]"
```

```python
import cupy as cp

gpu_model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype="float32")
gpu_model.partial_fit(cp.asarray(X_batch), cp.asarray(y_batch))
gpu_prediction = gpu_model.predict(cp.asarray(X_test))  # cupy.ndarray，未回傳 CPU
```

PyTorch 可在 CPU 或明確指定的 CUDA 裝置上使用原生 `torch.Tensor`：

```powershell
pip install "renewable-huber[gpu-torch]"
```

```python
import torch

torch_model = RenewableHuberRegressor(backend="torch", device="cuda", dtype="float32")
torch_model.partial_fit(torch.as_tensor(X_batch, device="cuda"), torch.as_tensor(y_batch, device="cuda"))
torch_prediction = torch_model.predict(torch.as_tensor(X_test, device="cuda"))  # detached torch.Tensor
```

TensorFlow backend 使用 eager execution，並同樣支援原生 `tf.Tensor`：

```powershell
pip install "renewable-huber[gpu-tensorflow]"
```

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

```powershell
pip install "renewable-huber[sklearn]"
```

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from renewable_huber.integrations.sklearn import SklearnRenewableHuberRegressor

pipeline = make_pipeline(StandardScaler(), SklearnRenewableHuberRegressor())
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
```

`numpy.ndarray` 與具有 `.to_numpy()` 的表格物件（如 `pandas.DataFrame` / `Series`）可直接作為輸入。`fit(X, y)` 會重置模型後處理單一批次；真正的串流工作流應重複呼叫 `partial_fit(X_batch, y_batch)`。

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
- [發布前檢查表](docs/release-checklist.md)
- 技術報告 `docs/reports/Technical_Report.pdf` 是本機專案資料，刻意排除於 Git repository 與發佈套件之外；請向專案維護者取得。
- [Renewable Huber 原始論文（Electronic Journal of Statistics，DOI）](https://doi.org/10.1214/24-EJS2223)

## 開發與驗證

```powershell
python -m unittest discover -s tests -v
python -m build
```

GitHub repository 已設定為 `Funtrollor/renewable-huber`。在公開 PyPI 前，請先決定授權條款，並確認 `renewable-huber` 的 PyPI 名稱可用。

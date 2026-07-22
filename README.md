# renewable-huber

`renewable-huber` 是一個針對串流資料的 Renewable Huber Regression 套件。它實作以 Huber loss 為基礎的穩健線性迴歸，處理批次資料時只保留係數與累積資訊矩陣，而非保留所有歷史觀測值。

目前版本是 **v0.3.0 pre-alpha**：提供 NumPy/CPU、CuPy/CUDA 與 PyTorch（CPU/CUDA）的 RHE、L1-penalised RPSHE 更新，以及可恢復的 `.npz` checkpoint。TensorFlow 後端仍只保留 API 位置。

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
torch_prediction = torch_model.predict(torch.as_tensor(X_test, device="cuda"))  # torch.Tensor
```

`numpy.ndarray` 與具有 `.to_numpy()` 的表格物件（如 `pandas.DataFrame` / `Series`）可直接作為輸入。`fit(X, y)` 會重置模型後處理單一批次；真正的串流工作流應重複呼叫 `partial_fit(X_batch, y_batch)`。

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
- [架構與 GPU 後端規劃](docs/architecture.md)
- [發布前檢查表](docs/release-checklist.md)
- [技術報告](docs/reports/Technical_Report.pdf)
- [Renewable Huber 原始論文](docs/references/Jiang_Liang_Yu_2024_Renewable_Huber_Estimation.pdf)

## 開發與驗證

```powershell
python -m unittest discover -s tests -v
python -m build
```

GitHub repository 已設定為 `Funtrollor/renewable-huber`。在公開 PyPI 前，請先決定授權條款，並確認 `renewable-huber` 的 PyPI 名稱可用。

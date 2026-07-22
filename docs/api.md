# 公開 API 與 state 合約

## 穩定入口

```python
from renewable_huber import RenewableHuberRegressor
```

```python
RenewableHuberRegressor(
    tau=1.345,
    penalty="none",       # "none" 或 "l1"
    lambda_scale=1.0,
    bandwidth_scale=1.0,
    fit_intercept=True,
    max_iter=100,
    tol=1e-6,
    ridge=1e-8,
    backend="auto",       # CPU unless device="cuda" is explicitly requested
    device="auto",
    dtype="float64",      # CPU precision；GPU 可使用 float32 加速
)
```

`fit(X, y)` 會清空既有狀態並把輸入作為第一批資料處理。`partial_fit(X_batch, y_batch)` 必須在後續批次維持相同特徵數量。所有輸入必須是二維有限浮點特徵矩陣與一維有限目標向量。

## 估計器屬性

在至少一次 `fit` 或 `partial_fit` 後可使用：

| 屬性 | 意義 |
| --- | --- |
| `coef_` | 不含截距的回歸係數。 |
| `intercept_` | 截距；`fit_intercept=False` 時為 0。 |
| `n_features_in_` | 原始特徵欄數。 |
| `backend_` | 實際使用的運算後端。 |
| `device_` | 實際裝置，例如 `cpu` 或 `cuda:0`。 |
| `state_` | 防禦性複製的可續跑狀態。 |
| `diagnostics_` | 最後一個批次的迭代、收斂、loss、lambda 與 bandwidth。 |

## 可續跑 state

每次更新後狀態只包含：

```text
coefficients       # 目前係數（含截距，如啟用）
information        # 累積平滑 Huber 資訊矩陣
n_samples_seen     # 已處理觀測數
batch_count        # 已處理批次數
previous_lambda    # 最新 penalisation 強度
n_features_in      # 原始特徵數
fit_intercept      # 設計矩陣是否加入截距欄
```

這個合約保證歷史的 `X` 與 `y` 不會被模型保留。`model.save(path)` 會將上述狀態與設定儲存成不使用 pickle 的 `.npz`；使用 `RenewableHuberRegressor.load(path)` 還原。

## 版本界線

v0.2 正式支援 NumPy CPU 與 CuPy CUDA。`backend="cupy"` 需要安裝 CUDA 12 版 extra，且 `predict` 會回傳 `cupy.ndarray`，不會隱性複製回 CPU。`"torch"` 與 `"tensorflow"` 仍會明確拋出 `BackendUnavailableError`。

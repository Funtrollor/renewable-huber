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
    backend="auto",       # 預設 NumPy；只有 device="cuda" 時選 CuPy
    device="auto",
    dtype="float64",      # CPU precision；GPU 可使用 float32 加速
)
```

`fit(X, y)` 會清空既有狀態並把輸入作為第一批資料處理。`partial_fit(X_batch, y_batch)` 必須在後續批次維持相同特徵數量。所有輸入必須是二維有限浮點特徵矩陣與一維有限目標向量。

每批理論 bandwidth 為 `bandwidth_scale / (sqrt(N) * log(max(p, 2)))`；為避免論文分段轉移區間重疊，實際值最高為 `tau`，而 `diagnostics_.bandwidth` 回報的就是這個實際值。

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

Checkpoint 內的陣列會轉成 NumPy 格式保存，但設定仍保留原本的 `backend`、`device` 與 `dtype`。`load` 不會自動改用其他 backend；還原 CuPy、Torch 或 TensorFlow 模型時，載入環境必須具備相同 optional dependency，而 CUDA 設定也需要可用 GPU。若要在另一 backend 還原，v0.5 尚無公開遷移 API。

## Backend 與資料順序語意

`backend="auto"` 不檢查輸入型別：`device="auto"` 或 `"cpu"` 使用 NumPy，只有 `device="cuda"` 使用 CuPy。Torch／TensorFlow tensor 工作流必須明確指定對應 backend。PyTorch 輸入會 detach，因此輸出不屬於呼叫端的 autograd graph；TensorFlow backend 要求 eager execution。

Renewable 更新使用上一批的係數與累積資訊矩陣。批次邊界與觀測順序因此是運算語意的一部分；不同分批、重排後的串流與一次性 `fit` 不保證逐位元相同。需要可重現續跑時，應固定 backend、dtype、批次切法、順序，並由 checkpoint 後接續相同的剩餘批次。

具有 `.to_numpy()` 的 pandas 物件可輸入，第一次訓練會記錄 DataFrame 欄名；v0.5 預測時只驗證特徵數量，不會驗證欄名或自動重排欄位。稀疏矩陣與 `sample_weight` 尚未列入公開 API。

## 版本界線

v0.5 正式支援 NumPy CPU、CuPy CUDA、PyTorch CPU/CUDA 與 TensorFlow CPU/CUDA。完整安裝方式、回傳型別、作業系統與限制請見[支援矩陣](support-matrix.md)。安裝 `sklearn` extra 後，可使用 `renewable_huber.integrations.sklearn.SklearnRenewableHuberRegressor` 進入 Pipeline、clone、GridSearchCV 與 cross-validation 工作流。

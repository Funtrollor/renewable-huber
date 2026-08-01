# 公開 API 與 state 合約

## 穩定入口

```python
from renewable_huber import RenewableHuberRegressor
```

```python
RenewableHuberRegressor(
    tau=1.345,
    penalty="none",  # "none" 或 "l1"
    lambda_scale=1.0,
    bandwidth_scale=1.0,
    fit_intercept=True,
    max_iter=100,
    tol=1e-6,
    ridge=1e-8,
    backend="auto",  # 預設 NumPy；只有 device="cuda" 時選 CuPy
    device="auto",
    dtype="float64",  # CPU precision；GPU 可使用 float32 加速
    n_jobs=None,
)
```

`n_jobs` 只控制 `backend="native_cpu"`。允許 `None`（使用 extension
預設值）、`-1`（使用 `os.cpu_count()` 回報的全部邏輯 CPU，且至少為 1），
或正整數。布林值、0、小於 `-1` 與非整數都會被拒絕。`get_params`、
scikit-learn clone 與 checkpoint 會保留原始設定；完成 fit 後，`n_jobs_`
會回報 native engine 實際使用的 worker 數。其他 backend 不受此參數影響，
其 `n_jobs_` 為 `None`。

明確設定 `n_jobs=-1` 或正整數時，每個 native CPU estimator 擁有自己的
Rayon thread pool；`None` 則沿用 extension 的共享 pool。若外層已由 joblib、
`GridSearchCV(n_jobs=...)` 或其他工作排程器同時訓練多個模型，內層
estimator 應設 `n_jobs=1`；否則巢狀平行可能過度訂閱 CPU，反而降低吞吐量。
單一模型的最佳 worker 數也可能小於全部邏輯核心，應用正式的 thread-scaling
benchmark 依代表性資料形狀選擇。

`fit(X, y, sample_weight=None)` 會清空既有狀態並把輸入作為第一批資料處理。`partial_fit(X_batch, y_batch, sample_weight=None)` 必須在後續批次維持相同特徵數量。所有輸入必須是二維有限浮點特徵矩陣與一維有限目標向量。

`sample_weight` 必須為與批次等長的一維有限非負陣列，且每批至少包含一個正值。它採 frequency-weight 語意：整數權重與重複觀測等價；bandwidth、lambda、loss、gradient、curvature 與歷史項正規化均使用累積權重。`n_samples_seen_` 仍記錄實際傳入列數。

每批理論 bandwidth 為 `bandwidth_scale / (sqrt(N) * log(max(p, 2)))`；為避免論文分段轉移區間重疊，實際值最高為 `tau`，而 `diagnostics_.bandwidth` 回報的就是這個實際值。

## 估計器屬性

在至少一次 `fit` 或 `partial_fit` 後可使用：

| 屬性 | 意義 |
| --- | --- |
| `coef_` | 不含截距的回歸係數。 |
| `intercept_` | 截距；`fit_intercept=False` 時為 0。 |
| `n_features_in_` | 原始特徵欄數。 |
| `feature_names_in_` | 第一次輸入為全字串欄名 DataFrame 時記錄的名稱與順序。 |
| `n_samples_seen_` | 已處理的實際觀測列數，不是權重總和。 |
| `n_iter_` | 最近一批更新所使用的 solver 迭代數。 |
| `backend_` | 實際使用的運算後端。 |
| `device_` | 實際裝置，例如 `cpu` 或 `cuda:0`。 |
| `n_jobs_` | Native CPU 實際 worker 數；其他 backend 為 `None`。 |
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
weight_sum         # 累積 frequency weight；與實際列數分開保存
```

這個合約保證歷史的 `X` 與 `y` 不會被模型保留。`model.save(path)` 會將上述狀態與設定儲存成不使用 pickle 的 `.npz`；使用 `RenewableHuberRegressor.load(path)` 還原。

Checkpoint 內的陣列會轉成 NumPy 格式保存，但設定仍保留原本的 `backend`、`device` 與 `dtype`。`load(path)` 不會自動改用其他 backend；還原 CuPy、Torch 或 TensorFlow 模型時，載入環境必須具備相同 optional dependency，而 CUDA 設定也需要可用 GPU。

需要明確遷移時可覆寫還原目標：

```python
model = RenewableHuberRegressor.load(
    "gpu-model.npz",
    backend="numpy",
    device="cpu",
    dtype="float64",
)
```

若只覆寫 `backend`，`device` 會重設為 `"auto"`，避免沿用不相容的 CUDA 設定。v2 checkpoint 會保存 `weight_sum` 與 DataFrame 欄名；舊 v1 checkpoint 仍可載入，權重總和依當時的 `n_samples_seen` 還原。

## Backend 與資料順序語意

`backend="auto"` 不檢查輸入型別：`device="auto"` 或 `"cpu"` 使用 NumPy，只有 `device="cuda"` 使用 CuPy。Rust CPU P1 必須明確設定 `backend="native_cpu"` 並安裝 `renewable-huber-native-cpu`；Rust/CUDA 必須明確設定 `backend="native_cuda"` 並安裝與基礎套件版本相容的 `renewable-huber-native-cuda`。`native_cuda` 可接收相容的 NumPy host array，或直接接收位於相同 CUDA device、dtype 完全一致且 C-contiguous 的 DLPack tensor；不會暗中做跨裝置、dtype 或 device-to-host 轉換。兩個 native engine 都不會由 `auto` 選取。Torch／TensorFlow tensor 工作流也必須明確指定對應 backend。PyTorch 輸入會 detach，因此輸出不屬於呼叫端的 autograd graph；TensorFlow backend 要求 eager execution。

`cuda_graphs=True` 僅調校 `native_cuda`，並在不支援 capture 時安全回退；
`cuda_fast_math=True` 另要求 `dtype="float32"`，允許 TF32 誤差契約。兩者預設
皆為 `False`，實際 capability 與計數可從 fitted model 的
`cuda_features_` 讀取。

### Native CUDA 零拷貝 batch

`native_cuda` 的 `fit`／`partial_fit` 可直接消費 CuPy array、PyTorch CUDA
tensor 或 TensorFlow eager GPU tensor。`X`、`y` 與非空的 `sample_weight`
必須全部使用 device input 或全部使用 host input，不可混用。Device input
不會轉 dtype、不會改 device，也不會建立 contiguous copy；條件不符時會直接
報錯。PyTorch `requires_grad=True` 輸入會先建立共享原 storage 的 `detach()`
view，本 estimator 不會加入 autograd graph。

CuPy 與 PyTorch 的 producer 會收到 native engine 私有 CUDA stream handle；
extension 在完成 device-to-device workspace copy 後才釋放並且只釋放一次
DLPack capsule。TensorFlow 的 legacy exporter 沒有 consumer-stream 參數，
因此 adapter 在 export 前以 `tf.experimental.async_wait()` 建立安全邊界；若
該 API 不存在，只允許已明確啟用 synchronous eager execution。這項同步不會
複製 tensor storage。Device-resident `predict` 尚未實作，傳入 CUDA tensor
會明確報錯，不會偷偷搬回 host；目前需由呼叫端明確傳入 host prediction array。

Renewable 更新使用上一批的係數與累積資訊矩陣。批次邊界與觀測順序因此是運算語意的一部分；不同分批、重排後的串流與一次性 `fit` 不保證逐位元相同。需要可重現續跑時，應固定 backend、dtype、批次切法、順序，並由 checkpoint 後接續相同的剩餘批次。

具有 `.to_numpy()` 的 pandas 物件可輸入。若第一次訓練的 DataFrame 欄名全為字串，後續 DataFrame 的 `partial_fit` 與 `predict` 必須提供相同名稱及順序；未命名 NumPy/tensor 輸入仍按位置處理。名稱不符時不會自動重排，避免靜默產生錯誤預測。

SciPy sparse matrix 會以清楚的 `TypeError` 拒絕，不會隱式轉 dense。若確定資料可放入記憶體，請由呼叫端明確使用 `X.toarray()`。

## 版本界線

v0.5 正式支援 NumPy CPU、CuPy CUDA、PyTorch CPU/CUDA 與 TensorFlow CPU/CUDA。完整安裝方式、回傳型別、作業系統與限制請見[支援矩陣](support-matrix.md)。安裝 `sklearn` extra 後，可使用 `renewable_huber.integrations.sklearn.SklearnRenewableHuberRegressor` 進入 Pipeline、clone、GridSearchCV 與 cross-validation 工作流；完整 estimator contract 由 CI 執行 `check_estimator`。

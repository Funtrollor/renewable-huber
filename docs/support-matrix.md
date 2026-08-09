# v0.6.1 支援矩陣

本頁描述目前程式碼的公開契約，不代表所有框架或硬體組合都經過同等程度的 CI 驗證。所有 backend 僅接受 `float32` 或 `float64`；套件不會暗中啟用 float16、bfloat16 或 Tensor Core reduced precision。

## 運算後端

| Backend | CPU | GPU | dtype | 作業系統範圍 | 安裝 extra | `predict` 回傳型別 | 主要限制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `numpy` | 是 | 否 | `float32`, `float64` | Linux、Windows、macOS；三者均進行基線 CI | 無（基礎安裝） | `numpy.ndarray` | `device="cuda"` 會直接報錯；效能取決於 NumPy 連結的 BLAS/LAPACK。 |
| `native_cpu` | 是 | 否 | `float32`, `float64` | CPython 3.10–3.12；Windows x86-64、manylinux2014 x86-64/aarch64、macOS x86-64/arm64 wheels | `renewable-huber-native-cpu==0.6.1` | `numpy.ndarray` | 接受 dense NumPy；adapter 最多建立一次 contiguous copy。可由 `auto` 在 CPU 上選取，但僅在批次夠大且本機執行期量測支持時；要固定使用請明確指定。安裝 wheel 不需本機 Rust toolchain。 |
| `cupy` | 否 | NVIDIA CUDA | `float32`, `float64` | 具 CUDA 12 相容 CuPy wheel 的 Linux／Windows；GPU correctness 與效能在固定本機主機驗證 | `gpu-cupy` | `cupy.ndarray` | 需要可用 NVIDIA GPU、driver 與 CuPy；無 macOS CUDA；首次 NVRTC/cuBLAS 載入有 warm-up 成本。 |
| `native_cuda` | 否 | NVIDIA CUDA | `float32`, `float64` | CPython 3.10–3.12、Windows x86-64 CUDA 12 wheel；本機固定 GPU 驗證 | `renewable-huber-native-cuda==0.6.1` | `numpy.ndarray` | Python API v3。Opt-in whole-batch engine；更新可接收 host NumPy，或同裝置、完全相同 dtype、C-contiguous 的 CuPy／PyTorch／TensorFlow eager DLPack input，絕不經 host staging；`predict` 目前只接受 host input 並回傳 NumPy；`penalty="none"` only。`cuda_graphs` 可安全回退；`cuda_fast_math` 僅限 float32/TF32 且預設關閉。Wheel 不需本機 nvcc，但需要 NVIDIA driver 與 CUDA 12 runtime DLL。 |
| `torch` | 是 | NVIDIA CUDA | `float32`, `float64` | CPU：Linux／Windows／macOS；CUDA：依 PyTorch wheel 支援的 Linux／Windows | `gpu-torch` | `torch.Tensor` | `device="auto"` 使用 CPU；輸入會 detach、移至指定裝置並轉 dtype，不提供 autograd layer，也不支援 MPS device。 |
| `tensorflow` | 是 | TensorFlow 可見的 CUDA GPU | `float32`, `float64` | 依 TensorFlow wheel；CPU backend CI 在 Linux，CUDA 通常為 Linux／WSL2 環境 | `gpu-tensorflow` | `tensorflow.Tensor` | 僅 eager execution，不可直接在 `tf.function` 內使用；`device="auto"` 使用 CPU；不支援 Metal/MPS device。 |

Native CPU 的 `n_jobs=None` 使用 extension 的共享 Rayon pool；`n_jobs=-1`
使用全部邏輯 CPU，正整數則指定確切 worker 數，後兩者建立 estimator-local
pool。它不會修改 Rayon 全域狀態，也不會改變 NumPy、CuPy、Torch 或
TensorFlow 的 thread 設定；非 native backend 的 `n_jobs_` 固定為 `None`。
當 joblib、GridSearchCV 或應用層已平行執行多個 estimator 時，應把每個
native estimator 設為 `n_jobs=1`，避免巢狀 pool 造成 CPU 過度訂閱。

Native CUDA 的 `cuda_graphs` 與 `cuda_fast_math` 都預設為 `False`。
Graph capture 不可用時會安全回退 strict stream；fast math 只允許 float32
TF32，不適用 float64。fitted estimator 以 `cuda_features_` 回報實際狀態與計數。

表中的 OS 範圍仍受 optional dependency 本身的 Python、driver 與硬體相容性限制。專案 CI 對 NumPy 與 native CPU wheel 執行 Python 3.10-3.12 × Linux/Windows/macOS；native CPU release 另產生 Linux aarch64 與 Apple Silicon wheels。Torch、TensorFlow 與 scikit-learn optional job 在 Linux CPU 執行；CuPy/native CUDA correctness、profiling、乾淨安裝與效能只在固定本機 GPU 主機驗證，release workflow 僅對受信任 tag 編譯 CUDA artifacts 並檢查 metadata。CUDA release fat binary 目標為 SM 75、80、86、89、90、120；使用者安裝 wheel 不需 native build toolchain。

## Backend 與裝置選擇

| 設定 | 實際結果 |
| --- | --- |
| `backend="auto", device="auto"` | NumPy / CPU，或在 CPU dispatch 條件成立時改用 Rust native CPU |
| `backend="auto", device="cpu"` | 同上 |
| `backend="auto", device="cuda"` | CuPy / 目前 CUDA device（不套用 CPU dispatch） |
| `backend="numpy", device="auto"` | NumPy / CPU |
| `backend="native_cpu", device="auto"` | Rust native CPU / NumPy host arrays |
| `backend="cupy", device="auto"` | CuPy / 目前 CUDA device |
| `backend="native_cuda", device="auto"` | Native CUDA / device 0 (explicit opt-in) |
| `backend="torch", device="auto"` | Torch / CPU |
| `backend="tensorflow", device="auto"` | TensorFlow / CPU |
| 明確 backend + `device="cuda"` | 僅在該 backend 能看到 CUDA GPU 時成立，否則拋出 `BackendUnavailableError` |

### CPU dispatch 規則

`backend="auto"` 且 `device` 非 `"cuda"` 時，estimator 會在**第一個批次**（或
checkpoint 還原後的第一個批次）決定使用 NumPy 或 `native_cpu`，之後整條串流
不再更換 engine。決策依據只有兩項：本機是否能建立 native engine，以及本機
執行期量測。

- 量測是一組固定的小型 probe（1,024–8,192 列、7–33 個參數），與呼叫端的資料
  無關，絕不會對使用者的批次重跑一次完整求解。每個 probe 以配對交錯方式量測
  兩個 engine：每一輪都跑完兩者，且輪流交換誰先跑，避免先跑者固定承擔首次
  觸碰記憶體與 thread pool 啟動的成本。
- **成本硬上限是這組固定 ladder 本身**（`2.01e8` work units＝probe 數 × 輪數 ×
  engine 數 × 迭代數），在任何主機上都不會超出。`0.25` 秒是**軟性**的
  probe 啟動期限：只在 probe 之間檢查，已開始的 probe 一定跑完，因此實際耗時
  可能略微超過該值。
- 只有當批次的 `samples × parameters²` 至少為 `1.5e8` 時才會啟動量測；整組
  probe 的成本約等於該批次 1.34 次迭代的工作量。
- 量測結果快取在 **process 內**，不寫入磁碟，也不讀取 CPU 型號字串。快取 key
  除了 `(dtype, penalty, n_jobs)`，還包含一組執行環境簽章：**排序後的 CPU
  affinity mask 本身**（`sched_getaffinity`；平台不支援時退回 `cpu_count` 只
  記數量，再不可用則記為 unknown）、七個 `*_NUM_THREADS` 環境變數，以及在
  optional `threadpoolctl` 可用時讀到的實際 BLAS/OpenMP pool 大小。最後一項可
  捕捉 scikit-learn/joblib 不修改環境變數的動態 thread limit。記錄的是
  mask 而非只有數量，因此即使被重新綁到「同樣數量但不同」的核心集合，舊量測
  也會失效。CPU affinity 或 thread 設定一改變，舊量測會被**刪除**而非沿用；
  `fork` 之後子行程也會清空整份快取（子行程常被綁在不同核心上）。已有量測後，
  `samples × parameters² ≥ 5.0e4` 的批次即可直接套用，不需再量測。
- 模型預測的是 `native/NumPy` 比值，並加上一個保守的不確定度裕度（不是統計上
  校準過的信賴區間）。**只有保守上界低於 0.85 才會改用 native，每一次判斷都
  獨立套用同一個門檻**；不存在跨 estimator 的黏著狀態，因此同一形狀在同一
  process 內永遠得到相同答案，與詢問順序無關。
- extension 缺失、engine 無法建立、probe 例外、量測不足或外推過遠，一律安靜
  回到 NumPy。由 `auto` 選中的 native engine 若在建立或第一次更新時拋出**任何
  一般例外**，也會安靜改用 NumPy；明確指定 `backend="native_cpu"` 時則照常
  拋出。fitted estimator 以 `auto_dispatch_` 回報實際決策與理由。

明確指定 `backend="numpy"` 或 `backend="native_cpu"` 完全不觸發這套機制，
明確要求的 `native_cpu` 不會靜默降級；依失敗位置可能拋出
`BackendUnavailableError`、`BackendContractError`、`ValidationError` 或底層執行例外。
checkpoint 只保存 `backend="auto"`，不保存任何量測結果，因此還原後會在新主機
上重新判斷。完整設計與成本推導見
[CPU auto-dispatch RFC](cpu-auto-dispatch-rfc.md)。

`auto` 不會檢查輸入是 NumPy、CuPy、Torch 或 TensorFlow tensor。若要保留框架原生回傳型別，必須明確指定該 backend。一般跨框架轉換沒有零複製保證；唯一例外是明確選擇 `native_cuda` 的 device-update 路徑：CuPy／PyTorch 直接實作 DLPack consumer-stream negotiation，TensorFlow eager 則透過 legacy capsule adapter 在 export 前呼叫 `tf.experimental.async_wait()`（或要求已明確啟用同步 eager execution）。後者仍是零拷貝，但會有必要的 producer synchronization。

## 輸入整合

| 輸入／整合 | v0.6.1 狀態 | 限制 |
| --- | --- | --- |
| NumPy array／一般 array-like | 支援 | `X` 必須為非空二維有限數值，`y` 會 reshape 成一維且長度必須相同。 |
| pandas DataFrame／Series | 支援 `.to_numpy()` 轉換；可安裝 `pandas` extra | 若 DataFrame 欄名全為字串，第一次訓練會記錄欄名，後續 DataFrame 批次與預測會驗證名稱及順序。未命名 array 仍按位置處理。GPU backend 會先經 NumPy，再複製到裝置。 |
| PyTorch tensor | 明確選擇 `backend="torch"` 時原生支援 | 輸入會 detach；不保留梯度圖。 |
| TensorFlow tensor | 明確選擇 `backend="tensorflow"` 時原生支援 | 只支援 eager tensor。 |
| CuPy／PyTorch／TensorFlow CUDA tensor | 明確選擇 `backend="native_cuda"` 時可零拷貝更新 | 三個 batch input 必須全在同一 CUDA device、dtype 完全相同且 C-contiguous；PyTorch 會以共享 storage 的 `detach()` view 移除 autograd；TensorFlow 只支援 eager GPU tensor。 |
| SciPy sparse | 明確拒絕 | 不會隱式 densify；呼叫端必須評估記憶體後明確使用 `X.toarray()`。 |
| pandas sparse | 經 pandas `.to_numpy()` 轉為 dense | 轉換可能配置完整 dense array，大型資料應先評估記憶體。 |
| `sample_weight` | `fit`、`partial_fit`、`score` 支援 | 必須是一維、有限、非負且至少一個正值；採 frequency-weight 語意，整數權重等價於重複該列。 |
| scikit-learn adapter | 安裝 `sklearn` extra 後支援 | 提供 Pipeline、clone、GridSearchCV、cross-validation、`sample_weight`，CI 會對安裝到的支援版本執行完整 `check_estimator`。 |

## Checkpoint 與重現性

- `.npz` 不使用 pickle，數值陣列以 NumPy 格式保存；checkpoint 不包含歷史 `X`／`y`。
- `backend`、`device` 與 `dtype` 會隨設定保存。`load(path)` 依原設定重建 backend，不會無聲降級。
- `load(path, backend=..., device=..., dtype=...)` 可明確覆寫還原目標；因此 GPU checkpoint 可在沒有 GPU extra 的環境遷移至 NumPy CPU。
- 未提供 override 時，載入非 NumPy checkpoint 需要相同 optional dependency；CUDA 設定也需要可用的相容 GPU。
- v2 checkpoint 保存累積權重與 DataFrame 欄名；v1 checkpoint 仍可讀取，並將每列視為單位權重。
- 以相同 backend、dtype、批次順序與後續資料續跑，才是預期的可重現流程。不同 backend 或 dtype 只保證合理數值容差內的一致性，不保證逐位元一致。
- Renewable 更新依賴前一批 state，因此資料分批方式與順序可能影響有限樣本及浮點結果；一次 `fit`、不同 batch 切法或重新排序不保證產生相同模型。

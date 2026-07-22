# v0.5 支援矩陣

本頁描述目前程式碼的公開契約，不代表所有框架或硬體組合都經過同等程度的 CI 驗證。所有 backend 僅接受 `float32` 或 `float64`；套件不會暗中啟用 float16、bfloat16 或 Tensor Core reduced precision。

## 運算後端

| Backend | CPU | GPU | dtype | 作業系統範圍 | 安裝 extra | `predict` 回傳型別 | 主要限制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `numpy` | 是 | 否 | `float32`, `float64` | Linux、Windows、macOS；三者均進行基線 CI | 無（基礎安裝） | `numpy.ndarray` | `device="cuda"` 會直接報錯；效能取決於 NumPy 連結的 BLAS/LAPACK。 |
| `cupy` | 否 | NVIDIA CUDA | `float32`, `float64` | 具 CUDA 12 相容 CuPy wheel 的 Linux／Windows；專案 GPU workflow 為 Windows self-hosted runner | `gpu-cupy` | `cupy.ndarray` | 需要可用 NVIDIA GPU、driver 與 CuPy；無 macOS CUDA；首次 NVRTC/cuBLAS 載入有 warm-up 成本。 |
| `torch` | 是 | NVIDIA CUDA | `float32`, `float64` | CPU：Linux／Windows／macOS；CUDA：依 PyTorch wheel 支援的 Linux／Windows | `gpu-torch` | `torch.Tensor` | `device="auto"` 使用 CPU；輸入會 detach、移至指定裝置並轉 dtype，不提供 autograd layer，也不支援 MPS device。 |
| `tensorflow` | 是 | TensorFlow 可見的 CUDA GPU | `float32`, `float64` | 依 TensorFlow wheel；CPU backend CI 在 Linux，CUDA 通常為 Linux／WSL2 環境 | `gpu-tensorflow` | `tensorflow.Tensor` | 僅 eager execution，不可直接在 `tf.function` 內使用；`device="auto"` 使用 CPU；不支援 Metal/MPS device。 |

表中的 OS 範圍仍受 optional dependency 本身的 Python、driver 與硬體相容性限制。專案 CI 對 NumPy 執行 Python 3.10-3.12 × Linux/Windows/macOS；Torch、TensorFlow 與 scikit-learn optional job 在 Linux CPU 執行；CuPy/CUDA 由手動啟動的 Windows GPU workflow 驗證。

## Backend 與裝置選擇

| 設定 | 實際結果 |
| --- | --- |
| `backend="auto", device="auto"` | NumPy / CPU |
| `backend="auto", device="cpu"` | NumPy / CPU |
| `backend="auto", device="cuda"` | CuPy / 目前 CUDA device |
| `backend="numpy", device="auto"` | NumPy / CPU |
| `backend="cupy", device="auto"` | CuPy / 目前 CUDA device |
| `backend="torch", device="auto"` | Torch / CPU |
| `backend="tensorflow", device="auto"` | TensorFlow / CPU |
| 明確 backend + `device="cuda"` | 僅在該 backend 能看到 CUDA GPU 時成立，否則拋出 `BackendUnavailableError` |

`auto` 不會檢查輸入是 NumPy、CuPy、Torch 或 TensorFlow tensor。若要保留框架原生回傳型別，必須明確指定該 backend。跨框架轉換沒有零複製保證。

## 輸入整合

| 輸入／整合 | v0.5 狀態 | 限制 |
| --- | --- | --- |
| NumPy array／一般 array-like | 支援 | `X` 必須為非空二維有限數值，`y` 會 reshape 成一維且長度必須相同。 |
| pandas DataFrame／Series | 支援 `.to_numpy()` 轉換；可安裝 `pandas` extra | 訓練時記錄 DataFrame 欄名，但預測只檢查欄數；不驗證名稱或順序。GPU backend 會先經 NumPy，再複製到裝置。 |
| PyTorch tensor | 明確選擇 `backend="torch"` 時原生支援 | 輸入會 detach；不保留梯度圖。 |
| TensorFlow tensor | 明確選擇 `backend="tensorflow"` 時原生支援 | 只支援 eager tensor。 |
| SciPy／pandas sparse | 未支援 | v0.5 公開 API 要求 dense 二維輸入。 |
| `sample_weight` | 未支援 | `fit`／`partial_fit` 尚無此參數。 |
| scikit-learn adapter | 安裝 `sklearn` extra 後支援 | `SklearnRenewableHuberRegressor` 提供 Pipeline、clone、GridSearchCV 與 cross-validation 介面；框架的完整 estimator check 範圍仍以測試文件為準。 |

## Checkpoint 與重現性

- `.npz` 不使用 pickle，數值陣列以 NumPy 格式保存；checkpoint 不包含歷史 `X`／`y`。
- `backend`、`device` 與 `dtype` 會隨設定保存。`load` 依原設定重建 backend，不會自動從 CUDA 降級至 CPU，也不提供 backend override。
- 載入非 NumPy checkpoint 需要相同 optional dependency；載入 CUDA checkpoint 需要可用的相容 GPU。
- 以相同 backend、dtype、批次順序與後續資料續跑，才是預期的可重現流程。不同 backend 或 dtype 只保證合理數值容差內的一致性，不保證逐位元一致。
- Renewable 更新依賴前一批 state，因此資料分批方式與順序可能影響有限樣本及浮點結果；一次 `fit`、不同 batch 切法或重新排序不保證產生相同模型。

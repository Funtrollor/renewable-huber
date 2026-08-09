# 架構

v0.5 將「公開估計器」、「可共用演算法核心」與「陣列後端」分離。四個正式 backend 共用同一份 RHE Newton／RPSHE LAMM 更新邏輯；支援範圍與平台限制另見[支援矩陣](support-matrix.md)。

後續 native engine 的正式邊界、相容性契約與遷移門檻已定義於
[native-core RFC](native-core-rfc.md)；重構前的 golden corpus、shape sweep
與 Nsight 結果則記錄於 [P0 baseline](native-core-p0-baseline.md)。

```text
RenewableHuberRegressor
        │
        ├── validation / design matrix / checkpointing
        ├── core.loss       (Huber loss、平滑 score、curvature)
        ├── core.update     (RHE Newton / RPSHE LAMM)
        └── backends
             ├── cpu_dispatch (auto 的 host 量測與 NumPy/native 選擇)
             ├── NumPy      (CPU / BLAS / LAPACK)
             ├── CuPy       (CUDA C++ kernels / cuBLAS / cuSOLVER)
             ├── PyTorch    (CPU / CUDA tensors)
             ├── TensorFlow (CPU / CUDA tensors，eager only)
             ├── Native CPU (Rust / PyO3 whole-batch engine)
             └── Native CUDA (Rust / CUDA C++ whole-batch engine)
```

## Backend 邊界

`core` 不直接 import pandas、scikit-learn、CuPy、PyTorch 或 TensorFlow，而是使用 backend 提供的陣列運算、線性方程解法與 scalar conversion。CuPy 可額外提供融合的 CUDA C++ Huber／score／curvature kernel；若 NVRTC 不可用，會回退至共用 CuPy 表達式而不改變 API。

Backend 只在第一次 `fit`／`partial_fit` 時解析：

Native CPU 的公開 `n_jobs` 先由 Python 驗證與解析，再於 resident
`NativeCpuEngine` 建立時傳入一次。正整數與 `-1` 會建立 estimator-local
Rayon thread pool，使不同 estimator 的 worker 上限彼此隔離；`None` 則沿用
extension 的 process-global Rayon pool，以避免預設路徑反覆建立執行緒。
兩種模式都不會修改 process-global Rayon 或 BLAS 設定。Checkpoint 儲存原始
constructor 值（`None`、`-1` 或正整數），fitted estimator 則以 `n_jobs_`
公開實際 worker 數；其他 backend 明確忽略此設定。

- `backend="auto", device="auto"` 與 `backend="auto", device="cpu"` 先解析成
  NumPy，並在該批次通過驗證、形狀已知之後交由
  `backends.cpu_dispatch` 決定是否改用 `native_cpu`。因為兩個 CPU backend
  共用完全相同的 host NumPy 陣列處理（`NativeCpuBackend` 直接繼承
  `NumPyBackend` 的 `asarray`／`copy`／`reshape`／`to_numpy`／`scalar`／`xp`，
  且不提供自己的 design matrix），這個替換不會改變 solver 收到的資料。
  決策每條串流只做一次，不會在串流中途更換 engine。
- `backend="auto", device="cuda"` 解析成 CuPy，且需要可用的 NVIDIA CUDA
  裝置；CPU dispatch 完全不參與。
- `backend="native_cpu"` 明確選擇選用的 Rust/PyO3 whole-batch CPU 核心；
  明確指定時不會經過任何量測，失敗也不會被替換成 NumPy。
- `backend="native_cuda"` 明確選擇選用的 Rust/CUDA whole-batch GPU 核心；除了相容的 NumPy host input，也接受相同裝置、相同 dtype、C-contiguous 的 CUDA DLPack tensor；
  P2 不會由 `auto` 自動選取，且目前只支援 `penalty="none"`。
- `backend="torch"` 與 `backend="tensorflow"` 必須由呼叫端明確選擇；不會依輸入 tensor 推斷。
- `device="auto"` 對 Torch 與 TensorFlow 也選擇 CPU；CUDA 必須明確要求。

輸入會由選定 backend 轉型並移至它的裝置。一般跨框架輸入沒有 DLPack 零複製保證，可能發生配置或主機／裝置複製；`native_cuda` device-update 是明確例外，它會消費 CuPy／PyTorch CUDA 或 TensorFlow eager GPU 的 DLPack capsule，並以 device-to-device copy 搬入 engine workspace，全程不經 host。CuPy／PyTorch producer 會收到 consumer stream；TensorFlow legacy exporter 無法協商 stream，因此先建立明確 producer synchronization boundary。PyTorch tensor 使用共享 storage 的 detached view；TensorFlow 只支援 eager execution。Native CUDA 的 device-resident `predict` 尚未提供，會拒絕 CUDA tensor 而不執行隱式 D2H copy。

## 演算法正確性邊界

依原論文 Eq. (2.8) 與 Eq. (3.9)，新到批次的 estimating equation 使用 ordinary Huber score（將殘差截在 `[-tau, tau]`）；Eq. (2.1) 的平滑 score 導數只用來建立並累積歷史資訊矩陣 `J`。兩者不可互換，否則求解的 estimating equation 會改變。RPSHE 更新另保留上一批的 lambda subgradient，並以目前累積樣本數正規化歷史與當前項。

提供 `sample_weight` 時，current loss、score 與 curvature 都乘上該列權重，歷史資訊矩陣也累積加權 curvature。正規化、bandwidth 與 lambda 使用 `weight_sum`；`n_samples_seen` 仍保存實際列數。這使整數權重與明確重複列具有相同數值語意。

## State 與同步邊界

係數和累積資訊矩陣在更新、預測期間留在選定 backend。`predict` 直接回傳該 backend 的陣列；套件不會自動把 CUDA 預測複製回 NumPy。收斂判斷、公開 scalar 屬性及 checkpoint serialization 是允許的同步邊界。

`.npz` checkpoint 會把數值 state 轉為 NumPy 儲存，但同時保留原始 `backend`、`device` 與 `dtype` 設定。`load(path)` 依原設定重建 backend；`load(..., backend=..., device=..., dtype=...)` 則提供顯式且可測試的遷移邊界。套件不會在缺少 GPU 時靜默降級。

## 效能原則

1. 對整個 batch 向量化，避免 Python per-row 迴圈。
2. 由 BLAS/LAPACK、cuBLAS/cuSOLVER 或框架原生 kernel 處理矩陣乘法與求解。
3. 重用 Newton Hessian workspace，並在 CuPy 路徑融合分支密集的 elementwise kernel。
4. 將資料、係數與資訊矩陣留在同一裝置，避免每批來回複製。
5. 使用可重現 benchmark 評估穩態吞吐量；CUDA context、NVRTC 與 library 首次載入時間需和穩態時間分開觀察。

GPU 對小批次未必較快。CuPy kernel microbenchmark 見 `scripts/benchmarks/benchmark_cuda_kernels.py`，端到端 NumPy/CuPy 比較見 `scripts/benchmarks/benchmark_numpy_cupy.py`；量測 GPU 時應先 warm up 並同步 CUDA stream。

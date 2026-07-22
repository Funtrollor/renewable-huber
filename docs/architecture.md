# 架構

套件將「演算法」、「陣列運算後端」與「使用者 API」分離：

```text
RenewableHuberRegressor
        │
        ├── validation / design matrix / checkpointing
        ├── core.loss       (Huber、平滑 score、curvature)
        ├── core.update     (RHE Newton / RPSHE LAMM)
        └── backends        (NumPy 現已實作；GPU 後續加入)
```

`core` 不依賴 pandas、scikit-learn、CuPy、PyTorch 或 TensorFlow。它只使用由 backend 提供的 Array-API 形狀數值運算、線性方程解法與 scalar conversion。這能讓同一份演算法在 GPU 後端完成後持續保持單一來源。

## 運算模式

| 模式 | v0.1 狀態 | 後續實作原則 |
| --- | --- | --- |
| `backend="numpy"` | 支援 | 以連結的 BLAS/LAPACK 執行 CPU 線性代數。 |
| `backend="cupy"` | 支援 | 資料、係數與資訊矩陣長駐 CUDA；避免每個 batch 回傳 NumPy。 |
| `backend="torch"` | 支援 | 接受原生 Torch tensor，支援明確指定的 CPU 或 CUDA device。 |
| `backend="tensorflow"` | 保留 | 接受原生 TensorFlow tensor；不混用 eager/graph state。 |

CuPy implementation 必須通過與 NumPy reference 的數值一致性測試才能列為支援。不同框架在相同裝置上交換資料時，adapter 層才使用 DLPack；API 層不得無提示轉換或同步 GPU。

Windows 上 CuPy 的 cuBLAS 可能在第一次矩陣運算才延遲載入。後端會自動把偵測到的 CUDA Toolkit `bin` 目錄加入目前 Python 行程的 DLL 搜尋路徑；若未安裝 Toolkit，請依 CuPy 文件安裝含 CUDA runtime 的 `cupy-cuda12x[ctk]`。

## 效能原則

1. 對整個 batch 向量化，避免 Python per-row 迴圈。
2. 讓已編譯的 BLAS/LAPACK／CUDA library 處理矩陣乘法與求解。
3. 模型 state 永遠留在選定 backend；只在 `predict` 或 serialization 的明確邊界轉換。
4. 先以 benchmark 找到瓶頸，再考慮 Numba、CuPy RawKernel 或 C++/CUDA extension。

GPU 的 CUDA context 與 cuBLAS 首次載入有固定成本，因此應以長時間串流吞吐量，而非第一個微小 batch 的時間，評估 GPU 效益。使用 `scripts/benchmarks/benchmark_numpy_cupy.py`，並在 GPU 上優先嘗試 `float32` 與至少數萬筆資料的 batch。

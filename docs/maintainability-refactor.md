# 可維護性重構 P0–P2 交接報告

- 分支：`codex/maintainability-p0`（14 個本機 commit，重構基底 `27322c4`）
- 依據：`renewable-huber 代碼結構與可維護性稽核報告`（稽核版本 df17d08）
- 狀態：P0、P1、P2 完成並驗證；**P3 未進行**
- 遠端狀態：Codex 以 GitHub Git Data API 將已審核工作樹發布到
  `codex/maintainability-p0`；Claude Code 不執行 commit/push。本機
  remote-tracking ref 必須在 `.git` 恢復可寫後 fetch 對齊。

---

## 0. 先讀這一節：編號會撞名

這份文件裡的 **P0/P1/P2 是稽核報告的「維護性優先序」**，和 `docs/native-core-p0-baseline.md`、`native-core-p1.md`、`native-core-p2.md`、`native-core-p4.md` 裡的 **native core 實作階段完全無關**。

| 編號 | 本文件的意思 | `docs/native-core-*.md` 的意思 |
|---|---|---|
| P0 | 修 C ABI 契約、建跨語言 contract、拆 `engine.cu` | 純 Python 基線 |
| P1 | Python backend capability contract | Rust CPU 引擎 |
| P2 | 拆 Rust 模組 | CUDA whole-batch 引擎 |
| P3 | persistence 與測試 profile（**未做**） | — |

看到 "P2" 時務必先確認是哪一套。

---

## 1. 這次重構的原則
> 專案根目錄的 `AGENTS.md` 是這份報告的精簡版，Codex 每個 session 會自動讀取。
> 它只放「違反會安靜壞掉」的不變條件與驗證指令；改動任何一邊時請保持兩邊一致。


三條，違反其中任何一條的修改都應該被退回：

1. **不改演算法、kernel 順序、stream 行為、公開 API。** 這是純結構重構。所有效能與正確性成果（CPU 1.45×–15.65×、CUDA 1.12×–2.04×、golden corpus）必須逐位元保持。
2. **隱性契約要變成會失敗的測試。** 稽核指出的問題不是「程式碼很醜」，而是「契約漂移時沒有任何東西會壞」。每一項修正都附帶一個會抓到回歸的測試。
3. **每一步都要能獨立驗證。** 特別是 `engine.cu` 的拆檔，分成八個可各自 build + 測試的步驟。

---

## 2. P0：降低立即風險

### 2.1 C ABI batch/intercept 契約（commit `d51503c`）

**問題**：`rh_cuda.h` 寫 `x_design` 形狀為 `(n_rows, n_parameters)`、intercept 由 caller 建構。但 `validate_batch` 早就同時接受 `n_columns == n_features_in`，`copy_batch` 在該情況於 device 端補上 intercept 欄。而 `NativeCudaBackend.native_design_matrix` 刻意不在 host 展開 —— **窄矩陣才是生產熱路徑**，header 描述的是沒人走的路。

**做法**：改文件對齊實作。改實作會讓 CUDA backend 退回 host 端 `column_stack`，在每次 H2D 前多複製整個 batch。

同時補上 `RhCudaDeviceBatch` 原本**完全沒有**的寬度說明，並在 `RhCudaUnpenalizedConfig` 註明 `n_parameters ∈ {n_features_in, n_features_in + 1}` 的 intercept 不變條件。

**順帶修掉一個真實漏洞**：`validate_config` 原本只檢查 `0 < n_features_in <= n_parameters`。直接用 C ABI 傳 `n_features_in=1, n_parameters=4` 加窄 batch，device 端的 append 只會填 2 欄，`d_design` 其餘欄位是前一批的殘留資料，solver 就對著未初始化記憶體求解。Rust wrapper 的 `validate_unpenalized_config` 本來就擋住這個組合，C ABI 沒有。現在兩邊一致。

> 注意：溢位。原本想寫 `n_features_in + 1 != n_parameters`，但 `n_features_in` 上界不受限，`INT64_MAX + 1` 是 UB。實作用減法 `engine->n_parameters - config->n_features_in` 比較，兩個運算元都是正數。

### 2.2 跨語言 contract manifest（commit `21ddf00`）

**問題**：同一組契約獨立寫在四個地方 —— `rh_cuda.h`、Rust 手寫 `mod ffi`、兩個 PyO3 result-dict builder、兩份 Python `_decode_result` —— 而且沒有任何一處會在漂移時失敗。三個具體缺口：

- Rust 那 18 處 `size_of::<ffi::X>()` 是**把 Rust 的 layout 報告給 C**，從來沒驗證過它等於 C 的 layout。
- `mod ffi` 被 `#[cfg(feature = "cuda")]` 包住，所以 CI 的 `cargo test -p rh-cuda-ffi` **根本沒編譯到那些 struct**。
- `rh_cuda_abi_version()` 在 header 有、`engine.cu` 有實作，但 Rust 的 extern block 沒宣告它。`version()["abi_version"]` 回傳的是 Rust 常數，自己對自己。

**做法**：`native/contracts/rh_cuda_contract.json` 成為單一真實來源。裡面的數字**由編譯 `rh_cuda.h` 產生，不是手算**（用一支 dump 程式印出 `sizeof`/`offsetof`）。

三份鏡像各自對照它：

| 鏡像 | 檔案 | 由誰檢查 |
|---|---|---|
| C++ | `native/cuda/src/abi_contract.cpp`（170 個 `static_assert`，不定義任何符號） | 真實編譯器。`rh_cuda.h` 只 include `<stdint.h>`，所以 `g++ -fsyntax-only` 就能驗，CI 免 GPU 免 CUDA Toolkit |
| Rust | `native/crates/rh-cuda-ffi/src/sys.rs` 的 `mod abi_layout`（`offset_of!`） | 既有的無 CUDA CI job |
| PyO3 | `rh-python-cuda` 的 `set_item` key 集合 | Python 裁判 |

`tests/test_native_cuda_contract.py` 是裁判，失敗時輸出 unified diff 並指名雙方檔案。

**關鍵決定：不用 bindgen。** 稽核報告建議 pregenerated bindgen，但 bindgen 需要 libclang 與 CUDA headers、只重生 `mod ffi`（那部分本來就對）、對 status code / dtype code / flag bits / Python dict keys 零覆蓋，而 `rh-cuda-ffi` 必須在完全沒有 CUDA toolchain 時也能編譯。manifest + 三鏡像成本更低、覆蓋更廣。

**Rust 側的關鍵重構**：把 `#[repr(C)]` records 從 cfg-gated 的 `mod ffi` 移到**未 gate 的 `mod abi`**，`mod ffi` 只剩 `pub use super::abi::*;` 加 extern block。下游全部仍寫 `ffi::RhCudaHostBatch`，18 處 `size_of` 一行都沒改，而 layout 測試從此在無 CUDA 的 CI job 裡跑。

**真正的 runtime 檢查**：宣告 `rh_cuda_abi_version()`，在建 engine 前比對，不符就拒絕；`version()["abi_version"]` 改回報實際連結的 library。這是唯一能抓到 stale binary 的機制 —— 任何 source-level 測試都取代不了。

### 2.3 `engine.cu` 拆檔（commits `3212d1e` / `d100ab6` / `76668f8`）

2,638 行單一 TU 拆成 7 個，最大 528 行。

| 檔案 | 行數 | 職責 |
|---|---:|---|
| `c_api.cu`（原 `engine.cu`） | 566 | guards、error translation、17 個 `extern "C"` |
| `pipeline.cu` | 341 | 一次 batch transition、host prediction |
| `linear_solver.cu` | 301 | Cholesky → LU → lazy SVD |
| `objective.cu` | 528 | objective、gradient/Hessian、candidate CUDA Graph |
| `batch.cu` | 138 | batch 驗證、staging、intercept append |
| `workspace.cu` | 270 | device 配置與 engine destructor |
| `engine_internal.cu` | 49 | memory pool registry、`Failure` key function |

加上 5 個 header：`engine_internal.cuh`、`blas_traits.cuh`、`engine_state.cuh`、以及各模組的 `.cuh`。

**與稽核建議的四點差異**（都有理由）：

- 不做 `memory_pool.cu` —— 40 行加一個 function-local static，沒有天然同伴，併入 `engine_internal.cu`。
- 不做 `prediction.cu` —— `predict_typed` 只有 50 行，且與 update 共用 `ensure_batch_capacity` / `d_design` / `d_residual`；獨立出去反而讓 `d_residual` 的雙重角色更難看見。
- **新增 `blas_traits.cuh`** —— 稽核清單沒有給 `Blas<T>`/`Solver<T>`（316 行）任何歸屬，而 4 個新 TU 需要它。放 `engine_state.cuh` 會讓每個 consumer 被迫拉進 `cublas_v2.h` 與 `cusolverDn.h`。
- **新增 `batch.cu`** —— 稽核清單沒有 batch 層的家。刻意獨立，因為這正是契約漂移與測試缺口所在。

拆解順序刻意把「改 linkage」（step 1）和「改 TU 數量」（step 2–8）分開，出事時知道是哪一類造成的。

### 2.4 C++ smoke test 補洞

`rh_cuda_smoke.cpp` 原有 317 行**完全沒動**（它是拆檔的迴歸基線），新增七個案例：

- **intercept append 差分測試（p=2、p=3）** —— 同一 zero state 跑寬/窄矩陣，要求 1e-12 內一致。兩條路只差在 `d_design` 怎麼填，後續 cuBLAS 看到的完全相同。一次抓到 intercept 放錯欄、row stride 錯、少元素、填錯值。
  - **p=3 不是多餘的**：`feature_columns == 1` 時 `features[row*cols+col]` 與 `features[col*rows+row]` 是同一個式子。把 kernel 索引轉置驗證過：p=2 仍然通過，p=3 失敗（`wide 0.166667 vs narrow 0.05`）。
- **device-input intercept append** —— `copy_kind == DeviceToDevice` 時 `copy_batch` 跳過 staging，直接讀 caller 的 device pointer。這條分支原本完全沒測到。
- **`d_y` alias 還原** —— device update 後緊接一次 host update 並要求成功。`ScopedPointerAlias` 若哪天不還原，下一次 host update 會寫進 producer 已釋放的記憶體。
- **device pointer 拒絕** —— 刻意不 assert 確切狀態碼：`cudaPointerGetAttributes` 對未註冊 host pointer 的行為隨 CUDA 版本而異（舊版 → `CUDA_ERROR`，11+ → `INVALID_ARGUMENT`），兩者都是正確拒絕。
- **跨 TU 狀態碼一致性** —— 見第 6 節。
- **ABI 版本自檢**。

---

## 3. P1：Python backend capability contract（commit `cf8b7a8`）

**問題**：`ArrayBackend` 宣告 9 個方法，但實際控制流另外在 **13 處**用 `getattr` 探測能力，外加一處 `backend.name in {"numpy","cupy"}` 字串嗅探。少掉一個能力會靜默退回 portable 路徑：結果仍正確、可能慢上數量級、而且沒有任何東西會失敗。

**做法**：`src/renewable_huber/backends/capabilities.py` 的 `capabilities_of()` 是唯一做探測的地方。

兩個欄位**不是固定值**，這是設計上最容易搞錯的地方：

- `effective_n_jobs` 在 native engine 建立前回報請求的執行緒數，建立後回報 engine 確認的數量。
- `cuda_features` 帶著會累加的 CUDA Graph counter。

所以 capabilities 存的是**零參數 accessor**（`read_n_jobs` / `read_cuda_features`）而非快照。`tests/test_backend_capabilities.py` 直接針對這個差別測試 —— 若改成快照，其他所有測試都還是會過。

`elementwise_workspace` 取代字串嗅探。native backend 繼承自 `NumPyBackend`，所以必須明確 `supports_elementwise_workspace = False` opt out，否則會靜默加入那個集合。有測試盯著這個四分法。

**共用 native adapter**：`backends/native_base.py` 吸收 CPU/CUDA 各自寫過一遍的 ABI 協商、engine mirror token、restore 樣板，以及那個**除了一行註解外逐字相同**的 `_decode_result`。兩個行為用 hook 保住而不是壓平：CUDA 的 canonical-empty 捷徑成為 `_new_engine_already_holds`，各自內聯 4 處的 discard 邏輯成為 `_engine_call` context manager。

**`partial_fit`** 從 69 行做五件事變成 26 行：validate → prepare → transition → commit。驗證完全發生在碰任何 fitted attribute 之前，所以被拒絕的 batch **可證明**不會留下半更新的 estimator。

**意外的效能收穫**：portable NumPy 路徑 median fit 2.94ms → 2.56ms。那三個 fused kernel 探測在求解迴圈裡每次迭代都跑，而失敗的 `getattr` 內部會拋出並捕捉 `AttributeError`；換成讀快取的 dataclass 就沒這個成本。

---

## 4. P2：拆 Rust 模組（commits `b66616d` / `655c12e`）

公開 crate API 不變，用 rustdoc 快照逐項驗證。

**`rh-cpu`**：1,530 行拆成九個，最大 418。kernels 按**操作對象**分組：`vector`（batch 上的 elementwise 與 reduction）、`gram`（p×p 累積、決定何時啟用 Rayon）、`objective`（求解器實際最小化的東西）。`bandwidth`/`lambda` 是純 scalar policy，三者都不屬於，留在 `kernels/mod.rs`。三個 parallelism threshold 留在 `lib.rs` 當 `pub(crate)` —— workspace、engine、kernels 都讀它，放進任何一個都會倒轉依賴方向。

**`rh-cuda-ffi`**：raw C 宣告原本躺在包裝它們的 safe API 同一檔案底部。`sys.rs` 現在單獨持有它們（未 gate 的 records、cfg-gated extern block、layout tests），讓「這份宣告跟 `rh_cuda.h` 一致嗎」變成可獨立完成的 review。其餘分為 `types` / `validation` / `engine` / `runtime`。

**`rh-python-cuda`**：DLPack 移到 `dlpack.rs`。那是整個 extension 裡唯一解參考其他框架交過來的指標、也是唯一正確性靠協定而非 Rust 型別系統保證的地方。

**效能未量前後對比，這是刻意的**：Rust 的 `mod` 是 crate 內的命名空間，crate 才是編譯單元，所以不像 `engine.cu` 拆檔那樣可能影響 inlining —— 機制上不存在。仍跑了一次 CUDA host sweep 作 sanity check，八個 shape 全部落在 P0 建立的區間內。

---

## 5. 怎麼重跑每一道 gate

```bash
# Python（含契約裁判與 capability contract）
.venv/bin/python -m unittest discover -s tests

# lint
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts

# Rust（layout tests 在這裡跑，不需要 CUDA）
cd native
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo check  --locked --workspace --all-targets
# 注意 crate 範圍：不能用 --workspace，理由見下方
cargo test   --locked -p rh-core -p rh-cpu -p rh-cuda-ffi --all-targets

# C ABI layout，不需要 GPU 也不需要 CUDA Toolkit
g++ -std=c++17 -fsyntax-only -I native/cuda/include native/cuda/src/abi_contract.cpp
```

> **`cargo test` 在 Linux 上不能用 `--workspace`。** PyO3 的 `extension-module`
> crate（`rh-python-cpu`、`rh-python-cuda`）在 Linux 連成獨立測試執行檔時，CPython
> 的符號無法解析，`cc` 會失敗。這不是回歸 —— `ci.yml` 早就用限定 crate 的寫法並
> 註記了原因。Windows 上因為 PyO3 連結 Python import library 的方式不同而不會出現。
> 那兩個 crate 由 `cargo check` / `clippy`（workspace 範圍）與實際的 wheel 建置涵蓋。

CUDA 部分（需要 nvcc）：

```bash
cmake -S native/cuda -B build/static -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DRH_CUDA_BUILD_SHARED=OFF -DRH_CUDA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/static && ctest --test-dir build/static --output-on-failure
```

**export table 不變條件**（見第 6 節）：

```bash
cmake -S native/cuda -B build/shared -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DRH_CUDA_BUILD_SHARED=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/shared
nm -D --defined-only --extern-only build/shared/librenewable_huber_cuda.so | grep rh_cuda_
```

Linux 用 `nm -D`，Windows 用 `dumpbin /exports`。基線在 `artifacts/baseline-exports.txt`（Windows 產生，符號名相同）。

---

## 6. 不變條件清單：違反會「安靜壞掉」

這一節是本文件最重要的部分。以下每一項若被破壞，**程式不會 crash、數值仍然正確**，只有契約悄悄失效。

| # | 不變條件 | 破壞後的症狀 | 守門的東西 |
|---|---|---|---|
| 1 | 匯出恰好 17 個 `rh_cuda_*` 符號 | ABI 表面擴大或縮小，下游連結行為改變 | `artifacts/baseline-exports.txt` 手動比對。Linux 上額外會看到幾個 libstdc++ 的 `std::string` template 實例化（`nm` 標記 `W`，弱符號）—— 那是 libstdc++ 標頭自帶 default visibility 所致，與本專案無關，比對時只看 `rh_cuda_` 前綴 |
| 2 | `Failure` 不得放進 header 的 anonymous namespace，且必須保留 out-of-line destructor | 每個 TU 得到不同型別，`catch (const Failure&)` 停止匹配，**所有錯誤狀態碼變成 `INTERNAL_ERROR (8)`** | smoke 的 `status_survives_translation_units` |
| 3 | `RhCudaEngine` 必須留在 global scope | `rh_cuda::RhCudaEngine` 與 `::RhCudaEngine` 成為兩個型別，ABI 的那個永遠 incomplete | 編譯錯誤（會 fail fast，相對安全） |
| 4 | `mod abi` 不得放回 `#[cfg(feature = "cuda")]` | layout tests 停止在無 CUDA 的 CI 跑，Rust 鏡像靜默失守 | `test_repr_c_records_are_not_gated_on_the_cuda_feature` |
| 5 | manifest 先改，四個鏡像後改 | 鏡像之間互相矛盾 | `tests/test_native_cuda_contract.py` |
| 6 | 契約測試的每個 parser 必須先 assert「找到的數量 == manifest 預期數」 | regex 哪天靜默 match 不到，整套契約測試變成永遠通過的空殼 | `assert_found()`（已內建） |
| 7 | `allocate<T>` 有**三種**實例化（`float`/`double`/**`int`**，後者給 `d_pivots`、`d_solver_info`） | 若「整理」成顯式實例化清單會漏掉 `<int>` | link error（安全） |
| 8 | 預設參數只屬於宣告，不得在定義重複（`smooth_objective`、`update_typed`） | 編譯錯誤（安全） | 編譯器 |
| 9 | `capabilities_of` 以實例快取 —— backend 建構後不得增減方法 | 快取過期，能力探測結果錯誤 | 無自動守門，靠 `resolve_backend` 的建構慣例 |
| 10 | `read_n_jobs` / `read_cuda_features` 必須是即時 accessor，不能改成快照 | `n_jobs_` 永遠回報建 engine 前的值；graph counter 凍結 | `tests/test_backend_capabilities.py::LiveAccessorTests` |
| 11 | Rust 拆模組時用 `pub(crate)` 而非 `pub` | 公開 API 意外擴大 | rustdoc 快照比對（手動） |
| 12 | cfg-gated 項目的 import 必須帶同樣的 gate | 無 CUDA 建置解析失敗 | `cargo check -p rh-cuda-ffi`（不帶 feature） |
| 13 | `n_parameters ∈ {n_features_in, n_features_in + 1}` | 窄 batch 時 device append 只填部分欄位，solver 讀殘留記憶體 | `validate_config` + smoke 的 `intercept_invariant_is_enforced` |

### 6.1 rustdoc 快照的陷阱

比對公開 API 時，**從私有模組 re-export 會讓 rustdoc 產生 redirect stub**（`engine/fn.predict.html` 之類，declaration 與 members 皆空）。第一次比對時它們被算成「新增 8 個公開項目」。工具必須排除含 `http-equiv="refresh"` 的頁面，否則模組重排會被誤報成 API 變更。

### 6.2 效能量測的雜訊特性

CUDA benchmark harness 在這台機器上，**同一份 binary 連跑三次 median 就會漂 2%–19%**（`wide float32` 最差）。單次前後對比曾顯示 `wide f32` +12%，實際是熱漂移。

要判定效能是否退步，必須用**交錯 A/B**（A/B/A/B 交替、比較配對差）而非「先跑五次 A 再跑五次 B」—— 後者比較的是兩個不同的熱狀態，不是兩份 binary。P0 用 14 對配對測量得到：f32 有 9 對為正、f64 有 8 對，與擲硬幣無異。

**此 harness 在最吵的 shape 上約只能分辨 ±10%。** 低於這個幅度的差異不要下結論。

---

## 7. 環境現況

### 7.1 WSL（新的主要工作環境）

- 位置：`/home/untrollor/renewable-huber`（ext4，**不是** `/mnt/c`；後者慢且有權限與大小寫問題）
- Ubuntu 24.04.3，kernel 6.6.87.2-microsoft-standard-WSL2
- `origin` 已指回 `https://github.com/Funtrollor/renewable-huber.git`
- 分支 `codex/maintainability-p0`，8 個 commit，working tree 乾淨
- `artifacts/` 的 105 個驗證基線已一併帶過來（gitignored）
- **已安裝並驗證**：
  - gcc 13.3.0、g++、make、cmake 3.28.3、ninja 1.11.1、pkg-config
  - Python 3.12.3、pip、venv（`.venv/`，editable 安裝指向 `src/`）
  - Rust 1.97.1（rustup，`~/.cargo`，含 rustfmt/clippy）
  - native CPU extension 已建置並安裝（`cp312` manylinux wheel），`abi=1 api=2`
- **GPU passthrough 可用**：`nvidia-smi` 看得到 RTX 5070 Ti，driver 596.49，
  `/usr/lib/wsl/lib/libcuda.so` 存在
- **CUDA Toolkit 12.9 已安裝**（`/usr/local/cuda`），native CUDA extension 已建置並載入，`abi=1 api=3`，`cuda_available=True`

已在 WSL 通過的 gate：

| gate | 結果 |
|---|---|
| `python -m unittest discover -s tests` | 197 passed, 43 skipped（多出的 9 個 skip 是 CUDA 測試） |
| `ruff check` / `ruff format --check` | 通過 |
| `g++ -fsyntax-only ... abi_contract.cpp` | 170 個 static_assert 成立 |
| `cargo fmt --all -- --check` | 乾淨 |
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 diagnostics |
| `cargo test -p rh-core -p rh-cpu -p rh-cuda-ffi` | 14 passed（含兩個 golden corpus replay 與 layout tests） |
| `cargo test -p rh-cuda-ffi --features cuda` | 5 passed（真正編譯 extern block 的組態） |
| `ctest`（C++ ABI smoke） | 1/1 passed |
| `nm -D` 匯出比對 | 與 Windows 基線的 17 個符號逐字相同 |

整個 CUDA 流程已在 Linux 驗證通過。過程中修掉三個只在 Linux 才會浮現的可攜性問題，
都記在 `native/crates/rh-cuda-ffi/build.rs` 與 `scripts/setup-wsl-venv.sh`：

| 症狀 | 原因 | 修法 |
|---|---|---|
| `unable to find library -lcudart` | `build.rs` 只在 `CUDA_PATH` 有設時才加連結搜尋路徑。那是 Windows 安裝程式設的變數，Linux 沒有 | 依序查 `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT`、從 `PATH` 上的 `nvcc` 反推 toolkit 根目錄、最後退回 `/usr/local/cuda`；後綴試 `lib64`、`targets/x86_64-linux/lib`、`lib` |
| import 時 `undefined symbol: __gxx_personality_v0` | CUDA C ABI 是**會拋例外的 C++**，需要 libstdc++ 的例外處理常式。MSVC 透過 .lib 內嵌的預設函式庫指令自動帶入，GNU 工具鏈不會，而 rustc 傳 `-nodefaultlibs` | Linux 連 `stdc++`、macOS 連 `c++` |
| maturin 要求 `patchelf` | 預設會把 CUDA `.so` 全部打包進 wheel 並改寫 rpath | 本機開發用 `--auditwheel skip`；系統已有 toolkit，`ldconfig` 會在 import 時解析。發布 wheel 由 packaging workflow 負責，不走這支腳本 |

前兩項是 `build.rs` 的實質 bug，不是環境設定問題 —— 那個檔案只在 Windows 上被驗證過。
這也說明為什麼把開發環境搬到 Linux 對這個專案有實質價值：CI 跑的就是 Linux。

若要在另一台機器重建：

```bash
bash scripts/setup-wsl-toolchain.sh --cuda   # NVIDIA wsl-ubuntu repo，只裝 toolkit
bash scripts/setup-wsl-venv.sh --cuda
```

> **絕對不要在 WSL 裡安裝 Linux 顯示驅動。** WSL 的 GPU 是透過 Windows 驅動
> passthrough，裝 Linux driver 會遮蔽掉它。上面的 repository 只含 toolkit。

### 7.2 Windows 副本（原位置，仍存在）

`C:\Users\Funtrollor\Desktop\bigDataAnynisit` **沒有刪除**。目前狀態：

- 有完整的 8 個 commit，但**沒有這份報告的 commit**（報告在 WSL 端提交）
- 有已建好並安裝的 native CPU + CUDA extension（`tmp/refactor-env`，Python 3.11）
- **目前唯一驗證過 CUDA 完整流程的環境**

同步方式：push 到 GitHub 後兩邊都從那裡拉。確認 WSL 環境可用之後再決定是否刪除 Windows 副本。

### 7.3 Agent 環境（Claude Code / Codex）

兩個 CLI 都裝在 WSL，透過 nvm 管理的 Node，全部在 `$HOME` 底下，不需要 root：

| 項目 | 版本 / 位置 |
|---|---|
| Node | v24.19.0（`~/.nvm`） |
| Claude Code | 2.1.223 |
| Codex CLI | 0.146.1 |

Windows 端的 session 紀錄已經搬過來：

- Claude Code 以**工作目錄**當索引鍵，規則是把所有非 `[a-zA-Z0-9]` 的字元各換成一個 `-`
  （`_` 也換，每個中文字各換一個）。所以
  `C:\Users\Funtrollor\Desktop\bigDataAnynisit` → `C--Users-Funtrollor-Desktop-bigDataAnynisit`，
  而 `/home/untrollor/renewable-huber` → `-home-untrollor-renewable-huber`。
  兩個 session（6.4 MB）已複製到後者。
- Codex 的 session 按日期分層並另有索引，`sessions/`、`session_index.jsonl`、
  `config.toml`、`skills` 已複製（270 MB，其中 51 個 session 提到這個專案）。

**`auth.json` 與憑證刻意沒有複製** —— 在 WSL 端重新登入。

搬過來的 transcript 記錄的是 Windows 路徑。實測主 session 1691 行裡只有 42 處
`C:\Users`、127 處提到 PowerShell，而 1457 行提到 `renewable-huber` —— 也就是說
**實質內容（改了什麼、為什麼、驗證結果）與路徑無關**，Windows 特定的部分集中在 shell
呼叫上。歷史可讀，但接手時仍應以本文件與 git history 為準：它們是被驗證過並萃取出來的
結論，而 transcript 是過程。

### 7.4 Windows/PowerShell 的坑（若還會用到）

- **`2>&1` 用在原生執行檔上會造成假失敗**：PowerShell 5.1 把每行 stderr 包成 ErrorRecord，即使 exit code 是 0 也會讓 `$?` 變成 `$false`。maturin 建置成功卻回報 exit 1 就是這個原因。導向檔案再讀取。
- **heredoc 不存在**，`<<'EOF'` 是 parser error。多行字串用 here-string 或改用 Bash 工具。
- **CMake generator 快取**：`cmake` crate 在 Windows 預設選 Visual Studio generator，與既有的 Ninja 快取衝突。建 CUDA wheel 前必須 `$env:CMAKE_GENERATOR = "Ninja"`（`scripts/native/build_native_cuda.ps1` 有做，直接呼叫 maturin 時要自己設）。
- **`Set-Content -Encoding utf8` 會加 BOM**，Python 讀檔要用 `utf-8-sig`。

---

## 8. 尚未進行：P3

稽核報告的 P3 有三項，**都沒做**：

1. `CheckpointPayload` —— 目前 `serialization.load_model` 最後一行直接呼叫 estimator 的私有 `_restore_state`，persistence 層知道 estimator 的私有生命週期。
2. 測試 profile。
3. benchmark shape sweep 拆分。

### 8.1 P3 第 2 項的前提是錯的，照抄會做白工

稽核報告寫「為 pytest 加上 `gpu`、`optional_backend`、`performance` markers」。實際情況：

- **沒有任何測試 import pytest。** 全部是 `unittest.TestCase`。
- **CI 一律跑 `python -m unittest`**；GPU suite 則由固定本機主機直接以
  `python -m unittest` 執行，從來沒跑過 pytest。
- **repo 裡沒有 `conftest.py`**，`pyproject.toml` 的 `[tool.pytest.ini_options]` 只有 `testpaths` 與 `addopts`。
- `pytest` 雖在 `dev` extra 裡宣告，但沒有東西用它。

只加 marker 對 CI 完全無作用。可行做法是加一個 root `conftest.py`，用 `pytest_collection_modifyitems` 依模組名自動標記，同時**不動那 18 個測試模組、也不破壞 `python -m unittest`**。這樣 pytest 使用者有 marker 可選，CI 維持現狀。

---

## 9. 已知未修的小瑕疵

重構過程中發現、判斷不屬於本次範圍而**刻意保留**的：

1. CUDA `state_dict()` 走 `state_dict_from_parts`，**不含 `state_is_detached` key**（CPU 版有）。目前無害，因為 `_decode_result` 用 `.get(..., False)` 保守複製。manifest 已把這個差異明文化。
2. `_restore_state` 不設 `_diagnostics`，所以剛 `load()` 的模型讀 `diagnostics_` 會拋 `NotFittedError`。
3. 可攜式 core 從不設 `used_regularized_fallback`（永遠 `False`），但兩個 `_decode_result` 都把它當**必要** key。
4. `_design_matrix` 只在傳入 `backend=` 時才委派 —— `partial_fit` 傳、`predict` 不傳。這是刻意的（CUDA engine 的 predict 需要完整 `n_parameters` 寬度，update 接受未展開的），已加註解說明。

---

## 10. 檔案尺寸現況

拆檔後仍在 500 行以上的檔案，都屬於「大但內聚」，**不建議為了行數再拆**：

| 檔案 | 行數 |
|---|---:|
| `native/crates/rh-cuda-ffi/src/engine.rs` | 693 |
| `native/crates/rh-python-cuda/src/lib.rs` | 627 |
| `native/crates/rh-python-cpu/src/lib.rs` | 615 |
| `src/renewable_huber/estimator.py` | 591 |
| `native/cuda/src/c_api.cu` | 566 |
| `native/cuda/src/objective.cu` | 553 |
| `native/cuda/src/huber_kernels.cu` | 542 |
| `src/renewable_huber/core/update.py` | 514 |

稽核報告自己就把 `core/update.py` 列為「大但內聚」的正面例子。`rh-python-cpu/src/lib.rs` 是唯一還沒動過的 PyO3 crate，若之後要拆，可比照 `rh-python-cuda` 的做法。

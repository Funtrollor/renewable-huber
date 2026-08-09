# 發布流程

專案以三個彼此獨立的 PyPI distribution 發布：

| Distribution | 內容 | 使用者是否需要編譯 |
| --- | --- | --- |
| `renewable-huber` | 公開 Python API 與 NumPy backend | 否 |
| `renewable-huber-native-cpu` | Rust／Rayon CPU extension | 否；從對應平台 wheel 安裝 |
| `renewable-huber-native-cuda` | Rust/CUDA 12 extension | 否；目前發布 Windows x86-64 wheel |

三個 distribution 在同一次 release 使用相同版本。Native wheel 會以精確相依條件
`renewable-huber==X.Y.Z` 鎖定 base package，避免 Python API、checkpoint 或 native ABI
不相容的組合被 pip 解析在一起。`scripts/native/validate_release_artifacts.py` 會在 CI
同時驗證來源 metadata 與最終 wheels。

`v0.6.0` tag 曾用於未完成的發布流程，沒有成為正式 PyPI release。Tag 保持不可變；
本次完整 native release 使用 `v0.6.1`，release notes 以 `v0.5.1` 為使用者升級基線。

## Wheel 支援範圍

CPU release matrix：

- CPython 3.10、3.11、3.12。
- manylinux2014 x86-64 與 aarch64。
- Windows x86-64。
- macOS x86-64 與 Apple Silicon arm64。

CUDA 12 release matrix 目前為 CPython 3.10–3.12、Windows x86-64。Release fat binary
包含 SM 75、80、86、89、90 與 120，因此建置 runner 必須使用 CUDA Toolkit 12.8
以上。使用者安裝已發布 wheel 時不需要 Rust、CMake、Visual Studio 或 `nvcc`，但仍需
相容的 NVIDIA driver，以及 CUDA 12 的 `cudart`、cuBLAS、cuSOLVER runtime DLL。

CUDA wheels 在固定的 GitHub-hosted `windows-2022` runner（Visual Studio 2022）
安裝 CUDA 12.9 build-only toolchain 後編譯；不用 `windows-latest`，因為它已移至
CUDA 12.9 尚未支援的 Visual Studio 2026。該 runner 沒有 GPU，也不執行 CUDA
runtime 測試。Pull request
與一般開發不使用 GitHub Actions GPU runner；correctness、profiling、乾淨安裝與
performance 驗證都在維護者的固定本機 GPU 主機執行。

macOS CPU wheels 使用 `macos-15-intel`（x86-64）與 `macos-15`（Apple Silicon）；
不要恢復已停止支援、會永久排隊的 `macos-13` 標籤。

## Release gate

1. 更新 `src/renewable_huber/_version.py`，並同步兩個 native `pyproject.toml` 的版本及
   精確 base dependency。
2. 把 `CHANGELOG.md` 的 Unreleased 內容移到該版本，確認授權與引用資訊。
3. 在固定本機 GPU 主機完成 correctness、C ABI smoke 與 performance gates，保存
   commit、環境指紋和 JSON 證據；合併至 `main` 後等待一般 CI 與 CPU wheel
   clean-install 通過。
4. 在 WSL2/Linux 本機先執行 metadata 與 required-profile dry-run：

   ```bash
   .venv/bin/python scripts/native/validate_release_artifacts.py --source-only
   .venv/bin/python scripts/run_test_profile.py --check
   .venv/bin/python scripts/run_test_profile.py core --verbose
   .venv/bin/python scripts/run_test_profile.py performance --verbose
   ```

5. 在 GitHub 對 `main` 手動執行 `release.yml`。這是 build-only rehearsal：建置並
   驗證完整 20 個 artifacts，但不建立 GitHub Release，也不發布到 PyPI/TestPyPI。
6. 從已通過一般 CI 與 build-only rehearsal 的**精確 `main` tip** 建立 `vX.Y.Z`
   signed tag。Release workflow 會拒絕版本不一致或不是目前 `main` tip 的 tag。
7. Workflow 建置 base wheel/sdist、15 個 CPU wheels、3 個 CUDA wheels，並執行
   Twine、metadata、ABI capability 與 artifact-set 檢查；GPU runtime 測試不在
   GitHub Actions 執行，採用第 3 步的本機證據。
8. 完整 artifact set 通過後才建立 GitHub Release。人工核准 PyPI 前，下載實際
   CUDA artifacts 至固定 GPU 主機，對 artifact hash 執行最後 smoke。
9. 三個 PyPI publish jobs 分別等待對應 GitHub Environment 的人工核准，再透過
   Trusted Publishing/OIDC 發布；不儲存長效 API token。

Release tag 範例：

```bash
git switch main
git pull --ff-only
git tag -s vX.Y.Z -m "renewable-huber vX.Y.Z"
git push origin vX.Y.Z
```

## PyPI Trusted Publishing

三個 PyPI project 必須各自建立 pending publisher，並精確設定 repository、workflow
與 environment：

| PyPI project | Workflow | GitHub environment |
| --- | --- | --- |
| `renewable-huber` | `release.yml` | `pypi` |
| `renewable-huber-native-cpu` | `release.yml` | `pypi-native-cpu` |
| `renewable-huber-native-cuda` | `release.yml` | `pypi-native-cuda` |

三個 environment 均應啟用 required reviewer。即使有人誤推 tag，artifact 仍不會在未經
核准時上傳 PyPI。GitHub Release 可先建立供維護者下載與人工驗證。

TestPyPI workflow只 rehearsal base distribution；完整三套 distribution 的建置驗證
以 `release.yml` 的手動 build-only run 為準。正式 tag 前必須確認三個 PyPI pending
publisher（或既有 project publisher）的 owner、repository、workflow 與 environment
完全匹配，且 Environment tag rules 允許目標 tag。

## 安裝後驗證

CPU 使用者：

```bash
python -m pip install renewable-huber-native-cpu==0.6.1
python -c "from renewable_huber import RenewableHuberRegressor; print(RenewableHuberRegressor(backend='native_cpu', n_jobs=-1))"
```

CUDA 12 使用者：

```powershell
python -m pip install renewable-huber-native-cuda==0.6.1
python -c "from renewable_huber import _native_cuda; print(_native_cuda.version()); print(_native_cuda.is_available())"
```

若 native distribution 已存在於 PyPI，禁止重新使用相同版本覆寫 wheel；修正版本與
changelog 後發布新的 patch release。

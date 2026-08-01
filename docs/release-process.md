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

CUDA wheels 由標記為 `self-hosted, windows, x64, gpu, cuda12` 的受控 runner 建置。
Pull request 不會在 self-hosted runner 自動執行；一般 GPU 驗證只能由維護者手動啟動，
避免不受信任程式碼接觸內部機器。

## Release gate

1. 更新 `src/renewable_huber/_version.py`，並同步兩個 native `pyproject.toml` 的版本及
   精確 base dependency。
2. 把 `CHANGELOG.md` 的 Unreleased 內容移到該版本，確認授權與引用資訊。
3. 合併至 `main`，等待一般 CI、CPU wheel clean-install 與 GPU validation 通過。
4. 在本機先執行 metadata dry-run：

   ```powershell
   python scripts/native/validate_release_artifacts.py --source-only
   ```

5. 從 `main` commit 建立 `vX.Y.Z` tag。Release workflow 會拒絕版本不一致或不在
   `main` 歷史上的 tag。
6. Workflow 建置 base wheel/sdist、15 個 CPU wheels、3 個 CUDA wheels，並執行
   Twine、metadata、ABI capability、乾淨環境安裝和 GPU 測試。
7. 完整 artifact set 通過後才建立 GitHub Release。
8. 三個 PyPI publish jobs 分別等待對應 GitHub Environment 的人工核准，再透過
   Trusted Publishing/OIDC 發布；不儲存長效 API token。

Release tag 範例：

```powershell
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

## 安裝後驗證

CPU 使用者：

```powershell
python -m pip install renewable-huber-native-cpu==X.Y.Z
python -c "from renewable_huber import RenewableHuberRegressor; print(RenewableHuberRegressor(backend='native_cpu', n_jobs=-1))"
```

CUDA 12 使用者：

```powershell
python -m pip install renewable-huber-native-cuda==X.Y.Z
python -c "from renewable_huber import _native_cuda; print(_native_cuda.version()); print(_native_cuda.is_available())"
```

若 native distribution 已存在於 PyPI，禁止重新使用相同版本覆寫 wheel；修正版本與
changelog 後發布新的 patch release。

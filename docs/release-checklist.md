# 發布檢查清單

## 每次 release 前

- [ ] 目標版本尚未存在於 PyPI／TestPyPI，對應 Git tag 與 GitHub Release 也不存在。
- [ ] 本次發布使用 `0.7.0`（tag `v0.7.0`）；既有 tag（包括未成功發布的 `v0.6.0`）
      一律不移動、不重用。
- [ ] Base、native CPU、native CUDA 使用相同的 PEP 440 版本。
- [ ] Native distributions 精確依賴同版 `renewable-huber==X.Y.Z`。
- [ ] `CHANGELOG.md`、README、API、架構、支援矩陣、`SECURITY.md` 與
      `CITATION.cff` 的版本及日期已更新，repository version scan 無非歷史漂移。
- [ ] `run_test_profile.py --check`、required `core`／`performance`／`native-cpu`
      profiles 均實際執行，沒有以 optional skip 取代驗收。
- [ ] Golden corpus、Python 測試、Rust tests、Ruff、rustfmt、Clippy 全部通過。
- [ ] 固定硬體上的 CPU 與 CUDA performance gates 無效能或正確性回歸。
- [ ] 本機固定 CUDA 12 主機的 GPU correctness、C ABI smoke、shape sweep 與
      interleaved performance gate 已通過，並記錄 commit、環境與 JSON 證據。
- [ ] 固定主機已安裝 CUDA 版 PyTorch，`cuda` profile 的
      `PyTorchDlpackIntegrationTests` 實際通過而非 skip；TensorFlow CUDA DLPack
      若仍未在 WSL2／Linux GPU 主機驗證，支援矩陣須維持「尚未在裝置上驗證」。
- [ ] `python scripts/native/validate_release_artifacts.py --source-only` 通過。
- [ ] `release.yml` 的手動 build-only run 在精確 release candidate SHA 成功；不建立
      GitHub Release，也不寫入任何 package index。
- [ ] Release commit 已合併到 `main`，該 SHA 的一般 CI 全綠，tag 名稱為 `vX.Y.Z`。

## Artifact gate

- [ ] Base wheel 與 sdist 通過 `twine check` 與 clean-install smoke test。
- [ ] 20 個 CPU wheels 全數產生：Python 3.10–3.13 × 5 個 OS/architecture targets。
- [ ] 8 個 CUDA 12 wheels 全數產生：Python 3.10–3.13 × Windows x86-64 與
      Linux x86-64（`manylinux_2_28`）。
- [ ] CUDA wheels 的 `Requires-Dist` 含完整 `nvidia-*-cu12` runtime closure，且不
      vendor 任何 NVIDIA 函式庫。
- [ ] CUDA wheel 在無 GPU hosted runner 可乾淨安裝、載入，並回報正確公開 API
      version、native ABI 與 capability metadata。
- [ ] `cuobjdump` 證明 CUDA wheel 含 SM 75/80/86/89/90/120 SASS，且只有 SM 120
      PTX；使用者不需本機 `nvcc`。
- [ ] 完整 artifact set 沒有重複檔名，且版本與 dependency contract 全部相符。
- [ ] CPU wheels 與 matching base wheel 在 CI 乾淨環境可一起安裝並通過
      `pip check`；本機建置的 CUDA candidate wheel 已在固定 GPU 主機通過
      `smoke_test_cuda_wheels.py`。
- [ ] GitHub Release 包含 base wheel/sdist 及所有 native wheels。
- [ ] 正式 publish approval 前，已從 workflow 下載實際 CUDA release wheels，並在
      固定 GPU 主機執行 smoke，證據綁定 release SHA 與 artifact hashes。

## PyPI 發布

- [ ] `pypi`、`pypi-native-cpu`、`pypi-native-cuda` environments 都有 required reviewer。
- [ ] 三個 PyPI projects 的 Trusted Publisher 設定與 `release.yml` 完全一致。
- [ ] 維護者已人工核准三個 publish jobs。
- [ ] 從 PyPI 新環境安裝 base-only、native CPU、native CUDA 三種組合並執行 smoke test。
- [ ] 未嘗試覆寫已存在的版本；修正一律發布新的 patch version。
- [ ] 部分 package 已成功發布後，僅重跑失敗 jobs；不得整個 workflow 重跑已上傳版本。

詳細步驟與 runner/runtime 要求請見 [發布流程](release-process.md)。

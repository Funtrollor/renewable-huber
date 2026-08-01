# 發布檢查清單

## 每次 release 前

- [ ] Base、native CPU、native CUDA 使用相同的 PEP 440 版本。
- [ ] Native distributions 精確依賴同版 `renewable-huber==X.Y.Z`。
- [ ] `CHANGELOG.md`、README 支援範圍與 API 文件已更新。
- [ ] Golden corpus、Python 測試、Rust tests、Ruff、rustfmt、Clippy 全部通過。
- [ ] 固定硬體上的 CPU 與 CUDA performance gates 無效能或正確性回歸。
- [ ] 手動 GPU validation 已在 CUDA 12 self-hosted runner 通過。
- [ ] `python scripts/native/validate_release_artifacts.py --source-only` 通過。
- [ ] Release commit 已合併到 `main`，tag 名稱為 `vX.Y.Z`。

## Artifact gate

- [ ] Base wheel 與 sdist 通過 `twine check` 與 clean-install smoke test。
- [ ] 15 個 CPU wheels 全數產生：Python 3.10–3.12 × 5 個 OS/architecture targets。
- [ ] 3 個 Windows x86-64 CUDA 12 wheels 全數產生。
- [ ] CUDA wheel 具備公開 API version、native ABI 與 capability metadata。
- [ ] CUDA fat binary architecture 清單符合 release policy，使用者不需本機 `nvcc`。
- [ ] 完整 artifact set 沒有重複檔名，且版本與 dependency contract 全部相符。
- [ ] CPU/CUDA wheels 與 matching base wheel 在乾淨環境可一起安裝並通過 `pip check`。
- [ ] GitHub Release 包含 base wheel/sdist 及所有 native wheels。

## PyPI 發布

- [ ] `pypi`、`pypi-native-cpu`、`pypi-native-cuda` environments 都有 required reviewer。
- [ ] 三個 PyPI projects 的 Trusted Publisher 設定與 `release.yml` 完全一致。
- [ ] 維護者已人工核准三個 publish jobs。
- [ ] 從 PyPI 新環境安裝 base-only、native CPU、native CUDA 三種組合並執行 smoke test。
- [ ] 未嘗試覆寫已存在的版本；修正一律發布新的 patch version。

詳細步驟與 runner/runtime 要求請見 [發布流程](release-process.md)。

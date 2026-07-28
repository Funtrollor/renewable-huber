# GitHub 與 PyPI 發布檢查表

## 已完成的發布基礎

- [x] 使用 `src/renewable_huber/_version.py` 作為唯一版號來源。
- [x] 採用 Apache-2.0，並提供論文歸屬與獨立實作聲明。
- [x] 將 GitHub 連結指向 `Funtrollor/renewable-huber`。
- [x] 排除資料、模型、研究 PDF、legacy 程式與本機工具狀態。
- [x] 在 Windows、Linux、macOS 與 Python 3.10–3.12 執行 CI。
- [x] 建置並在乾淨環境 smoke-test wheel 與 sdist。
- [x] 提供手動、自架 NVIDIA runner 的 CuPy/CUDA 驗證工作流。
- [x] 建立 tag 驅動的 GitHub Release 與構件上傳工作流。
- [x] 建立 `CHANGELOG.md`、`CITATION.cff`、維護文件與 GitHub 模板。

## 第一次正式發布前

- [ ] 凍結公開 API，完成效能基準與論文／legacy golden-case 驗證。
- [x] 更新 `CHANGELOG.md` 的 `Unreleased` 項目並決定正式版號 `0.5.1`。
- [x] 在 `_version.py` 設定版號 `0.5.1`。
- [ ] 在具備 NVIDIA CUDA 的 runner 執行 GPU validation。
- [x] 確認 `renewable-huber` 在 PyPI 與 TestPyPI 尚未建立（2026-07-28）。
- [x] 設定 PyPI 與 TestPyPI Trusted Publishing。
- [x] 發布 `0.5.0` 至 TestPyPI，並在新的虛擬環境完成安裝與 fit/predict smoke test。
- [ ] 建立簽署的 `vX.Y.Z` tag；GitHub Actions 會驗證 tag 與套件版號一致。
- [ ] 人工檢查 GitHub Release 內的 wheel、sdist 與 release notes。
- [ ] 核准後才發布至正式 PyPI。

完整操作與失敗處理請見 [release-process.md](release-process.md)。

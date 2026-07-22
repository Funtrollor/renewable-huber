# GitHub 與 PyPI 發布檢查表

- [ ] 決定並加入開源授權條款；目前尚未指定授權。
- [ ] 確認 `renewable-huber` 在 PyPI 的名稱可用，或調整 `[project].name`。
- [x] 將 `pyproject.toml` 的 GitHub 連結改為實際 repository。
- [ ] 建立乾淨的 Git repository，確認資料庫、模型與 `data/` 未被加入版本控制。
- [ ] 在 Windows、Linux 與 macOS 的支援 Python 版本執行測試。
- [ ] 在具備 NVIDIA CUDA 的 runner 執行 `tests/test_cupy_backend.py`，比較 CPU/GPU 數值結果。
- [ ] 執行 `python -m build`，檢查 wheel 與 sdist 只含必要套件檔案。
- [ ] 先上傳 TestPyPI 並以新的虛擬環境安裝驗證。
- [ ] 建立 Git tag 與 GitHub release 後，再發布正式 PyPI 版號。

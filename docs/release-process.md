# 版本與 GitHub Release 流程

本專案目前只自動建立 GitHub Release，不會把任何構件上傳至 PyPI 或 TestPyPI。

## 版號

`src/renewable_huber/_version.py` 是唯一版號來源。`pyproject.toml` 透過 Hatch 的
dynamic version 讀取此檔案；不要在其他設定檔重複維護版號。

發版時：

1. 將 `_version.py` 的 `__version__` 更新為預定的 PEP 440 版號。
2. 把 `CHANGELOG.md` 的 `Unreleased` 內容移到對應版號與日期下。
3. 建立 Pull Request，等待必要 CI 通過並合併至 `main`。
4. 在最新 `main` 建立並推送 `vX.Y.Z` tag；`X.Y.Z` 必須與 `__version__` 完全相同。

建議使用簽署 tag：

```powershell
git switch main
git pull --ff-only
git tag -s vX.Y.Z -m "renewable-huber vX.Y.Z"
git push origin vX.Y.Z
```

## 自動化內容

`.github/workflows/release.yml` 會在 `v*` tag 推送後：

1. 驗證 tag 名稱與 `_version.py` 相同。
2. 執行完整單元測試、lint 與格式檢查。
3. 建置 wheel 與 sdist，並以 Twine 驗證 metadata。
4. 將構件保留為 GitHub Actions artifact。
5. 建立含自動產生 release notes 的 GitHub Release，附上 wheel 與 sdist。

任何步驟失敗時都不會建立 GitHub Release。修正應透過 Pull Request 進入 `main`，
刪除失敗 tag 後再以正確版號重建；不要在既有 release tag 上改寫歷史。

## PyPI（尚未啟用）

在專案準備完成前，不加入 API token，也不建立 `pypi` environment。未來應優先採用
PyPI Trusted Publishing（OIDC），並將 TestPyPI 與正式 PyPI 發布設為需要人工核准的
GitHub environment。啟用前必須完成 `release-checklist.md` 的所有項目。

# 版本與 GitHub Release 流程

本專案使用 GitHub Actions 與 PyPI Trusted Publishing（OIDC）發布，不保存長效 API
token。TestPyPI 與正式 PyPI 使用不同的 workflow 與 GitHub environment。

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
6. 由僅具 `id-token: write` 權限的獨立 job 將相同構件發布至正式 PyPI。

正式 PyPI job 必須等待建置驗證與 GitHub Release 成功，並通過 `pypi` environment 的
部署規則。修正應透過 Pull Request 進入 `main`，以新修補版號重新發布；已上傳 PyPI
的檔案與版號不可覆寫。

## 第一次設定 Trusted Publishing

在 PyPI 與 TestPyPI 分別建立 pending publisher，欄位如下：

| Registry | Workflow | GitHub environment |
| --- | --- | --- |
| PyPI | `release.yml` | `pypi` |
| TestPyPI | `test-pypi.yml` | `testpypi` |

共同欄位為 PyPI project name `renewable-huber`、owner `Funtrollor`、repository
`renewable-huber`。GitHub repository 也必須存在同名 environment；環境名稱是 OIDC
信任條件的一部分，大小寫與拼字必須完全相同。

## TestPyPI 驗證

合併發布設定後，在 GitHub Actions 手動執行 `TestPyPI` workflow（只允許從 `main`
執行）。發布成功後，在全新的虛擬環境執行：

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple/ renewable-huber==X.Y.Z
python -c "from renewable_huber import RenewableHuberRegressor; print(RenewableHuberRegressor())"
```

`--extra-index-url` 讓 NumPy 等相依套件仍從正式 PyPI 解析。確認 TestPyPI 的 wheel 與
sdist metadata、匯入與最小 fit/predict smoke test 後，才推送正式 tag。

## 失敗處理

- 建置或測試失敗：修正後以 Pull Request 合併，再建立新 tag。
- GitHub Release 成功但 PyPI 失敗：不要重建或移動既有 tag；先修正信任設定，再從
  GitHub Actions 重新執行失敗的 `publish-pypi` job。
- PyPI 已接受部分檔案：不要使用 `skip-existing` 隱藏不一致；檢查 registry 狀態並以
  新修補版號重新發布。

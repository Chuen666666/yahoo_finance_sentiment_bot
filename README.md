# Yahoo 股市 RSS 公司情緒分析工具

這是一個個人、非商業研究用的命令列工具與獨立 Web 應用。它讀取 Yahoo 股市官方 RSS，辨識內容中提到的台灣上市櫃公司，針對每家公司分別輸出：

- `positive`：偏利多
- `negative`：偏利空
- `neutral`：中立或資訊不足
- `uncertain`：未提及目標公司，或模型信心低於門檻，需人工複核
- 支援判斷的原文片段與字元位置

## 第一版範圍

- 只讀取 RSS 內原有的標題、摘要、出處、發布時間和連結。
- 不抓取 Yahoo 新聞全文、文章留言或個股留言板。
- 保留 Yahoo 與原媒體出處，不把資料重新發布成新聞資料庫。
- 資料結構已保留 `kind` 和 `parent_id`，未來取得合法授權的文章／留言來源後可新增 connector。

Yahoo 的 RSS 使用說明：<https://tw.stock.yahoo.com/rss-help>

## 執行環境

- Windows 10／11；使用 PowerShell 5.1 或 PowerShell 7。
- 64 位元 Python 3.10 以上；目前主要在 Python 3.14 驗證。
- 需要網路連線，才能讀取 Yahoo 股市 RSS，以及同步 TWSE／TPEx 公司資料。
- 網站預設使用本機 `127.0.0.1:8000`，不需要另外安裝資料庫服務。
- 新聞、公司清單、分析結果和自動追蹤設定會保存在 `data/yfsent.db`。
- 已訓練模型預設位於 `artifacts/model.joblib`。

先確認 Python 可以使用：

```powershell
python --version
```

## 第一次安裝

在 PowerShell 進入專案目錄並建立虛擬環境：

```powershell
cd "C:\Users\yu tsai\OneDrive\Desktop\coding\yahoo_finance_sentiment_bot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,web,dev]"
```

如果 PowerShell 不允許執行 `Activate.ps1`，可只對目前這個終端機暫時放行，再重新啟用：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

第一次建立資料庫時，先同步上市櫃公司資料：

```powershell
yfsent sync-entities
```

網站啟動前必須已有模型檔 `artifacts/model.joblib`。如果模型尚未建立，且已有完成的 `data/labels.csv`，執行：

```powershell
yfsent train --labels data/labels.csv --skip-finbert
```

移除 `--skip-finbert` 會額外評估 FinBERT，第一次執行時需要下載模型且會花較長時間。

若只想執行不需模型的 RSS、公司辨識與 CSV 功能，可安裝較小的開發環境：

```powershell
pip install -e ".[dev]"
```

## 日常啟動與關閉

### 啟動網站

每次開啟新的 PowerShell 後執行：

```powershell
cd "C:\Users\yu tsai\OneDrive\Desktop\coding\yahoo_finance_sentiment_bot"
.\.venv\Scripts\Activate.ps1
yfsent serve --refresh-minutes 30
```

看到以下訊息代表啟動成功：

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

保持這個 PowerShell 視窗開啟，並用瀏覽器進入：

- 網站首頁：<http://127.0.0.1:8000>
- API 文件：<http://127.0.0.1:8000/docs>
- 健康檢查：<http://127.0.0.1:8000/api/health>

如果不想先啟用虛擬環境，也可以直接執行其中的程式：

```powershell
.\.venv\Scripts\yfsent.exe serve --refresh-minutes 30
```

### 關閉網站

回到正在執行伺服器的 PowerShell，按：

```text
Ctrl+C
```

等待終端機顯示伺服器已關閉並回到命令提示字元。只關閉瀏覽器分頁不會停止伺服器；直接關閉該 PowerShell 視窗也會中止服務，但建議優先使用 `Ctrl+C` 正常結束。

若已啟用虛擬環境，可再執行以下指令退出：

```powershell
deactivate
```

停止網站不會刪除資料庫、模型、新聞或自動追蹤設定。下次啟動後會繼續使用原有資料。

### 重新啟動

修改程式、重新訓練模型或更新套件後，先按 `Ctrl+C` 關閉，再重新執行：

```powershell
yfsent serve --refresh-minutes 30
```

模型會在伺服器啟動時載入，因此重新訓練後必須重新啟動網站，才會使用新模型。

## 使用流程

### 1. 同步公司名稱與股票代碼

```powershell
yfsent sync-entities
```

資料來自：

- TWSE 上市公司基本資料 `t187ap03_L`
- TPEx 上櫃公司基本資料 `mopsfin_t187ap03_O`

如需補充常用別名，修改 `data/aliases.json`：

```json
{
  "2330": ["TSMC", "台灣積體電路"],
  "2317": ["Foxconn"]
}
```

### 2. 取得 RSS

股票代碼或可辨識的公司名稱會使用個股 RSS：

```powershell
yfsent fetch --query 2330 --limit 100
yfsent fetch --query 台積電 --limit 100
```

一般關鍵字會讀取官方財經分類 RSS，再於本機比對標題與摘要：

```powershell
yfsent fetch --query AI --limit 100
```

### 3. 建立並填寫標註表

```powershell
yfsent export-labels --query 台積電 --limit 500 --output data/labels.csv
```

以 Excel 或試算表填寫以下欄位：

- `sentiment`：只能填 `positive`、`negative` 或 `neutral`。
- `evidence_text`：直接複製支持判斷的原文；中立資料可以留空。
- `notes`：選填，用來記錄難以判定或語意相反的案例。

判斷對象是 `entity_name` 指定的公司，不是整篇新聞的語氣。例如「聯電失單，台積電受惠」對聯電是 `negative`，對台積電是 `positive`。

建議標註 300–500 筆，並盡量讓三個類別都有足夠樣本。

### 4. 訓練與比較模型

```powershell
yfsent train --labels data/labels.csv
```

訓練會：

1. 擷取目標公司所在子句及前後文，避免把大盤或其他公司的情緒套到目標公司。
2. 以 RSS 項目分組交叉驗證字元 n-gram TF-IDF 模型。
3. 由交叉驗證結果自動校準信心門檻；低於門檻的結果輸出為 `uncertain`。
4. 使用 `yiyanghkust/finbert-tone-chinese` 評估同一批資料。
5. 依 Macro-F1 選擇模型並儲存於 `artifacts/model.joblib`。
6. 將標註的證據片段整理至 `artifacts/evidence_lexicon.json`。
7. 把完整指標、混淆矩陣、類別分數與門檻政策輸出到 `artifacts/metrics.json`。

第一次使用 FinBERT 會下載模型。若只想先測 CPU 快速基線：

```powershell
yfsent train --labels data/labels.csv --skip-finbert
```

Windows 上第一次載入 `scikit-learn`／SciPy 可能需要一段時間；初次匯入完成後會明顯加快。

### 5. 分析並輸出 CSV

```powershell
yfsent analyze --query 台積電 --limit 50 --output data/output/tsmc.csv
```

若查詢的是公司，建議加上 `--target-only`，只輸出該公司，並把摘要中沒有實際提及該公司的項目列為 `uncertain`：

```powershell
yfsent analyze --query 台積電 --limit 50 --target-only --output data/output/tsmc.csv
```

預設使用訓練時校準的門檻。若想減少誤判、接受較多人工複核，可以自行提高門檻，例如：

```powershell
yfsent analyze --query 台積電 --limit 50 --target-only `
  --confidence-threshold 0.46 --output data/output/tsmc_conservative.csv
```

若只分析資料庫中已下載的 RSS：

```powershell
yfsent analyze --query 台積電 --limit 50 --offline
```

輸出為 UTF-8 BOM CSV，可直接用 Windows Excel 開啟。每列代表一個「RSS 項目 × 公司」。重要欄位如下：

- `raw_sentiment`：模型原始最高分的類別。
- `final_sentiment`／`sentiment`：套用門檻後的結果，可能為 `uncertain`。
- `review_required`：是否建議人工複核。
- `target_mentioned`：標題或摘要是否真的提及目標公司。
- `confidence`：模型對原始類別的信心分數。
- `evidence_text`、`evidence_start`、`evidence_end`：支持判斷的原文及位置。

### 6. 使用新聞情緒網站

依照「日常啟動與關閉」啟動伺服器，再以瀏覽器開啟 <http://127.0.0.1:8000>。網站目前支援：

- 以公司名稱、股票代碼或關鍵字搜尋已分析新聞。
- 依 `positive`、`negative`、`neutral`、`uncertain` 篩選。
- 按「抓取最新新聞」立即取得 RSS 並執行情緒分析。
- 把目前條件加入自動追蹤；伺服器運作期間每 30 分鐘更新一次。
- 顯示來源連結、目標公司、信心分數與模型判斷依據。

基本操作流程：

1. 在搜尋框輸入公司名稱、股票代碼或關鍵字，例如「台積電」或「2330」。
2. 按「抓取最新新聞」，從 RSS 取得新資料並立即分析。
3. 使用情緒選單篩選正面、負面、中立或待確認結果。
4. 按「追蹤目前條件」，讓伺服器依 `--refresh-minutes` 設定自動更新。
5. 點擊新聞標題前往 Yahoo 或原始媒體頁面閱讀全文。

自動更新只會在伺服器運作期間執行。關閉網站後追蹤條件仍會保留，但必須等下次啟動伺服器才會繼續更新。

可用 `--refresh-minutes 0` 停止背景更新，或用 `--port` 更改連接埠：

```powershell
yfsent serve --refresh-minutes 0 --port 8080
```

Web API 文件位於 <http://127.0.0.1:8000/docs>。主要端點為：

- `GET /api/articles`：查詢及篩選分析結果。
- `POST /api/refresh`：抓取並分析指定條件的最新 RSS。
- `GET/POST/DELETE /api/watchlist`：管理自動追蹤條件。
- `GET /api/health`：確認模型版本與排程設定。

目前預設只監聽 `127.0.0.1`，適合本機驗收。尚未加入登入、流量限制與公開部署設定，不應直接暴露到網際網路。

## 常見問題

### 找不到 `yfsent` 指令

通常是虛擬環境尚未啟用。請先執行：

```powershell
.\.venv\Scripts\Activate.ps1
```

也可以改用模組方式執行：

```powershell
.\.venv\Scripts\python.exe -m yfsent serve --refresh-minutes 30
```

### 顯示找不到模型

確認 `artifacts/model.joblib` 存在；若不存在，使用已標註資料訓練：

```powershell
yfsent train --labels data/labels.csv --skip-finbert
```

### 顯示公司資料庫是空的

執行公司資料同步後再重試：

```powershell
yfsent sync-entities
```

### 連接埠 8000 已被使用

改用其他連接埠，例如：

```powershell
yfsent serve --refresh-minutes 30 --port 8080
```

接著開啟 <http://127.0.0.1:8080>。

### RSS 抓取失敗

先確認網路連線，稍後再按一次「抓取最新新聞」。工具只使用 Yahoo 官方 RSS，不會改抓新聞網頁；如果官方 RSS 暫時無法使用，會保留資料庫中已取得的舊資料。

## 測試

測試不會連線至 Yahoo、TWSE、TPEx 或 Hugging Face：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 已知限制

- RSS 只涵蓋 Yahoo 當下提供的近期項目，並不是歷史新聞庫。
- 現成 FinBERT 使用的語料與台灣繁體財經新聞不同，因此一定要看本專案標註資料的實測結果。
- 300–500 筆適合建立基線，不足以涵蓋所有公司、產業、反諷與複雜因果語句。
- 情緒分數是文字分類結果，不是股價預測或投資建議。

## 擴充訓練資料

輸出新的 Yahoo RSS 標註候選時，可排除已標過的項目，並用多個關鍵字縮小範圍：

```powershell
yfsent export-labels `
  --exclude-labels data/labels.csv `
  --keyword 虧損 --keyword 下修 --keyword 跌停 `
  --output data/labels_candidates_negative.csv
```

標完後可原子化去重合併；若同一個 `item_id/entity_code` 出現不同情緒，指令會停止而不覆寫輸出：

```powershell
yfsent merge-labels `
  --input data/labels.csv `
  --input data/labels_candidates.csv `
  --output data/labels.csv
```

FinChina-SA 提供簡體中文金融新聞的機構級情緒標註，來源為公開 GitHub、授權為
Apache-2.0。它只適合作為輔助訓練資料；Yahoo 人工標註仍是交叉驗證與模型選擇的
依據。匯入器要求機構名稱實際出現在內文，並對每類做可重現的數量上限：

```powershell
yfsent import-finchina --max-per-class 600
yfsent train `
  --labels data/labels.csv `
  --aux-labels data/auxiliary/finchina_train.csv `
  --skip-finbert
```

外部資料不會加入驗證折，因此 `metrics.json` 的分數仍反映 Yahoo 標註資料；
`auxiliary_samples` 與 `auxiliary_class_counts` 會另外記錄實際加入訓練的數量。
應分別訓練 Yahoo-only 與輔助資料版本，只有 Yahoo 驗證集指標確實改善時才採用外部資料。

## 授權

本專案程式碼採用 [MIT License](LICENSE) 授權。

Yahoo、TWSE、TPEx、FinChina-SA，以及新聞原始媒體的內容、商標與資料，仍分別受其自身授權條款及權利規範約束，不因本專案採用 MIT License 而改變。

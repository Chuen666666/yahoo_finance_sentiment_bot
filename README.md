# Yahoo 股市 RSS 公司情緒分析工具

這是一個個人、非商業研究用的命令列工具。它讀取 Yahoo 股市官方 RSS，辨識內容中提到的台灣上市櫃公司，針對每家公司分別輸出：

- `positive`：偏利多
- `negative`：偏利空
- `neutral`：中立或資訊不足
- 支援判斷的原文片段與字元位置

## 第一版範圍

- 只讀取 RSS 內原有的標題、摘要、出處、發布時間和連結。
- 不抓取 Yahoo 新聞全文、文章留言或個股留言板。
- 保留 Yahoo 與原媒體出處，不把資料重新發布成新聞資料庫。
- 資料結構已保留 `kind` 和 `parent_id`，未來取得合法授權的文章／留言來源後可新增 connector。

Yahoo 的 RSS 使用說明：<https://tw.stock.yahoo.com/rss-help>

## 安裝

專案可使用 Python 3.10 以上版本。PowerShell 範例：

```powershell
cd yahoo_finance_sentiment_bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
```

若目前只想測試 RSS、公司辨識和 CSV，不需先安裝大型模型：

```powershell
pip install -e ".[dev]"
```

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
- `evidence_text`：直接複製支援判斷的原文；中立資料可以留空。
- `notes`：選填，用來記錄難以判定或語意相反的案例。

判斷對象是 `entity_name` 指定的公司，不是整篇新聞的語氣。例如「聯電失單，台積電受惠」對聯電是 `negative`，對台積電是 `positive`。

建議標註 300–500 筆，並盡量讓三個類別都有足夠樣本。

### 4. 訓練與比較模型

```powershell
yfsent train --labels data/labels.csv
```

訓練會：

1. 以 RSS 項目分組交叉驗證字元 n-gram TF-IDF 模型。
2. 使用 `yiyanghkust/finbert-tone-chinese` 評估同一批資料。
3. 依 Macro-F1 選擇模型並儲存於 `artifacts/model.joblib`。
4. 將標註的證據片段整理至 `artifacts/evidence_lexicon.json`。
5. 把完整指標、混淆矩陣與類別分數輸出到 `artifacts/metrics.json`。

第一次使用 FinBERT 會下載模型。若只想先測 CPU 快速基線：

```powershell
yfsent train --labels data/labels.csv --skip-finbert
```

Windows 上第一次載入 `scikit-learn`／SciPy 可能需要一段時間；初次匯入完成後會明顯加快。

### 5. 分析並輸出 CSV

```powershell
yfsent analyze --query 台積電 --limit 50 --output data/output/tsmc.csv
```

若只分析資料庫中已下載的 RSS：

```powershell
yfsent analyze --query 台積電 --limit 50 --offline
```

輸出為 UTF-8 BOM CSV，可直接用 Windows Excel 開啟。每列代表一個「RSS 項目 × 公司」，並包含模型、信心分數、證據文字及原文起訖位置。

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

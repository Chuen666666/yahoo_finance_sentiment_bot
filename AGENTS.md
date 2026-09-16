# Agent Guide

本文件適用於 `yahoo_finance_sentiment_bot` 專案下的所有檔案。修改前先閱讀本文件與 `README.md`。

## 專案目標

這是一個個人、非商業研究用的 Python CLI：讀取 Yahoo 股市官方 RSS，辨識其中提到的台灣上市櫃公司，針對每家公司分別判斷市場語意為 `positive`、`negative` 或 `neutral`，並輸出支持判斷的原文片段。

第一版只處理 RSS 提供的標題與摘要。除非使用者明確提供合法授權的資料來源，否則不得新增 Yahoo 新聞全文、文章留言或個股留言板爬蟲，也不得規避登入、robots、流量限制或其他存取控制。

## 開發環境

- Python 3.10 以上；目前主要驗證環境為 Windows、Python 3.14。
- PowerShell 虛擬環境：`.venv`。
- 核心資料管線盡量使用標準函式庫；`torch`、`transformers`、`scikit-learn` 等套件必須維持延遲載入，避免一般 RSS 指令被大型 ML 相依套件阻塞。
- 不提交 `.venv`、SQLite 資料庫、下載的模型、訓練輸出或使用者標註資料。

安裝與驗證指令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[ml,dev]"

python -m ruff check src tests
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## 主要模組

- `src/yfsent/cli.py`：命令列介面與流程協調。
- `src/yfsent/feeds.py`：Yahoo 官方 RSS 取得、解析、清理與去重。
- `src/yfsent/entities.py`：TWSE／TPEx 公司資料同步、別名與實體辨識。
- `src/yfsent/database.py`：SQLite schema 與資料存取。
- `src/yfsent/training.py`：標註驗證、交叉驗證、模型比較與 artifact 輸出。
- `src/yfsent/sentiment.py`：TF-IDF 與 FinBERT 推論介面。
- `src/yfsent/text.py`：目標公司上下文、斷句與證據片段萃取。
- `src/yfsent/labeling.py`：UTF-8 BOM 標註／分析 CSV。
- `tests/fixtures/`：禁止連線外部網站的固定測試資料。

## 不可破壞的行為

1. 情緒判斷單位是「RSS 項目 × 公司」，不是整篇文字只產生一個標籤。
2. 同一句文字可對不同公司產生不同情緒，例如「甲公司失單，乙公司受惠」。
3. `evidence_text` 必須是未正規化原文中的精確子字串；`evidence_start`、`evidence_end` 必須能切回完全相同的文字。
4. 模型可使用正規化文字推論，但儲存與輸出必須保留原始繁體中文。
5. 中立結果不得強行產生情緒證據。
6. 股票代碼必須使用完整邊界匹配；重複或歧義別名不得任意指定公司。
7. CSV 維持 UTF-8 BOM，確保 Windows Excel 可直接開啟。
8. RSS 項目必須保留原始 URL、出處與發布時間，不得把衍生結果描述成 Yahoo 或原媒體的判斷。

## 資料與網路規則

- Yahoo 僅使用官方 RSS 入口；個股 RSS 失效時應清楚報錯或退回分類 RSS 本機篩選，不得改為頁面爬蟲。
- 公司清單只使用 TWSE 與 TPEx 官方 OpenAPI；外部資料失敗時不得以虛構資料補齊。
- 網路測試必須是明確標示的 smoke test；一般單元測試不得連網。
- 所有請求需保留 timeout、有限重試與清楚的 User-Agent，不得做高頻輪詢。
- Python 3.14 的 SSL 相容處理只可取消 `VERIFY_X509_STRICT`；不得停用 CA 或主機名稱驗證。
- 任何新聞文字、人工標註或模型輸出都視為研究資料，不得寫入原始碼或測試 fixture，除非是人工製作且不含個資的最小測試句。

## 模型與標註規則

- 合法標籤只有 `negative`、`neutral`、`positive`。
- 訓練／驗證切分必須以 `item_id` 分組，避免同篇新聞不同公司落入不同資料集造成洩漏。
- 新增模型時必須與現有基線比較 Macro-F1，並輸出各類別 precision、recall、F1 與混淆矩陣。
- 不得只用 accuracy 宣稱模型改善。
- 不得把情緒分類描述為股價預測、交易訊號或投資建議。
- 變更 FinBERT label mapping 前，必須以模型設定與測試案例確認 `LABEL_0/1/2` 對應關係。
- 模型檔只能從本專案自己產生或可信模型來源下載；不要載入不明來源的 `joblib`／pickle。

## 修改原則

- 優先修改既有模組，不要建立功能重疊的新入口或第二套資料模型。
- 對外 CLI 參數、CSV 欄位或 SQLite schema 有變更時，同步更新 README、測試與必要的遷移／相容處理。
- 錯誤訊息使用清楚的繁體中文，技術欄位與固定標籤維持英文。
- 保持型別提示與小型純函式；外部 I/O 和文字／模型邏輯應分離，方便離線測試。
- 不修改工作區內與本專案無關的檔案。

## 完成條件

任何功能變更完成前至少確認：

```powershell
python -m ruff check src tests
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

若修改 RSS 或官方 API connector，另做一次低頻 live smoke test，並保留離線 fixture 測試。交付時說明測試結果、尚未驗證的外部相依，以及是否新增或改變使用者資料格式。

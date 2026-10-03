# ALPHA/TW BALANCED 13.4

台股「盤後選股 + 盤中重排」研究工具。V13.4 延續 V13.3 的 benchmark-aligned historical analog、Rank-first、盤中雷達與手機優先介面，並針對實際推薦名單校正三個失真來源：短線追價、單月營收尖峰、以及中長線對估值／獲利品質／資料完整度不足的處理。

## V13.4 核心校正

- **短線不再只獎勵衝刺**：降低 5/20 日單純動能、爆量與突破的總權重，增加趨勢持續性、風險品質及既有不追價機制。
- **營收看持續性**：公司分數同時看最新 YoY、近 3 月平均、近 6 月中位數、6 個月正成長比例與成長加速度；單月 YoY/MoM 異常暴增會做 spike penalty，不再直接視為穩定轉折。
- **獲利品質獨立計分**：近四季 EPS 正值比例、4Q EPS 合計、近兩季相對前兩季改善、毛利率趨勢與波動共同構成 quality score。
- **長線估值真正進入模型**：PE 採成長調整後的合理區間，不用單一固定 PE 門檻；高成長可容許較高估值，但極高估值會實質降低長線分數。
- **基本面紅旗防護**：持續營收大幅衰退、近四季 EPS 合計為負、最新毛利率為負等，會對中／長線加上明確 guardrail；短線仍保留交易機會，不用硬刪除。
- **金融業使用不同方法**：金融股不再以製造業的月營收／毛利率邏輯評分，主要看 EPS 品質與估值。
- **流動性只做 tie-breaker**：20 日平均成交金額占最終權重 3%–7%，讓同分時偏向較容易執行的股票，但不會把模型變成大型權值股排行。
- **TOP 5 軟性分散**：原始分數完全保留；同產業重複出現時只做 0.8–2 分的小幅 shortlist 調整，分數差距大的股票仍可同時入選。
- **缺資料不再完全無代價**：Full model 若缺基本面／歷史案例／法人證據，仍重新正規化，但會附加小幅 evidence penalty，避免資料不足的新股和證據完整股票被視為同樣可靠。


## 介面

V13.3 不再使用大面積漸層 Hero 與一張張圓角卡片。首頁改成市場帶 + 排名列 + 展開詳情：

- 排名列只保留：名次、股票、細產業、價格/漲跌、總評、最多 2 個理由。
- 營收、EPS、毛利率、本益比、三大法人、券商分點代理與 K 線放在展開區。
- 手機 K 線預設只顯示 K 棒、20 日線、60 日線、成交量；日期最多 3 個刻度。
- 專業分數留在模型內，不在主畫面大量曝光。
- 早期三層金字塔概念保留為「核心型 / 成長型 / 短打型」；S1–S5 改成「基本面轉折 / 突破 / 趨勢成形 / 偏離過大 / 過熱」等白話階段，僅作理解提示，不硬改排名。

## 模型架構

### 1. 盤後核心

- 全上市櫃建立流動性候選池，不以單一硬門檻把推薦清空。
- 技術結構、相對大盤、歷史相似走勢、營收/EPS/毛利率、三大法人依持有時間重新加權。
- 10 / 40 / 120 交易日分開估計。
- 歷史相似案例採 walk-forward-safe 設計：每個歷史時點使用「當時」的加權指數資料；不再用今天的大盤報酬套到所有歷史日期。
- 相似案例做時間去重，降低同一波行情被相鄰交易日重複計數的問題。
- 正報酬比例採 Beta(2,2) 平滑，避免小樣本看起來過度精準。

### 2. 盤中雷達

即時來源為 FinMind `taiwan_stock_tick_snapshot`（Sponsor）。盤中約每 60 秒刷新前一日/盤後模型挑出的候選池，使用：

- 相對加權/櫃買指數強弱
- 是否守在盤中均價之上
- 量比
- 當日高低位置
- 開盤後強弱
- 買賣盤壓力（低權重）
- 成交金額
- 過熱與價差扣分

盤中資訊只佔最終排名的一部分：短線 45%、中線 15%、長線 5%。沒有 Sponsor 即時權限、資料日期落後或 API 失敗時，系統不會以盤後資料冒充即時行情。

## 資料

- 歷史日線：Yahoo Finance（`auto_adjust=True`），SQLite 快取。
- 最新正式盤後：TWSE / TPEx 官方資料覆蓋 Yahoo 最新一根；官方 OpenAPI 失敗時有日期指定備援。
- 公司/籌碼：FinMind v4（月營收、財務報表、PER、三大法人）。
- 盤中：FinMind Sponsor 即時快照。
- 「主力」沒有官方單一分類；本專案只在啟用 Sponsor 券商分點資料時顯示「券商分點主力代理」，不以三大法人冒充主力。

## Streamlit Cloud

1. 將本資料夾所有檔案放在同一個 GitHub repository 根目錄。
2. Streamlit Community Cloud 入口選 `app.py`。
3. 在 App 的 Secrets / 環境變數設定：

```text
FINMIND_TOKEN=你的_token
ENABLE_BRANCH_FLOW=0
```

若有 FinMind Sponsor 券商分點權限且希望顯示主力代理，再把 `ENABLE_BRANCH_FLOW=1`。

## 本機執行

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows 也可執行 `INSTALL_AND_RUN.bat`。

模型需求與本版實作的逐項對照見 `MODEL_DESIGN.md`。

## Debug

```bash
python self_test.py
```

離線測試涵蓋：

- benchmark-aligned historical analog
- 10/40/120 日模型與交易區間
- 100 組長歷史序列壓力測試（僅為程式壓測，不宣稱是真實市場績效）
- 月營收 / EPS / 毛利率 / PER / 法人解析
- 盤中分數、過熱扣分、盤中重排
- 10/1 正式盤後資料備援情境
- 細產業與白話總評

## 100 檔真實台股 Walk-forward 驗效

```bash
python validate_100.py --stocks 100 --period 8y --points 6
```

固定亂數 seed 從當下 TWSE/TPEx 真實股票母體抽取具足夠歷史資料的 100 檔，選取共同歷史切點；每個切點讓 100 檔在同一天橫向排名，每次預測只提供截止該日以前的資料，再讀取後續 10/40/120 交易日實現報酬。輸出位於：

```text
data/validation_v133/sample_100.csv
data/validation_v133/walkforward_observations.csv
data/validation_v133/validation_report.json
data/validation_v133/validation_report.md
```

驗效主要看：同一歷史日期的橫向 Spearman rank IC、當日前 20% 分數組相對同日母體的中位報酬提升、正報酬比例提升、相對加權指數報酬，以及 P10。

此驗效刻意只測「價格 + 大盤歷史相似核心」。目前公司基本面與法人資料沒有做完整的 point-in-time 歷史快照回填，因此不把今天知道的財報/法人資訊灌回過去，以免產生 look-ahead bias。


### GitHub 一鍵執行真實 100 檔驗效

專案附 `/.github/workflows/validate-100.yml`。上傳 GitHub 後可在 **Actions → Validate 100 real Taiwan stocks → Run workflow** 手動執行；完成後會產生 `alpha-radar-v133-validation` artifact。這個 workflow 不需要 FinMind token，驗的是價格/大盤核心。

### 本次封裝環境的驗效狀態

V13.3 封裝時已實際啟動上述 100 檔驗效程式，但此執行環境封鎖外部 DNS，TWSE / TPEx 股票母體與 Yahoo 歷史行情無法下載；程式只取得內建 8 檔備援名單後即中止。因此**沒有**以合成資料冒充 100 檔真實績效，也不宣稱本封裝已完成真實市場績效驗證。完整狀態見 `VALIDATION_STATUS.md`。


## V13.3.2 UI hotfix
- 修正 Streamlit 頂部工具列覆蓋內容：恢復安全的 top padding。
- 手機額外加入 `safe-area-inset-top`，避免瀏海／動態島與瀏覽器工具列遮住品牌列。
- 不改模型、排名、資料或回測邏輯；既有 V13.3 快照可直接沿用。


## V13.3.2 changes

- Short-horizon anti-chasing correction: the no-chase ceiling is now anchored to breakout / moving-average structure instead of current price, so `DO_NOT_CHASE` can actually trigger on a fresh scan.
- Stage-4/Stage-5 overextension now feeds back into technical ranking as a smooth penalty instead of being display-only.
- Entry quality below 70 and `DO_NOT_CHASE` apply modest horizon-specific ranking penalties; rank-first is retained, so the list never disappears because of a single hard gate.
- Warm updates reuse the existing multi-year SQLite history and refresh full Yahoo history periodically / when coverage is insufficient. Official TWSE/TPEx EOD still supplies the newest complete session.
- Universe prefilter reads only the most recent 140 bars; full history is loaded only for deep candidates.
- Progress now reports Yahoo batch counts, TWSE/TPEx/index status, historical-comparison counts, and company-data counts.
- Monthly revenue and quarterly financial statements use longer cache TTLs because they do not change intraday; institutional flow remains short-TTL.

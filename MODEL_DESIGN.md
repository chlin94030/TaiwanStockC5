# Alpha Radar V13.3 — 模型需求對照

V13.3 的目標不是預測「一定會漲」，而是把全市場股票依不同持有時間做相對排序，並把盤中訊號限制在已通過盤後模型的候選池內。

| 對話需求 | V13.3 實作 |
|---|---|
| 全上市櫃、不因條件太嚴而沒有推薦 | 先做最低資料品質/流動性檢查，再 rank-first；TOP 5 永遠從可用候選中排序，不用單一硬門檻清空結果 |
| 大盤 Regime | 加權指數：收盤 ≥ 20MA 且 20MA ≥ 60MA = 偏多；收盤 <20MA 但 ≥60MA = 中性；其餘偏弱 |
| 5/20/60/120/240MA | 全部進入價格特徵；主畫面只顯示必要線條，避免手機圖表擁擠 |
| ATR14 | True Range ATR，用於風險、停損及追價距離 |
| Bollinger | 20日 ±2σ，納入歷史相似狀態；不在預設手機 K 線塞滿圖例 |
| KD(9) | K/D 與交叉方向納入短中期 timing；不是硬門檻 |
| MACD(12,26,9) | Histogram 方向納入 timing / 技術品質 |
| Stage 1–5 | UI 改為白話：基本面轉折、突破、趨勢成形、偏離過大、過熱 |
| Core / Next Compounder / Tactical | UI 改為核心型 / 成長型 / 短打型；沿用原始加分邏輯，但改用可計算的領先、穩定、營收/EPS、突破、量能等代理值 |
| Entry base 70 | 進場品質從 70 起算；突破/回測/洗盤後站回、KD/MACD 改善加分，過度乖離扣分 |
| ATR/MA 停損 | 每個週期提供較理想區、轉強確認、不追價、跌破重看價 |
| 倉位依 Regime | 偏多/中性/偏弱依短中長週期給「一般部位/稍微保守/縮小部位/小部位」節奏提示 |
| 營收/財報/估值 | 最新月營收億元、MoM、YoY、近4季 EPS/累積、毛利率、PER |
| 籌碼 | 近一月外資、投信、自營商、三大法人；券商分點主力代理只有權限可用時才顯示 |
| 盤中推薦 | FinMind Sponsor 即時快照；盤後模型先選候選池，盤中短/中/長只以 45%/15%/5% 權重重排；急漲與價差過大會扣分 |
| 歷史驗效 | benchmark-aligned historical analog；每個歷史點使用當時大盤，鄰近案例去重；另有 100 檔真實台股同日橫向 walk-forward 驗效程式 |
| 手機介面 | 市場帶 + 排名列 + 展開詳情；主畫面不堆技術術語，K 線預設只保留 K 棒、20/60日線與量 |

## 為什麼盤中不取代盤後模型

盤中資訊很快、也很吵。V13.3 先由多年日線、公司資料與籌碼決定「值得追蹤的股票」，盤中只處理今天的相對強弱、量能與價格位置。這可降低單根急拉、瞬間大量或買賣盤雜訊把中長期排序整個翻掉的風險。

## 驗效邊界

`validate_100.py` 只驗證可做 point-in-time 控制的價格/大盤核心。財報、月營收與法人若要加入歷史回測，必須保留每個歷史日期當時已公開的版本與發布時點；V13.3 不會把今天已知的資料倒灌回歷史，以免製造 look-ahead bias。


## V13.3.2 ranking correction

The rank-first philosophy remains. Short-term momentum is still rewarded, but >12% / >20% distance from MA20 and very hot 5-day/RSI readings receive a smooth anti-chasing penalty. `DO_NOT_CHASE` is now structurally reachable because the chase ceiling no longer moves upward with the current close.

## V13.4 Balanced Evidence calibration

V13.4 不是把另一套模型的股票名單當答案，而是用兩套模型的分歧找失真來源：

1. **短線熱股偏誤**：技術層降低近期衝刺／爆量的支配力，保留 RS、趨勢與突破，但提高 risk quality，並沿用 MA20/ATR 不追價機制。
2. **單月尖峰不是中線趨勢**：營收由「最新一月」擴成 3–6 月持續性、正成長比例與加速度；異常尖峰會降權。
3. **中長線必須讓負基本面有實質影響**：若資料明確顯示 EPS、毛利及營收同步惡化，價格再強也會被中／長線 guardrail 降分。
4. **品質、估值、流動性與分散**：長線 company evidence 權重提高至 42%；估值採成長調整，成交金額僅為小幅 execution-quality factor；TOP 5 以 soft diversification 避免近似分數全部集中同一產業。
5. **金融業分流**：金融業不以製造業營收／毛利率框架評估，避免會計表達差異造成假訊號。

盤後最終權重：

| 週期 | 技術 | 歷史相似案例 | 公司證據 | 法人/籌碼 | 流動性 |
|---|---:|---:|---:|---:|---:|
| 短線 | 45% | 20% | 10% | 18% | 7% |
| 中線 | 36% | 22% | 27% | 10% | 5% |
| 長線 | 28% | 22% | 42% | 5% | 3% |

這些權重仍屬研究模型參數，最終應以 `validate_100.py` 的真實 walk-forward 結果持續校正，不能因單次推薦名單「看起來合理」就視為已證實最優。

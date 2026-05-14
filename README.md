# 香港跑馬仔麻雀 RL 訓練平台

本專案是本地單機桌面程式，核心已全面轉為 PyTorch Actor-Critic + Replay Buffer + Self-play 的 RL 模型路線。舊的 JSON / LinearDiscardModel / CLI 訓練流程已棄用，所有日常操作都集中在 PySide6 GUI。

## 目前功能

- 香港跑馬仔麻雀 4 人牌局環境
- 136 張牌，不使用花牌
- 胡牌判定：標準 4 面 1 對、七對子
- 支援摸牌、棄牌、食、碰、明槓、食糊、自摸、流局
- PySide6 GUI：遊戲、訓練、模型管理、設定分頁
- PyTorch Actor-Critic 模型：shared encoder + policy head + value head
- Replay Buffer + Self-play：四個玩家模型互相對打並同時訓練
- GUI 背景訓練：可啟動、暫停、繼續、停止並保存
- 即時訓練進度、勝率、流局率、平均局長、policy/value loss 曲線
- GUI 模型管理：掃描 `models/` 目錄、載入 `.pt` 模型、顯示 metadata

## 安裝

Python 3.10+ 建議。

```powershell
python -m pip install -r requirements.txt
```

## 執行

```powershell
python main.py
```

或：

```powershell
python main.py gui
```

## GUI 分頁

### 遊戲

- `玩家對 AI`：你坐 P1，點選底部手牌後按 `出牌`。
- `AI 對 AI`：四家自動對局，可觀察模型表現。
- `單步`：推進一個 AI 行動。
- `自動/暫停`：切換自動對局。
- `自摸`：只有玩家可自摸時啟用。
- `載入最新模型`：從 `models/` 載入最新 `.pt` checkpoint。

如果 `models/` 沒有 `.pt` 模型，系統會提示先到「訓練」分頁建立模型；遊戲仍可用 heuristic fallback 讓介面不失效。

### 訓練

可在 GUI 中設定並執行 RL 訓練：

- 輸出模型路徑，例如 `models/rl_latest.pt`
- 繼續訓練來源，可留空從零開始
- Episodes
- Batch size
- Replay capacity
- Learning rate
- Opponent pool 機率
- UI 更新間隔

訓練在背景 thread 執行，不會凍結 UI。面板會即時顯示：

- 當前 episode / 總 episodes
- 進度百分比
- 勝率
- 流局率
- 最近局勝率
- 平均局長
- policy loss / value loss 折線圖

訓練完成或停止後會保存 checkpoint，並自動載入成遊戲 AI。

### 模型管理

- 掃描 `models/` 下所有 `.pt` 模型
- 選取模型並載入到遊戲
- 顯示版本號、訓練局數、玩家模型數、state/action 維度等 metadata

### 設定

目前顯示本地單機與規則設定摘要。

## RL 架構

### State Encoding

`hkmahjong_ai/rl_encoder.py` 會把 observation 編碼成固定長度 tensor，包含：

- 手牌計數：34 dim
- 四家牌河計數：4 x 34 dim
- 四家副露數
- 四家副露牌種計數：4 x 34 dim
- 剩餘牌牆數 normalized
- 當前玩家座位 one-hot：4 dim
- 輪數進度 normalized

### Action Space

- `0..33`：打出對應牌
- `34`：副露 pass
- `35`：食
- `36`：碰
- `37`：槓

出牌與食碰槓都由 Actor 網絡透過合法 action mask 選擇。

### Training Loop

1. `HKMahjongEnv` 產生目前玩家 observation。
2. `encode_state()` 轉成 tensor。
3. 依出牌或副露情境產生 action mask。
4. Actor 輸出合法 action policy。
5. 執行 action，環境推進。
6. ReplayBuffer 儲存 `(state, action, reward, next_state, done, masks)`。
7. Critic 估計 `V(state)` 與 `V(next_state)`。
8. 使用 `reward + gamma * V(next_state)` 計算 target。
9. 使用 advantage 更新 policy loss，並用 MSE 更新 value loss。

Reward：

- 勝者：`+1.0`
- 放銃者：`-0.7`
- 其他輸家：`-0.15`
- 流局：`0.0`
- 另有少量 hand potential shaped reward 作初期引導

## 專案結構

```text
.
├── assets/                    # 牌面、牌桌、按鈕與 HUD 素材
├── docs/                      # 素材提示詞與 GUI smoke 截圖
├── hkmahjong_ai/
│   ├── ui/
│   │   ├── app.py             # 主 GUI 與分頁
│   │   ├── training_worker.py # 背景訓練 worker
│   │   ├── loss_chart.py      # 即時 loss 曲線
│   │   ├── model_manager.py   # 模型掃描與 metadata
│   │   └── table_view.py      # 牌桌渲染
│   ├── agents.py              # Random / Heuristic fallback
│   ├── env.py                 # 麻雀環境
│   ├── hand_eval.py           # 胡牌判定
│   ├── opponent_pool.py
│   ├── replay_buffer.py
│   ├── rl_agent.py
│   ├── rl_encoder.py
│   ├── rl_model.py
│   ├── rl_train.py
│   └── tiles.py
├── models/                    # .pt 模型輸出
├── tools/generate_assets.py
├── main.py
└── requirements.txt
```

## 素材

目前內建 SVG 佔位素材，可直接執行。若要重新生成：

```powershell
python tools/generate_assets.py
```

更高完成度素材生成提示詞請見 [docs/ASSET_PROMPTS.md](docs/ASSET_PROMPTS.md)。同名 PNG 會優先於 SVG 載入，可直接覆蓋素材而不改程式。

## 已知限制

- RL 從零開始會很弱，需要大量 self-play 才會有可用策略。
- 目前 action space 將「食」視為單一類型；若同時有多個食牌組合，環境會取第一個合法組合。
- 尚未實作暗槓、加槓、搶槓、番數、包牌。
- Loss chart 是輕量 Qt 自繪折線圖，不依賴 matplotlib 或 pyqtgraph。

## 下一步建議

- 將 chow 組合細分成更完整 action space。
- 加入模型評估面板，支援兩個 `.pt` 模型可視化對戰比較。
- 增加 training checkpoint 自動命名與最佳模型保存。
- 加入更穩定的 PPO clipped objective。

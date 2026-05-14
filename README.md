# 香港麻雀 AI 訓練平台

本專案是本地單機 Python 桌面程式，核心是香港麻雀 AI 的訓練、評估、模型版本比較與可視化對局。GUI 已改為 PySide6 牌桌式介面，保留既有訓練平台能力。

## 功能

- 香港麻雀基本 4 人牌局環境
- 支援摸牌、棄牌、吃、碰、食糊、自摸、流局
- PySide6 桌面 GUI，提供完整牌桌、四家座位、手牌、牌河、副露、狀態 HUD
- 圖像資源驅動 UI：牌面、牌背、牌桌、按鈕、面板、座位框位於 `assets/`
- 支援玩家手動遊玩、AI 對 AI 自動對局
- 支援自我對局訓練、模型儲存、模型載入、評估與版本比較
- 保留可替換更強模型的結構

## 專案結構

```text
.
├── assets/
│   ├── tiles/                 # 34 張麻雀牌面與牌背 SVG
│   └── ui/                    # 牌桌、按鈕、HUD、座位框 SVG
├── docs/
│   └── ASSET_PROMPTS.md       # ChatGPT image generation 提示詞
├── hkmahjong_ai/
│   ├── ui/                    # PySide6 GUI 分層
│   ├── agents.py
│   ├── compare.py
│   ├── env.py
│   ├── evaluate.py
│   ├── hand_eval.py
│   ├── model.py
│   ├── tiles.py
│   └── train.py
├── models/
├── tools/
│   └── generate_assets.py     # 重新生成預設 SVG 素材
├── main.py
└── requirements.txt
```

## 安裝

Python 3.10+ 建議。

```powershell
python -m pip install -r requirements.txt
```

## 執行 GUI

```powershell
python main.py gui
```

也可以直接：

```powershell
python main.py
```

## GUI 操作

- `玩家對 AI`：你坐 P1，點擊底部手牌出牌。
- `AI 對 AI`：四家全自動，會顯示所有手牌方便觀察。
- `單步`：推進一個 AI 行動。
- `自動/暫停`：切換自動對局。
- `自摸`：只有玩家可自摸時啟用。
- `訓練 200 局`：從目前模型路徑繼續訓練，或建立新模型。
- `評估 100 局`：快速評估目前模型。
- `載入模型`：載入 JSON 模型檔。

## 訓練模型

```powershell
python main.py train --episodes 1000 --out models/model_v1.json
```

從既有模型繼續訓練：

```powershell
python main.py train --episodes 1000 --model models/model_v1.json --out models/model_v2.json
```

## 評估模型

```powershell
python main.py evaluate --model models/model_v1.json --episodes 500 --opponent heuristic
```

## 比較兩個模型

```powershell
python main.py compare --a models/model_v1.json --b models/model_v2.json --episodes 500
```

## 素材說明

目前內建預設 SVG 素材，可直接執行。若要重新生成：

```powershell
python tools/generate_assets.py
```

若要替換成更高完成度的 AI 生成素材，請參考 [docs/ASSET_PROMPTS.md](docs/ASSET_PROMPTS.md)。

目前 GUI 預設載入：

- `assets/tiles/tile_00.svg` 至 `tile_33.svg`
- `assets/tiles/tile_back.svg`
- `assets/ui/table_background.svg`
- `assets/ui/button_green.svg`
- `assets/ui/status_panel.svg`
- `assets/ui/seat_frame.svg`

## 目前規則範圍

第一版仍聚焦 AI 訓練環境與可擴充架構：

- 34 種牌，每種 4 張
- 不含花牌
- 支援基本 4 面子 1 對子糊牌
- 支援吃、碰、放銃、自摸、流局
- 尚未完整實作槓、番種、包牌、花牌、複雜計分

後續可在 `hkmahjong_ai/env.py` 與 `hkmahjong_ai/hand_eval.py` 擴充完整香港計分與進階動作。

## 模型輸入輸出

模型目前是可訓練線性棄牌策略：

- 輸入：手牌計數、候選棄牌、門前/副露數、牌牆剩餘、牌種特徵、鄰張/孤張/對子等特徵
- 輸出：每個合法棄牌的分數
- 行動：選擇分數最高的棄牌，訓練時可加入探索
- 儲存：JSON 權重檔，可版本化比較

這個結構刻意保持簡單，方便之後替換成神經網絡、強化學習 replay buffer 或更完整的狀態編碼。

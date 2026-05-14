# 香港跑馬仔麻雀 AI 訓練平台

本專案是本地單機的香港跑馬仔麻雀 AI 訓練與桌面遊戲平台。核心路線已轉為 PyTorch Actor-Critic + Replay Buffer + Self-play，GUI 使用 PySide6，訓練、模型管理與對戰都可在主界面完成。

## 功能重點

- 香港跑馬仔麻雀基本流程，136 張牌，不使用花牌。
- 胡牌只判斷標準 4 面 1 對與七對子，不計番。
- PyTorch Actor-Critic 模型，四家各自持有模型 self-play。
- Replay Buffer、opponent pool、episode-level reward 回填。
- GUI 可觀看玩家對 AI、AI 對 AI、自動對局。
- GUI 可啟動、暫停、停止訓練，並即時顯示進度、統計與 loss 曲線。
- 模型使用 `.pt` checkpoint，包含版本、訓練局數、state/action 維度等 metadata。

## 安裝

```powershell
python -m pip install -r requirements.txt
```

需要 Python 3.10+。如要使用 GPU 訓練，請依照你的 CUDA 版本安裝相容的 PyTorch。

## 啟動 GUI

```powershell
python main.py
```

或：

```powershell
python main.py gui
```

## GUI 結構

- 遊戲：香港麻雀牌桌、四家座位、手牌、牌河、副露、狀態與操作按鈕。
- 訓練：設定輸出模型、載入既有模型、episodes、batch size、learning rate、entropy、opponent pool 機率、每局更新次數。
- 模型管理：掃描 `models/` 內可用 `.pt` 模型，顯示 metadata，載入為對戰 AI。
- 設定：保留後續擴充入口。

訓練會在背景 thread 執行，不會凍結 UI。訓練面板會即時更新：

- 當前 episode / 總 episodes
- 進度百分比
- 勝率、流局率
- 最近 N 局勝率、最近 N 局流局率
- 平均局長
- Policy loss / Value loss 折線圖
- 訓練設備 CPU / CUDA
- 已訓練總局數與儲存路徑

## RL 訓練流程

1. `HKMahjongEnv` 產生目前玩家 observation。
2. `encode_state()` 將手牌、牌河、副露、剩餘牌牆、座位、輪數、被打出的牌轉成 tensor。
3. Actor 根據合法 action mask 輸出出牌或食碰槓過的機率。
4. 環境執行 action，RLAgent 記錄 decision。
5. 每局結束後，將終局 reward 以折扣方式回填到該局所有 transition。
6. Replay Buffer 儲存 `(state, action, reward, next_state, done, masks)`。
7. 每局結束後執行多次 batch update，更新 policy loss 與 value loss。
8. 定期儲存 snapshot 到 opponent pool，降低 self-play collapse 風險。

Reward：

- 胡牌者：`+1.0`
- 放銃者：`-0.7`
- 其他未勝玩家：`-0.15`
- 流局：全員 `0.0`
- 可選 hand potential shaped reward 作為早期訓練引導。

## 模型相容性

目前 state encoder 維度為 `350`，action space 為 `38`：

- `0..33`：出牌
- `34`：pass
- `35`：chow
- `36`：pong
- `37`：kong

舊版 `state_dim=316` checkpoint 不再相容，請重新訓練或用 GUI 從零開始產生新 `.pt` 模型。

## 素材

素材放在 `assets/`。目前保留可替換皮膚結構，可用下列工具重新生成佔位素材：

```powershell
python tools/generate_assets.py
```

圖片生成提示詞整理在 [docs/ASSET_PROMPTS.md](docs/ASSET_PROMPTS.md)。

## 驗證

基本語法檢查：

```powershell
python -m compileall hkmahjong_ai main.py
```

短訓練 smoke test 可在 GUI 訓練面板設定少量 episodes 執行，確認 progress、loss chart 與 checkpoint metadata 正常更新。

## 已知限制

- 目前不是完整競技級麻雀規則，只覆蓋本專案指定的跑馬仔基本胡牌結構。
- Actor-Critic 為輕量自研實作，尚未加入 PPO clipped objective。
- 視覺素材仍以可替換資源為主，可再替換成更高品質 sprite sheet 與 HUD 圖像。

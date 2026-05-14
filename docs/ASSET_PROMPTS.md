# 視覺素材生成提示詞

目前專案已內建一組可執行的 SVG 預設素材，位置在 `assets/`。若要升級成更高完成度的美術資源，可用以下提示詞生成 PNG 或 SVG，再用相同檔名覆蓋現有素材。

## 麻雀牌面 Sprite

Create a clean high-quality Hong Kong mahjong tile set sprite sheet for a desktop game UI, front-facing tiles, ivory white tile body, glossy ceramic texture, traditional Chinese tile symbols, complete 34 tile faces, consistent lighting, minimal shadow, transparent background, polished game asset style, highly legible at small sizes, not photorealistic, premium casual game art

建議輸出：
- 單張 sprite sheet：`assets/tiles/mahjong_tiles_sprite.png`
- 或切成 34 張：`assets/tiles/tile_00.png` 至 `assets/tiles/tile_33.png`
- 牌背：`assets/tiles/tile_back.png`

## 綠色牌桌背景

Create a top-down Hong Kong mahjong table background for a desktop game interface, deep green felt table, elegant wooden trim, subtle warm ambient lighting, clean symmetrical layout, premium casual game style, no tiles, no characters, no text, no watermark, polished and modern with classic mahjong atmosphere

建議輸出：
- `assets/ui/table_background.png`

## UI 按鈕組

Create a set of polished desktop game UI buttons for a Hong Kong mahjong game, rounded rectangular buttons, green and gold color palette, subtle gloss, high readability, premium casual game style, labels area left blank, transparent background, consistent game UI kit

建議輸出：
- `assets/ui/button_green.png`

## 狀態面板

Create a polished status panel UI asset for a Hong Kong mahjong desktop game, dark green and gold frame, elegant Chinese-inspired detailing, clean center space for dynamic text and counters, premium game HUD style, transparent background

建議輸出：
- `assets/ui/status_panel.png`

## 玩家頭像框 / 座位框

Create decorative player seat frames for a Hong Kong mahjong game UI, top-down board game layout, elegant gold and dark green accents, clean readable framing for avatar, score, and seat wind, premium polished casual game interface, transparent background

建議輸出：
- `assets/ui/seat_frame.png`

## 替換規則

- 若替換成 PNG，請保持相同視覺比例。
- 牌面建議使用直式比例約 `128x176`。
- 桌面背景建議 `1600x1000` 或更高。
- 若使用 sprite sheet，需要後續在 `hkmahjong_ai/ui/assets.py` 增加裁切邏輯。
- 目前 GUI 預設載入 SVG；若放入同名 PNG，可擴充 `AssetManager` 的解析優先順序。

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TILES = ASSETS / "tiles"
UI = ASSETS / "ui"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def tile_svg(body: str, accent: str = "#0f4f3f") -> str:
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="128" height="176" viewBox="0 0 128 176">
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fffaf0"/>
      <stop offset="0.55" stop-color="#f2e7ce"/>
      <stop offset="1" stop-color="#d7c39c"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="5" stdDeviation="4" flood-color="#07140f" flood-opacity="0.28"/>
    </filter>
  </defs>
  <rect x="10" y="8" width="108" height="156" rx="16" fill="url(#tile)" filter="url(#shadow)"/>
  <rect x="16" y="14" width="96" height="138" rx="11" fill="#fffdf7" opacity="0.82"/>
  <rect x="22" y="20" width="84" height="126" rx="8" fill="none" stroke="{accent}" stroke-opacity="0.25" stroke-width="2"/>
  {body}
</svg>
"""


def text_tile(text: str, color: str, sub: str = "") -> str:
    extra = ""
    if sub:
        extra = f'<text x="64" y="132" text-anchor="middle" font-family="Georgia, serif" font-size="24" font-weight="700" fill="{color}">{sub}</text>'
    body = f"""
  <text x="64" y="90" text-anchor="middle" font-family="Microsoft JhengHei, Microsoft YaHei, MingLiU, SimSun, serif" font-size="62" font-weight="800" fill="{color}">{text}</text>
  {extra}
"""
    return tile_svg(body, color)


def wan_tile(n: int) -> str:
    numerals = "一二三四五六七八九"
    return text_tile(numerals[n - 1], "#b42020", "萬")


def dot_tile(n: int) -> str:
    positions = {
        1: [(64, 82)],
        2: [(46, 60), (82, 104)],
        3: [(44, 55), (64, 82), (84, 109)],
        4: [(42, 54), (86, 54), (42, 110), (86, 110)],
        5: [(42, 54), (86, 54), (64, 82), (42, 110), (86, 110)],
        6: [(42, 48), (86, 48), (42, 82), (86, 82), (42, 116), (86, 116)],
        7: [(42, 42), (86, 42), (64, 70), (42, 98), (86, 98), (42, 126), (86, 126)],
        8: [(42, 40), (86, 40), (42, 68), (86, 68), (42, 96), (86, 96), (42, 124), (86, 124)],
        9: [(38, 42), (64, 42), (90, 42), (38, 82), (64, 82), (90, 82), (38, 122), (64, 122), (90, 122)],
    }
    circles = []
    palette = ["#1f7a4d", "#b42020", "#245eac"]
    for i, (x, y) in enumerate(positions[n]):
        color = palette[i % len(palette)]
        circles.append(f'<circle cx="{x}" cy="{y}" r="11" fill="none" stroke="{color}" stroke-width="5"/>')
        circles.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    return tile_svg("\n  ".join(circles), "#245eac")


def bamboo_tile(n: int) -> str:
    if n == 1:
        body = """
  <path d="M64 36 C87 50 88 91 64 132 C40 91 41 50 64 36Z" fill="#1f7a4d" stroke="#0b3c31" stroke-width="3"/>
  <path d="M64 44 C58 68 58 96 64 122" fill="none" stroke="#f5d66b" stroke-width="4"/>
  <circle cx="64" cy="82" r="10" fill="#b42020"/>
"""
        return tile_svg(body, "#1f7a4d")
    positions = {
        2: [(52, 60), (76, 104)],
        3: [(46, 52), (64, 84), (82, 116)],
        4: [(48, 52), (80, 52), (48, 112), (80, 112)],
        5: [(48, 50), (80, 50), (64, 82), (48, 114), (80, 114)],
        6: [(46, 46), (82, 46), (46, 82), (82, 82), (46, 118), (82, 118)],
        7: [(46, 38), (82, 38), (64, 64), (46, 92), (82, 92), (46, 122), (82, 122)],
        8: [(46, 36), (82, 36), (46, 64), (82, 64), (46, 94), (82, 94), (46, 124), (82, 124)],
        9: [(38, 40), (64, 40), (90, 40), (38, 82), (64, 82), (90, 82), (38, 124), (64, 124), (90, 124)],
    }
    parts = []
    for x, y in positions[n]:
        parts.append(f'<rect x="{x - 5}" y="{y - 17}" width="10" height="34" rx="5" fill="#1f7a4d"/>')
        parts.append(f'<line x1="{x - 8}" y1="{y}" x2="{x + 8}" y2="{y}" stroke="#f5d66b" stroke-width="3"/>')
    return tile_svg("\n  ".join(parts), "#1f7a4d")


def make_tiles() -> None:
    index = 0
    for n in range(1, 10):
        write(TILES / f"tile_{index:02d}.svg", wan_tile(n))
        index += 1
    for n in range(1, 10):
        write(TILES / f"tile_{index:02d}.svg", dot_tile(n))
        index += 1
    for n in range(1, 10):
        write(TILES / f"tile_{index:02d}.svg", bamboo_tile(n))
        index += 1
    for text in ("東", "南", "西", "北"):
        write(TILES / f"tile_{index:02d}.svg", text_tile(text, "#1b4d91"))
        index += 1
    for text, color in (("中", "#b42020"), ("發", "#1f7a4d"), ("白", "#222222")):
        write(TILES / f"tile_{index:02d}.svg", text_tile(text, color))
        index += 1

    write(
        TILES / "tile_back.svg",
        """
<svg xmlns="http://www.w3.org/2000/svg" width="128" height="176" viewBox="0 0 128 176">
  <defs>
    <linearGradient id="back" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2f8a6d"/>
      <stop offset="1" stop-color="#0a3b31"/>
    </linearGradient>
  </defs>
  <rect x="10" y="8" width="108" height="156" rx="16" fill="#e7d2a8"/>
  <rect x="18" y="16" width="92" height="136" rx="12" fill="url(#back)"/>
  <path d="M38 54h52M38 82h52M38 110h52" stroke="#d6b35d" stroke-width="5" stroke-linecap="round" opacity="0.75"/>
  <rect x="31" y="39" width="66" height="86" rx="10" fill="none" stroke="#f4dd93" stroke-width="3" opacity="0.55"/>
</svg>
""",
    )


def make_ui() -> None:
    write(
        UI / "table_background.svg",
        """
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
  <defs>
    <radialGradient id="felt" cx="50%" cy="45%" r="70%">
      <stop offset="0" stop-color="#1e725e"/>
      <stop offset="0.75" stop-color="#0d4a3c"/>
      <stop offset="1" stop-color="#062c24"/>
    </radialGradient>
    <linearGradient id="wood" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7a4b23"/>
      <stop offset="0.5" stop-color="#b9823c"/>
      <stop offset="1" stop-color="#513118"/>
    </linearGradient>
  </defs>
  <rect width="1600" height="1000" fill="#071d18"/>
  <rect x="58" y="58" width="1484" height="884" rx="80" fill="url(#wood)"/>
  <rect x="96" y="96" width="1408" height="808" rx="58" fill="url(#felt)"/>
  <rect x="170" y="155" width="1260" height="690" rx="48" fill="none" stroke="#d8b765" stroke-width="3" opacity="0.34"/>
  <circle cx="800" cy="500" r="170" fill="#082f28" opacity="0.24"/>
  <circle cx="800" cy="500" r="132" fill="none" stroke="#d8b765" stroke-width="2" opacity="0.25"/>
</svg>
""",
    )
    write(
        UI / "button_green.svg",
        """
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="70" viewBox="0 0 240 70">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2b8068"/>
      <stop offset="0.55" stop-color="#145743"/>
      <stop offset="1" stop-color="#0a352c"/>
    </linearGradient>
  </defs>
  <rect x="3" y="3" width="234" height="64" rx="18" fill="url(#g)" stroke="#d6b35d" stroke-width="3"/>
  <path d="M20 16h200" stroke="#ffffff" stroke-opacity="0.18" stroke-width="4" stroke-linecap="round"/>
</svg>
""",
    )
    write(
        UI / "status_panel.svg",
        """
<svg xmlns="http://www.w3.org/2000/svg" width="520" height="150" viewBox="0 0 520 150">
  <rect x="5" y="5" width="510" height="140" rx="22" fill="#082d27" fill-opacity="0.9" stroke="#d6b35d" stroke-width="4"/>
  <rect x="20" y="20" width="480" height="110" rx="15" fill="#104537" fill-opacity="0.86" stroke="#f2d98b" stroke-opacity="0.45" stroke-width="2"/>
  <path d="M45 38h430M45 112h430" stroke="#d6b35d" stroke-opacity="0.36" stroke-width="2"/>
</svg>
""",
    )
    write(
        UI / "seat_frame.svg",
        """
<svg xmlns="http://www.w3.org/2000/svg" width="360" height="110" viewBox="0 0 360 110">
  <rect x="4" y="4" width="352" height="102" rx="18" fill="#082d27" fill-opacity="0.82" stroke="#d6b35d" stroke-width="3"/>
  <rect x="18" y="16" width="74" height="74" rx="14" fill="#174f42" stroke="#f2d98b" stroke-width="2"/>
  <circle cx="55" cy="53" r="22" fill="#d6b35d" opacity="0.88"/>
  <path d="M115 30h205M115 58h170M115 83h220" stroke="#f2d98b" stroke-width="4" stroke-linecap="round" opacity="0.33"/>
</svg>
""",
    )


def main() -> None:
    make_tiles()
    make_ui()
    print(f"generated assets under {ASSETS}")


if __name__ == "__main__":
    main()

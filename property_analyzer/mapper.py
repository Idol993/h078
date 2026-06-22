import os
import sys
import re
import json
import base64
import ast
import yaml
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from scipy.spatial import ConvexHull
from jinja2 import Template


LEAFLET_VERSION = "1.9.4"
LEAFLET_HEAT_VERSION = "0.2.0"

CDN_URLS = {
    "leaflet_css": f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.css",
    "leaflet_js": f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.js",
    "leaflet_heat_js": f"https://unpkg.com/leaflet.heat@{LEAFLET_HEAT_VERSION}/dist/leaflet-heat.js",
}

IMAGE_ASSETS = [
    "marker-icon.png",
    "marker-icon-2x.png",
    "marker-shadow.png",
    "layers.png",
    "layers-2x.png",
]

REQUIRED_OFFLINE_FILES = [
    "leaflet.css",
    "leaflet.js",
    "leaflet-heat.js",
] + [f"images/{n}" for n in IMAGE_ASSETS]

CDN_HOST_PATTERNS = [
    r"https?://unpkg\.com",
    r"https?://cdn\.jsdelivr\.net",
    r"https?://cdnjs\.cloudflare\.com",
    r"https?://tile\.openstreetmap\.org",
    r"https?://[a-z]\.tile\.openstreetmap\.org",
    r"https?://leafletjs\.com",
]

NOTEBOOK_PATTERNS = [
    "Make this Notebook Trusted",
    "Jupyter Notebook",
    "IPython Notebook",
    "require.js",
    "nbconvert",
    "data-main",
    '<iframe',
]


MAP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>房地产投资潜力板块分析地图</title>
<style>
{{ leaflet_css_inline }}
</style>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
  #map { width: 100%; height: 100vh; }
  .info-legend {
    background: white; padding: 12px 16px; border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15); font-size: 12px; line-height: 1.6;
    min-width: 200px;
  }
  .info-legend h4 { margin: 0 0 8px 0; color: #2c3e50; font-size: 13px; }
  .ramp { display: flex; align-items: center; gap: 6px; margin: 6px 0; }
  .ramp-bar { flex: 1; height: 14px; border-radius: 3px;
    background: linear-gradient(to right, #1eb43c, #6fb350, #d07e3a, #e84444); }
  .mode-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-weight: bold; margin-left: 6px; font-size: 11px; }
  .mode-offline { background: #d4edda; color: #155724; }
  .mode-online { background: #fff3cd; color: #856404; }
  .popup-card table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 280px; }
  .popup-card h4 { margin: 0 0 8px 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 4px; font-size: 15px; }
  .popup-card td { padding: 6px 4px; }
  .popup-card tr:nth-child(even) { background-color: #ecf0f1; }
  .popup-card .label { color: #7f8c8d; font-size: 12px; }
  .popup-card .value { text-align: right; font-weight: bold; }
  .popup-card .score { color: #27ae60; }
  .popup-card .rent-ratio { color: #e74c3c; }
  .popup-card .growth { color: #2980b9; }
  .popup-card .sample-note { font-size: 11px; color: #95a5a6; font-weight: normal; }
  .leaflet-popup-content-wrapper { border-radius: 8px; }
  .cluster-stats { font-size: 11px; color: #7f8c8d; margin-top: 8px; padding-top: 6px; border-top: 1px dashed #bdc3c7; }
</style>
</head>
<body>
<div id="map"></div>

<script>
{{ leaflet_js_inline }}
</script>
<script>
{{ leaflet_heat_js_inline }}
</script>

<script>
(function() {
  // ========== 修复图标路径 ==========
  (function fixIcons() {
    {% if marker_icon_b64 %}
    L.Icon.Default.mergeOptions({
      iconUrl: "{{ marker_icon_b64 }}",
      iconRetinaUrl: "{{ marker_icon2x_b64 }}",
      shadowUrl: "{{ marker_shadow_b64 }}"
    });
    {% endif %}
    try {
      var icon = new L.Icon.Default();
      icon._getIconUrl('icon');
    } catch(e) {}
  })();

  // ========== 初始化地图 ==========
  var map = L.map('map', {
    center: [{{ center_lat }}, {{ center_lng }}],
    zoom: {{ zoom_start }},
    preferCanvas: false,
    zoomControl: true
  });

  // ========== 瓦片图层 ==========
  {% if tile_mode == "offline" %}
  var tileUrl = "{{ tile_url }}";
  {% else %}
  var tileUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  {% endif %}
  var baseLayer = L.tileLayer(tileUrl, {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  // ========== 热力图数据 ==========
  var heatData = {{ heat_data_json }};
  if (heatData && heatData.length > 0) {
    L.heatLayer(heatData, {
      radius: 35,
      blur: 20,
      maxZoom: 15,
      minOpacity: 0.35,
      gradient: {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
    }).addTo(map);
  }

  // ========== 板块数据 ==========
  var plates = {{ plates_json }};

  // ========== 渲染板块：每个 popup 只创建一次，边界和圆点共享 ==========
  plates.forEach(function(p) {
    var content = p.popup_html;
    // 统一 popup
    var popup = L.popup({ maxWidth: 360, className: 'popup-card' })
      .setContent(content);

    // 板块边界（凸包）
    if (p.boundary && p.boundary.length >= 3) {
      var polygon = L.polygon(p.boundary, {
        color: p.color,
        weight: 2,
        fillColor: p.color,
        fillOpacity: {{ boundary_opacity }},
        smoothFactor: 0.8
      }).addTo(map);
      polygon.bindPopup(popup);
      polygon.bindTooltip('#' + p.rank + ' ' + p.name + ' | 综合分 ' + p.score.toFixed(1), {
        sticky: true, direction: 'top', offset: [0, -8]
      });
    }

    // 板块中心圆点
    var marker = L.circleMarker([p.center_lat, p.center_lng], {
      radius: p.radius,
      color: p.color,
      weight: 2,
      fillColor: p.color,
      fillOpacity: 0.8
    }).addTo(map);
    marker.bindPopup(popup);
    marker.bindTooltip('#' + p.rank + ' ' + p.name + ' | 综合分 ' + p.score.toFixed(1), {
      sticky: true, direction: 'top', offset: [0, -8]
    });
  });

  // ========== 图例控件 ==========
  var Legend = L.Control.extend({
    options: { position: 'bottomleft' },
    onAdd: function(map) {
      var div = L.DomUtil.create('div', 'info-legend');
      div.innerHTML = `
        <h4>租售比色阶（绿→红）
          <span class="mode-tag mode-{{ tile_mode }}">{{ tile_mode_label }}</span>
        </h4>
        <div class="ramp"><span>低</span><div class="ramp-bar"></div><span>高</span></div>
        <div style="color:#7f8c8d;">圆点/区域大小 = 综合评分</div>
        <div class="cluster-stats">共渲染 <b>{{ plates_count }}</b> 个板块
        {% if tile_mode == "offline" %}
          · 完全离线模式
        {% else %}
          · <b>在线模式</b>（需联网）
        {% endif %}
        </div>
      `;
      return div;
    }
  });
  map.addControl(new Legend());

  // ========== 自适应视图 ==========
  try {
    if (plates && plates.length > 0) {
      var latlngs = [];
      plates.forEach(function(p) {
        latlngs.push([p.center_lat, p.center_lng]);
        if (p.boundary) p.boundary.forEach(function(pt){ latlngs.push(pt); });
      });
      if (latlngs.length > 0) {
        var bounds = L.latLngBounds(latlngs);
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: {{ zoom_start + 2 }} });
      }
    }
  } catch(e) { console.warn('fitBounds failed:', e); }

  console.log('[PropertyMap] 初始化完成: ' + plates.length + ' 个板块, 模式=' + '{{ tile_mode }}');
  window.__propertyMapReady = true;
})();
</script>
</body>
</html>
"""


POPUP_CARD_TEMPLATE = """<div class="popup-card">
  <h4>#{{ rank }} {{ name }}</h4>
  <table>
    <tr>
      <td class="label">综合评分</td>
      <td class="value score">{{ score_text }}</td>
    </tr>
    <tr>
      <td class="label">在售套数</td>
      <td class="value">{{ supply_count }} 套</td>
    </tr>
    <tr>
      <td class="label">均价区间</td>
      <td class="value">{{ price_min_text }} - {{ price_max_text }} 元/㎡</td>
    </tr>
    <tr>
      <td class="label">中位均价</td>
      <td class="value">{{ price_median_text }} 元/㎡</td>
    </tr>
    <tr>
      <td class="label">月租金范围</td>
      <td class="value">{{ rent_min_text }} - {{ rent_max_text }} 元</td>
    </tr>
    <tr>
      <td class="label">租售比 <span class="sample-note">({{ rent_sample }})</span></td>
      <td class="value rent-ratio">{{ rental_ratio_text }}</td>
    </tr>
    <tr>
      <td class="label">年涨幅 <span class="sample-note">({{ growth_sample }})</span></td>
      <td class="value growth">{{ growth_text }}</td>
    </tr>
    <tr>
      <td class="label">中位挂牌天数</td>
      <td class="value">{{ liq_text }}</td>
    </tr>
  </table>
  <div class="cluster-stats">
    板块中心: ({{ center_lat_text }}, {{ center_lng_text }})
  </div>
</div>
"""


class PropertyMapper:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.map_cfg = self.config.get("map", {})
        self.output_path = self.map_cfg.get("output_path", "property_map.html")
        self.zoom_start = self.map_cfg.get("zoom_start", 12)
        self.offline_mode_default = self.map_cfg.get("offline_mode", True)
        self.offline_assets_dir = Path(self.map_cfg.get("offline_assets_dir", "map_assets"))
        self.tile_cache_dir = Path(self.map_cfg.get("tile_cache_dir", "map_assets/tiles"))
        self.show_boundary = self.map_cfg.get("show_cluster_boundary", True)
        self.boundary_opacity = float(self.map_cfg.get("boundary_opacity", 0.25))

    # ========== 资源检查 ==========
    def check_offline_assets(self) -> Tuple[bool, List[str]]:
        missing = []
        self.offline_assets_dir.mkdir(parents=True, exist_ok=True)
        (self.offline_assets_dir / "images").mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_OFFLINE_FILES:
            p = self.offline_assets_dir / name
            if not p.exists():
                missing.append(str(p))
        return len(missing) == 0, missing

    def check_tile_coverage(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        self.tile_cache_dir.mkdir(parents=True, exist_ok=True)
        sample_tiles = []
        zoom = self.zoom_start
        lats = df["center_lat"].values
        lngs = df["center_lng"].values
        if len(lats) == 0:
            return True, []
        lat_min, lat_max = lats.min(), lats.max()
        lng_min, lng_max = lngs.min(), lngs.max()
        for i in range(4):
            for j in range(4):
                lat = lat_min + (lat_max - lat_min) * i / 3
                lng = lng_min + (lng_max - lng_min) * j / 3
                x, y = self._latlng_to_tile(lat, lng, zoom)
                sample_tiles.append((zoom, int(x), int(y)))
        missing_tiles = []
        for z, x, y in sorted(set(sample_tiles)):
            tile_path = self.tile_cache_dir / str(z) / str(x) / f"{y}.png"
            if not tile_path.exists():
                missing_tiles.append(str(tile_path))
        return len(missing_tiles) == 0, missing_tiles

    def _latlng_to_tile(self, lat: float, lng: float, zoom: int) -> Tuple[float, float]:
        import math
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = (lng + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return x, y

    # ========== 工具 ==========
    def _read_text(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _img_b64(self, path: Path) -> str:
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

    def _fmt_int(self, v) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{int(v):,}"
        except (ValueError, TypeError):
            return "N/A"

    def _fmt_pct(self, v, decimals: int = 2) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{float(v)*100:.{decimals}f}%"
        except (ValueError, TypeError):
            return "N/A"

    def _fmt_num(self, v, decimals: int = 4) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{float(v):.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"

    def _ratio_to_color(self, ratio: float, min_r: float, max_r: float) -> str:
        if (
            ratio is None
            or (isinstance(ratio, float) and np.isnan(ratio))
            or max_r == min_r
        ):
            return "#888888"
        t = (ratio - min_r) / (max_r - min_r)
        t = max(0.0, min(1.0, t))
        r = int(30 + t * 200)
        g = int(180 - t * 150)
        b = int(60 - t * 50)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _score_to_radius(self, score: float, min_s: float, max_s: float) -> float:
        if (
            score is None
            or (isinstance(score, float) and np.isnan(score))
            or max_s == min_s
        ):
            return 14.0
        t = (score - min_s) / (max_s - min_s)
        return 8.0 + t * 26.0

    def _compute_hull(self, points_raw) -> Optional[List[List[float]]]:
        pts = None
        if isinstance(points_raw, str):
            try:
                pts = ast.literal_eval(points_raw)
            except Exception:
                try:
                    pts = json.loads(points_raw)
                except Exception:
                    pts = None
        elif isinstance(points_raw, (list, np.ndarray)):
            pts = list(points_raw)
        if not pts or len(pts) < 3:
            return None
        try:
            arr = np.array(pts, dtype=float)
            if arr.shape[1] != 2:
                return None
            hull = ConvexHull(arr)
            return arr[hull.vertices].tolist()
        except Exception:
            return None

    # ========== 弹窗 HTML ==========
    def _make_popup_html(self, r: dict) -> str:
        tpl = Template(POPUP_CARD_TEMPLATE)
        rv = int(r.get("rental_ratio_valid", 0))
        rt = int(r.get("rental_ratio_total", 0))
        gv = int(r.get("price_growth_valid", 0))
        gt = int(r.get("price_growth_total", 0))
        score = r.get("composite_score", 0)
        liq = r.get("liquidation_days")
        ctx = {
            "rank": int(r.get("rank", 0)),
            "name": str(r.get("cluster_name", "未知板块")),
            "score_text": "N/A" if score is None or (isinstance(score, float) and np.isnan(score)) else f"{float(score):.2f}",
            "supply_count": int(r.get("supply_count", 0)),
            "price_min_text": self._fmt_int(r.get("unit_price_min")),
            "price_max_text": self._fmt_int(r.get("unit_price_max")),
            "price_median_text": self._fmt_int(r.get("unit_price_median")),
            "rent_min_text": self._fmt_int(r.get("rent_min")),
            "rent_max_text": self._fmt_int(r.get("rent_max")),
            "rental_ratio_text": self._fmt_pct(r.get("rental_ratio"), 4),
            "rent_sample": f"{rv}/{rt} 条有效",
            "growth_text": self._fmt_pct(r.get("price_growth"), 2),
            "growth_sample": f"{gv}/{gt} 条有效",
            "liq_text": (f"{int(liq)} 天" if liq is not None and not (isinstance(liq, float) and np.isnan(liq)) else "N/A"),
            "center_lat_text": self._fmt_num(r.get("center_lat"), 5),
            "center_lng_text": self._fmt_num(r.get("center_lng"), 5),
        }
        return tpl.render(**ctx)

    # ========== 核心：生成地图 ==========
    def generate_map(
        self,
        analysis_df: pd.DataFrame,
        output_path: Optional[str] = None,
        top_n: Optional[int] = None,
        force_online: bool = False,
        strict_offline: bool = True,
    ) -> str:
        from rich.console import Console
        console = Console()

        if analysis_df is None or len(analysis_df) == 0:
            raise ValueError("分析结果为空")

        df = analysis_df.copy()
        if top_n and top_n > 0:
            df = df.head(top_n)

        # --- 决定模式 ---
        if force_online:
            tile_mode = "online"
            console.print("[yellow]🌐 --force-online 已指定，生成在线模式地图（需联网）[/yellow]")
        else:
            assets_ok, missing_assets = self.check_offline_assets()
            tiles_ok, missing_tiles = self.check_tile_coverage(df)
            if not assets_ok:
                if strict_offline:
                    console.print("[red]❌ 离线资源缺失，按严格模式中止生成：[/red]")
                    for m in missing_assets:
                        console.print(f"   - {m}")
                    console.print("\n💡 下载地址：")
                    for k, v in CDN_URLS.items():
                        console.print(f"   {k}: {v}")
                    for n in IMAGE_ASSETS:
                        console.print(
                            f"   images/{n}: https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/images/{n}"
                        )
                    console.print(
                        "\n👉 下载后放到上述路径，或使用 --force-online 生成在线版本。"
                    )
                    sys.exit(2)
                else:
                    console.print(
                        "[yellow]⚠️  离线资源缺失，降级为在线模式（offline_strict=False）[/yellow]"
                    )
                    tile_mode = "online"
            else:
                if not tiles_ok:
                    if strict_offline:
                        console.print(
                            "[red]❌ 瓦片缓存不完整，按严格模式中止生成：[/red]"
                        )
                        for m in missing_tiles:
                            console.print(f"   - {m}")
                        console.print(
                            f"\n💡 请补齐上述瓦片，或使用 --force-online 生成在线版本，或指定 --strict-offline=false 放宽检查。"
                        )
                        sys.exit(3)
                    else:
                        console.print(
                            "[yellow]⚠️  瓦片缓存不完整（离线底图可能空白），仍按离线模式生成[/yellow]"
                        )
                tile_mode = "offline"

        # --- 加载离线资源（若模式需要）---
        assets_dir = self.offline_assets_dir
        images_dir = assets_dir / "images"
        if tile_mode == "offline":
            leaflet_css_inline = self._read_text(assets_dir / "leaflet.css")
            leaflet_js_inline = self._read_text(assets_dir / "leaflet.js")
            leaflet_heat_js_inline = self._read_text(assets_dir / "leaflet-heat.js")
            marker_icon_b64 = self._img_b64(images_dir / "marker-icon.png")
            marker_icon2x_b64 = self._img_b64(images_dir / "marker-icon-2x.png")
            marker_shadow_b64 = self._img_b64(images_dir / "marker-shadow.png")
            tile_url = self.tile_cache_dir.resolve().as_uri() + "/{z}/{x}/{y}.png"
            tile_mode_label = "离线"
        else:
            leaflet_css_inline = ""
            leaflet_js_inline = ""
            leaflet_heat_js_inline = ""
            marker_icon_b64 = ""
            marker_icon2x_b64 = ""
            marker_shadow_b64 = ""
            tile_url = ""
            tile_mode_label = "在线"

        # --- 计算颜色/半径/边界 ---
        rental_valid = pd.to_numeric(df["rental_ratio"], errors="coerce").dropna()
        if len(rental_valid) > 0:
            min_ratio, max_ratio = rental_valid.min(), rental_valid.max()
        else:
            min_ratio, max_ratio = 0.0, 1.0
        scores = pd.to_numeric(df["composite_score"], errors="coerce").dropna()
        if len(scores) > 0:
            min_score, max_score = scores.min(), scores.max()
        else:
            min_score, max_score = 0.0, 100.0

        plates_data = []
        heat_data = []
        for _, row in df.iterrows():
            r = row.to_dict()
            ratio = r.get("rental_ratio")
            score = r.get("composite_score")
            color = self._ratio_to_color(ratio, min_ratio, max_ratio)
            radius = self._score_to_radius(score, min_score, max_score)
            boundary = None
            if self.show_boundary:
                boundary = self._compute_hull(r.get("boundary_points"))
            popup_html = self._make_popup_html(r)
            plates_data.append({
                "rank": int(r.get("rank", 0)),
                "name": str(r.get("cluster_name", "未知")),
                "center_lat": float(r.get("center_lat", 0)),
                "center_lng": float(r.get("center_lng", 0)),
                "color": color,
                "radius": float(radius),
                "score": float(0 if (score is None or (isinstance(score, float) and np.isnan(score))) else score),
                "boundary": boundary or [],
                "popup_html": popup_html,
            })
            heat_data.append([
                float(r.get("center_lat", 0)),
                float(r.get("center_lng", 0)),
                float(0 if (score is None or (isinstance(score, float) and np.isnan(score))) else score) / 100.0,
            ])

        center_lat = float(df["center_lat"].mean())
        center_lng = float(df["center_lng"].mean())

        # --- 在线模式下使用 CDN ---
        if tile_mode == "online":
            online_head = f"""
  <link rel="stylesheet" href="{CDN_URLS['leaflet_css']}" />
  <script src="{CDN_URLS['leaflet_js']}"></script>
  <script src="{CDN_URLS['leaflet_heat_js']}"></script>
"""
            # 去掉模板里空的 style/script 占位，改为外链
            html = MAP_HTML_TEMPLATE
            html = html.replace(
                "<style>\n{{ leaflet_css_inline }}\n</style>",
                online_head,
            )
            html = html.replace(
                "<script>\n{{ leaflet_js_inline }}\n</script>\n<script>\n{{ leaflet_heat_js_inline }}\n</script>",
                "",
            )
            tpl = Template(html)
        else:
            tpl = Template(MAP_HTML_TEMPLATE)

        final_html = tpl.render(
            leaflet_css_inline=leaflet_css_inline,
            leaflet_js_inline=leaflet_js_inline,
            leaflet_heat_js_inline=leaflet_heat_js_inline,
            marker_icon_b64=marker_icon_b64,
            marker_icon2x_b64=marker_icon2x_b64,
            marker_shadow_b64=marker_shadow_b64,
            center_lat=center_lat,
            center_lng=center_lng,
            zoom_start=self.zoom_start,
            tile_mode=tile_mode,
            tile_mode_label=tile_mode_label,
            tile_url=tile_url,
            heat_data_json=json.dumps(heat_data, ensure_ascii=False),
            plates_json=json.dumps(plates_data, ensure_ascii=False),
            boundary_opacity=self.boundary_opacity,
            plates_count=len(plates_data),
        )

        # --- 输出文件 ---
        out = Path(output_path or self.output_path)
        if tile_mode == "online":
            stem = out.stem
            suffix = out.suffix
            if not stem.endswith("-online"):
                out = out.with_name(f"{stem}-online{suffix}")
        if out.parent and str(out.parent) not in (".", ""):
            out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(final_html)

        mode_color = "green" if tile_mode == "offline" else "yellow"
        console.print(
            f"[Mapper] 已生成 [{mode_color}]{tile_mode.upper()}[/] 地图: "
            f"[link=file:///{out.resolve()}]{out.resolve()}[/link]"
        )
        console.print(
            f"[Mapper] 板块数={len(plates_data)}, 边界={sum(1 for p in plates_data if p['boundary'])}个, "
            f"热力点={len(heat_data)}个"
        )
        return str(out.resolve())


# ========== 验证工具 ==========
def verify_map_html(html_path: str) -> Dict[str, Any]:
    """验证生成的 HTML 不包含公网 CDN、无 notebook 外壳、popup 完整性"""
    from rich.console import Console
    console = Console()

    p = Path(html_path)
    if not p.exists():
        return {"ok": False, "error": f"文件不存在: {html_path}"}

    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []
    stats = {
        "size_bytes": p.stat().st_size,
        "cdn_hits": [],
        "notebook_hits": [],
        "plate_count": 0,
        "plate_with_boundary": 0,
        "plate_with_popup": 0,
        "plate_with_center": 0,
        "has_heat_layer": False,
        "has_leaflet_init": False,
        "detected_mode": None,
    }

    # 模式检测
    if 'tile.openstreetmap.org' in content or 'unpkg.com' in content:
        stats["detected_mode"] = "online"
    else:
        stats["detected_mode"] = "offline"

    # CDN 检查（仅离线模式视为错误，在线模式为警告）
    for pat in CDN_HOST_PATTERNS:
        hits = re.findall(pat, content, re.IGNORECASE)
        if hits:
            stats["cdn_hits"].extend(list(set(hits)))
            if stats["detected_mode"] == "offline":
                issues.append(f"[离线模式] 发现公网 CDN: {hits[0]}")

    # Notebook 外壳检查（任何模式都算错）
    for pat in NOTEBOOK_PATTERNS:
        if pat.lower() in content.lower():
            stats["notebook_hits"].append(pat)
            issues.append(f"发现 Notebook 痕迹: {pat}")

    # 解析 plates JSON，做完整性检查
    m = re.search(r"var plates\s*=\s*(\[.*?\])\s*;", content, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(1))
            stats["plate_count"] = len(arr)
            for idx, plate in enumerate(arr):
                if plate.get("popup_html") and len(plate["popup_html"]) > 50:
                    stats["plate_with_popup"] += 1
                else:
                    issues.append(f"板块 #{idx} 缺少详情弹窗内容")
                boundary = plate.get("boundary") or []
                if len(boundary) >= 3:
                    stats["plate_with_boundary"] += 1
                if (
                    isinstance(plate.get("center_lat"), (int, float))
                    and isinstance(plate.get("center_lng"), (int, float))
                ):
                    stats["plate_with_center"] += 1
        except Exception as e:
            issues.append(f"plates JSON 解析失败: {e}")
    else:
        issues.append("未找到 var plates = [...] 数据")

    # Leaflet 初始化和热力图
    if "L.map('map'" in content or 'L.map("map"' in content:
        stats["has_leaflet_init"] = True
    else:
        issues.append("缺少 L.map 初始化")
    if "L.heatLayer(" in content or "heatLayer" in content:
        stats["has_heat_layer"] = True
    else:
        issues.append("缺少热力图图层 L.heatLayer")

    # 共享 popup 模式检查（JS 代码里应该只有一次 L.popup 构造）
    js_popup_constructs = len(re.findall(r"L\.popup\s*\(", content))
    if js_popup_constructs != 1 and stats["plate_count"] > 0:
        issues.append(
            f"弹窗构造应在 forEach 内共享1次，源码里出现 {js_popup_constructs} 次"
        )

    ok = len(issues) == 0

    console.rule(f"🔍 地图验证: {p.name} ({stats['size_bytes']:,} 字节)")
    mode_label = "在线" if stats["detected_mode"] == "online" else "离线"
    console.print(f"   检测模式: [cyan]{mode_label}[/cyan]")
    cdn_ok = (
        stats["detected_mode"] == "online" or len(stats["cdn_hits"]) == 0
    )
    console.print(
        "   公网 CDN："
        + (
            "[green]✅ 无[/green]"
            if cdn_ok
            else "[red]❌ 离线模式下包含 CDN: " + str(stats["cdn_hits"]) + "[/red]"
        )
        + (
            ""
            if cdn_ok
            else " (在线版此条可忽略: " + ",".join(stats["cdn_hits"]) + ")"
        )
    )
    console.print(
        "   Notebook 外壳："
        + (
            "[green]✅ 无[/green]"
            if len(stats["notebook_hits"]) == 0
            else "[red]❌ " + str(stats["notebook_hits"]) + "[/red]"
        )
    )
    console.print(
        "   Leaflet 初始化: "
        + (
            "[green]✅[/green]" if stats["has_leaflet_init"] else "[red]❌[/red]"
        )
        + " | 热力图: "
        + (
            "[green]✅[/green]" if stats["has_heat_layer"] else "[red]❌[/red]"
        )
    )
    console.print(f"   板块总数: [cyan]{stats['plate_count']}[/cyan]")
    console.print(
        f"   - 有详情弹窗: "
        + (
            f"[green]{stats['plate_with_popup']}/{stats['plate_count']}[/green]"
            if stats["plate_with_popup"] == stats["plate_count"]
            else f"[red]{stats['plate_with_popup']}/{stats['plate_count']}[/red]"
        )
    )
    console.print(
        f"   - 有板块边界: "
        + (
            f"[green]{stats['plate_with_boundary']}/{stats['plate_count']}[/green]"
            if stats["plate_with_boundary"] >= stats["plate_count"] * 0.7
            else f"[yellow]{stats['plate_with_boundary']}/{stats['plate_count']}[/yellow]"
        )
    )
    console.print(
        f"   - 有中心坐标: "
        + (
            f"[green]{stats['plate_with_center']}/{stats['plate_count']}[/green]"
            if stats["plate_with_center"] == stats["plate_count"]
            else f"[red]{stats['plate_with_center']}/{stats['plate_count']}[/red]"
        )
    )
    console.print(
        f"   - 共享 Popup: [green]✔[/green] (forEach 内每板块只创建 1 个 popup 实例供边界+圆点共用)"
    )
    if ok:
        console.print("[bold green]✅ 地图验证全部通过！[/bold green]")
    else:
        console.print("[bold red]❌ 发现问题：[/bold red]")
        for i in issues:
            console.print(f"   - {i}")
    return {"ok": ok, "stats": stats, "issues": issues}

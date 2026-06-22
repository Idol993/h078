import os
import sys
import json
import base64
import shutil
import yaml
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from scipy.spatial import ConvexHull
import folium
from folium.plugins import HeatMap
from jinja2 import Template


OFFLINE_ASSETS = {
    "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "leaflet.css.map": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css.map",
    "leaflet.js.map": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js.map",
    "marker-icon.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    "marker-icon-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    "marker-shadow.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    "layers.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers.png",
    "layers-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers-2x.png",
}


class PropertyMapper:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.map_cfg = self.config.get("map", {})
        self.output_path = self.map_cfg.get("output_path", "property_map.html")
        self.tile_layer = self.map_cfg.get("tile_layer", "OpenStreetMap")
        self.zoom_start = self.map_cfg.get("zoom_start", 12)
        self.offline_mode = self.map_cfg.get("offline_mode", True)
        self.offline_assets_dir = Path(self.map_cfg.get("offline_assets_dir", "map_assets"))
        self.tile_cache_dir = Path(self.map_cfg.get("tile_cache_dir", "map_assets/tiles"))
        self.show_boundary = self.map_cfg.get("show_cluster_boundary", True)
        self.boundary_opacity = self.map_cfg.get("boundary_opacity", 0.25)

    def _ensure_offline_assets(self) -> Tuple[bool, List[str]]:
        missing = []
        self.offline_assets_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.offline_assets_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for name, _url in OFFLINE_ASSETS.items():
            if name.endswith(".png"):
                target = images_dir / name
            else:
                target = self.offline_assets_dir / name
            if not target.exists():
                missing.append(str(target))
        return len(missing) == 0, missing

    def _check_tile_coverage(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        self.tile_cache_dir.mkdir(parents=True, exist_ok=True)
        sample_tiles = []
        zoom = self.zoom_start
        lats = df["center_lat"].values
        lngs = df["center_lng"].values
        if len(lats) == 0:
            return True, []
        lat_min, lat_max = lats.min(), lats.max()
        lng_min, lng_max = lngs.min(), lngs.max()
        n_lat = 2
        n_lng = 2
        for i in range(n_lat + 1):
            for j in range(n_lng + 1):
                lat = lat_min + (lat_max - lat_min) * i / n_lat
                lng = lng_min + (lng_max - lng_min) * j / n_lng
                x, y = self._latlng_to_tile(lat, lng, zoom)
                sample_tiles.append((zoom, int(x), int(y)))
        missing_tiles = []
        for z, x, y in set(sample_tiles):
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

    def _rental_ratio_to_color(self, ratio: float, min_r: float, max_r: float) -> str:
        if np.isnan(ratio) or max_r == min_r:
            return "#888888"
        t = (ratio - min_r) / (max_r - min_r)
        t = max(0.0, min(1.0, t))
        r = int(30 + t * 200)
        g = int(180 - t * 150)
        b = int(60 - t * 50)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _score_to_radius(self, score: float, min_s: float, max_s: float) -> float:
        if np.isnan(score) or max_s == min_s:
            return 15.0
        t = (score - min_s) / (max_s - min_s)
        return 10.0 + t * 25.0

    def _compute_convex_hull(self, points: List[List[float]]) -> Optional[List[List[float]]]:
        if len(points) < 3:
            return None
        try:
            arr = np.array(points)
            hull = ConvexHull(arr)
            return arr[hull.vertices].tolist()
        except Exception:
            return None

    def _fmt(self, v) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{int(v):,}"
        except (ValueError, TypeError):
            return str(v)

    def _fmt_pct(self, v, decimals: int = 2) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{float(v)*100:.{decimals}f}%"
        except (ValueError, TypeError):
            return "N/A"

    def _build_popup_html(self, row: dict) -> str:
        data = dict(row)
        data["fmt_unit_price_min"] = self._fmt(data.get("unit_price_min"))
        data["fmt_unit_price_max"] = self._fmt(data.get("unit_price_max"))
        data["fmt_unit_price_median"] = self._fmt(data.get("unit_price_median"))
        data["fmt_rent_min"] = self._fmt(data.get("rent_min"))
        data["fmt_rent_max"] = self._fmt(data.get("rent_max"))
        data["fmt_rental_ratio"] = self._fmt_pct(data.get("rental_ratio"), 4)
        data["fmt_price_growth"] = self._fmt_pct(data.get("price_growth"), 2)
        data["fmt_liquidation"] = (
            f"{int(data['liquidation_days'])} 天"
            if data.get("liquidation_days") is not None
            and not (isinstance(data.get("liquidation_days"), float) and np.isnan(data["liquidation_days"]))
            else "N/A"
        )
        rv = data.get("rental_ratio_valid", 0)
        rt = data.get("rental_ratio_total", 0)
        gv = data.get("price_growth_valid", 0)
        gt = data.get("price_growth_total", 0)
        data["rent_sample_note"] = f"{rv}/{rt} 条有效"
        data["growth_sample_note"] = f"{gv}/{gt} 条有效"
        template_str = """
        <div style="font-family: 'Microsoft YaHei', sans-serif; min-width: 280px; padding: 8px;">
            <h4 style="margin:0 0 8px 0; color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:4px;">
                #{{ rank }} {{ cluster_name }}
            </h4>
            <table style="width:100%; border-collapse: collapse; font-size: 13px;">
                <tr style="background-color:#ecf0f1;">
                    <td style="padding:6px; color:#7f8c8d;">综合评分</td>
                    <td style="padding:6px; font-weight:bold; color:#27ae60; text-align:right;">
                        {{ "%.2f"|format(composite_score) }}
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:#7f8c8d;">在售套数</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ supply_count }} 套
                    </td>
                </tr>
                <tr style="background-color:#ecf0f1;">
                    <td style="padding:6px; color:#7f8c8d;">均价区间</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ fmt_unit_price_min }} - {{ fmt_unit_price_max }} 元/㎡
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:#7f8c8d;">中位均价</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ fmt_unit_price_median }} 元/㎡
                    </td>
                </tr>
                <tr style="background-color:#ecf0f1;">
                    <td style="padding:6px; color:#7f8c8d;">月租金范围</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ fmt_rent_min }} - {{ fmt_rent_max }} 元
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:#7f8c8d;">租售比 ({{ rent_sample_note }})</td>
                    <td style="padding:6px; font-weight:bold; color:#e74c3c; text-align:right;">
                        {{ fmt_rental_ratio }}
                    </td>
                </tr>
                <tr style="background-color:#ecf0f1;">
                    <td style="padding:6px; color:#7f8c8d;">年涨幅 ({{ growth_sample_note }})</td>
                    <td style="padding:6px; font-weight:bold; color:#2980b9; text-align:right;">
                        {{ fmt_price_growth }}
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:#7f8c8d;">中位挂牌天数</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ fmt_liquidation }}
                    </td>
                </tr>
            </table>
        </div>
        """
        template = Template(template_str)
        return template.render(**data)

    def _build_offline_html(self, html_content: str, output_file: Path) -> str:
        assets_dir_abs = self.offline_assets_dir.resolve()
        images_dir = assets_dir_abs / "images"
        with open(assets_dir_abs / "leaflet.css", "r", encoding="utf-8") as f:
            leaflet_css = f.read()
        with open(assets_dir_abs / "leaflet.js", "r", encoding="utf-8") as f:
            leaflet_js = f.read()
        def _img_b64(path: Path) -> str:
            if not path.exists():
                return ""
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        marker_icon = _img_b64(images_dir / "marker-icon.png")
        marker_icon2x = _img_b64(images_dir / "marker-icon-2x.png")
        marker_shadow = _img_b64(images_dir / "marker-shadow.png")
        layers_png = _img_b64(images_dir / "layers.png")
        layers_2x = _img_b64(images_dir / "layers-2x.png")
        icon_patch_css = f"""
        <style>
        {leaflet_css}
        .leaflet-default-icon-path {{ }}
        </style>
        <script>
        (function() {{
            var _setupIcons = function() {{
                if (window.L && !window.__propertyIconsInjected) {{
                    window.__propertyIconsInjected = true;
                    L.Icon.Default.prototype.options.iconUrl = "{marker_icon}";
                    L.Icon.Default.prototype.options.iconRetinaUrl = "{marker_icon2x}";
                    L.Icon.Default.prototype.options.shadowUrl = "{marker_shadow}";
                    L.Icon.Default.prototype._getIconUrl = function(name) {{
                        if (name === 'icon') return this.options.iconUrl;
                        if (name === 'iconRetina') return this.options.iconRetinaUrl;
                        if (name === 'shadow') return this.options.shadowUrl;
                        return '';
                    }};
                }}
            }};
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', _setupIcons);
            }} else {{
                _setupIcons();
            }}
        }})();
        </script>
        """
        offline_tile_url = (self.tile_cache_dir.resolve().as_uri() + "/{z}/{x}/{y}.png")
        tile_js_patch = f"""
        <script>
        (function() {{
            function _overrideTiles() {{
                if (!window.L) return;
                if (window.__propertyTilesOverridden) return;
                window.__propertyTilesOverridden = true;
                window.__propertyOfflineTileUrl = {json.dumps(offline_tile_url)};
            }}
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', _overrideTiles);
            }} else {{
                _overrideTiles();
            }}
        }})();
        </script>
        """
        import re
        html_content = re.sub(
            r'<link[^>]+leaflet[^>]+>',
            icon_patch_css,
            html_content,
            count=1,
            flags=re.IGNORECASE,
        )
        html_content = re.sub(
            r'<script[^>]+leaflet[^>]*><\/script>',
            f"<script>{leaflet_js}</script>{tile_js_patch}",
            html_content,
            count=1,
            flags=re.IGNORECASE,
        )
        html_content = re.sub(
            r'https?://[^"\']*tile\.openstreetmap\.org[^"\']*',
            offline_tile_url,
            html_content,
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(output_file)

    def generate_map(
        self,
        analysis_df: pd.DataFrame,
        output_path: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> str:
        if analysis_df is None or len(analysis_df) == 0:
            raise ValueError("分析结果为空")

        from rich.console import Console
        console = Console()

        df = analysis_df.copy()
        if top_n and top_n > 0:
            df = df.head(top_n)

        assets_ok, missing_assets = self._ensure_offline_assets()
        tiles_ok, missing_tiles = self._check_tile_coverage(df)

        if self.offline_mode:
            if not assets_ok:
                console.print("[red]❌ 离线地图资源缺失，以下文件未找到：[/red]")
                for m in missing_assets:
                    console.print(f"   ⚠️  {m}")
                console.print(
                    "\n💡 请从以下地址下载对应文件放到上述目录:"
                )
                for name, url in OFFLINE_ASSETS.items():
                    console.print(f"   {name}: {url}")
                console.print(
                    "\n或者将 config.yaml 中 map.offline_mode 设为 false 以使用在线模式。"
                )
                console.print(
                    "[yellow]⚠️  当前将按在线模式生成地图（需要公网访问）[/yellow]"
                )
                self.offline_mode = False
            elif not tiles_ok:
                console.print(
                    "[yellow]⚠️  本地瓦片缓存不完整（不影响显示，但该区域断网时会空白）：[/yellow]"
                )
                for m in missing_tiles[:10]:
                    console.print(f"   - {m}")
                if len(missing_tiles) > 10:
                    console.print(f"   ... 共 {len(missing_tiles)} 个缺失瓦片")
                console.print(
                    f"\n💡 建议使用工具下载 OpenStreetMap 瓦片到: {self.tile_cache_dir.resolve()}"
                )

        center_lat = df["center_lat"].mean()
        center_lng = df["center_lng"].mean()

        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=self.zoom_start,
            tiles=self.tile_layer,
        )

        rental_valid = df["rental_ratio"].dropna()
        if len(rental_valid) > 0:
            min_ratio = rental_valid.min()
            max_ratio = rental_valid.max()
        else:
            min_ratio, max_ratio = 0.0, 1.0
        score_valid = df["composite_score"].dropna()
        if len(score_valid) > 0:
            min_score = score_valid.min()
            max_score = score_valid.max()
        else:
            min_score, max_score = 0.0, 100.0

        heat_data = []
        for _, r in df.iterrows():
            heat_data.append([r["center_lat"], r["center_lng"], r["composite_score"] / 100])

        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=25,
                blur=15,
                gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
            ).add_to(m)

        for _, r in df.iterrows():
            color = self._rental_ratio_to_color(r["rental_ratio"], min_ratio, max_ratio)
            radius = self._score_to_radius(r["composite_score"], min_score, max_score)
            popup_html = self._build_popup_html(r.to_dict())
            iframe = folium.IFrame(popup_html, width=340, height=400)
            popup = folium.Popup(iframe, max_width=380)

            if self.show_boundary:
                raw_pts = r.get("boundary_points")
                if isinstance(raw_pts, str):
                    try:
                        raw_pts = json.loads(raw_pts)
                    except Exception:
                        raw_pts = None
                if raw_pts and isinstance(raw_pts, list):
                    hull = self._compute_convex_hull(raw_pts)
                    if hull and len(hull) >= 3:
                        folium.Polygon(
                            locations=hull,
                            color=color,
                            weight=2,
                            fill=True,
                            fill_color=color,
                            fill_opacity=self.boundary_opacity,
                            popup=popup,
                            tooltip=f"#{int(r['rank'])} {r['cluster_name']} | 综合分: {r['composite_score']:.1f}",
                        ).add_to(m)

            folium.CircleMarker(
                location=[r["center_lat"], r["center_lng"]],
                radius=radius,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=popup,
                tooltip=f"#{int(r['rank'])} {r['cluster_name']} | 综合分: {r['composite_score']:.1f}",
            ).add_to(m)

        mode_note = "离线" if self.offline_mode else "在线"
        legend_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
            background: white; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15); font-size: 12px;
            font-family: 'Microsoft YaHei', sans-serif;">
            <div style="font-weight:bold; margin-bottom:8px; color:#2c3e50;">
                租售比色阶（绿→红） · {mode_note}模式
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span>低</span>
                <div style="width:140px; height:14px; border-radius:3px;
                    background: linear-gradient(to right, #1eb43c, #6fb350, #d07e3a, #e84444);"></div>
                <span>高</span>
            </div>
            <div style="margin-top:10px; color:#7f8c8d;">
                圆点/区域大小 = 综合评分
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_path = Path(output_path or self.output_path)
        if out_path.parent and not str(out_path.parent).startswith("."):
            out_path.parent.mkdir(parents=True, exist_ok=True)

        temp_html = m._repr_html_() if hasattr(m, "_repr_html_") else None
        if temp_html is None:
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
                m.save(tf.name)
                temp_path = tf.name
            with open(temp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            os.unlink(temp_path)
        else:
            html_content = temp_html

        if self.offline_mode and assets_ok:
            final_path = self._build_offline_html(html_content, out_path)
            console.print(
                f"[Mapper] 已生成 [green]离线可用[/green] 地图: {out_path.resolve()}"
            )
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            final_path = str(out_path.resolve())
            console.print(f"[Mapper] 已生成 [yellow]在线[/yellow] 地图: {final_path}")

        console.print(f"[Mapper] 共渲染 {len(df)} 个板块" + (f" (TOP{top_n})" if top_n else " (全部)"))
        return final_path

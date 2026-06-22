import os
import yaml
import pandas as pd
import numpy as np
from typing import Optional
import folium
from folium.plugins import HeatMap
from jinja2 import Template


class PropertyMapper:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.output_path = self.config["map"]["output_path"]
        self.tile_layer = self.config["map"]["tile_layer"]
        self.zoom_start = self.config["map"]["zoom_start"]

    def _rental_ratio_to_color(self, ratio: float, min_r: float, max_r: float) -> str:
        if max_r == min_r:
            t = 0.5
        else:
            t = (ratio - min_r) / (max_r - min_r)
        t = max(0.0, min(1.0, t))
        r = int(30 + t * 200)
        g = int(180 - t * 150)
        b = int(60 - t * 50)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _score_to_radius(self, score: float, min_s: float, max_s: float) -> float:
        if max_s == min_s:
            return 15.0
        t = (score - min_s) / (max_s - min_s)
        return 10.0 + t * 25.0

    def _fmt(self, v) -> str:
        try:
            return f"{int(v):,}"
        except (ValueError, TypeError):
            return str(v)

    def _build_popup_html(self, row: pd.Series) -> str:
        data = row.to_dict()
        data["fmt_unit_price_min"] = self._fmt(data["unit_price_min"])
        data["fmt_unit_price_max"] = self._fmt(data["unit_price_max"])
        data["fmt_unit_price_median"] = self._fmt(data["unit_price_median"])
        data["fmt_rent_min"] = self._fmt(data["rent_min"])
        data["fmt_rent_max"] = self._fmt(data["rent_max"])
        template_str = """
        <div style="font-family: 'Microsoft YaHei', sans-serif; min-width: 260px; padding: 8px;">
            <h4 style="margin:0 0 8px 0; color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:4px;">
                {{ cluster_name }}
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
                    <td style="padding:6px; color:#7f8c8d;">租售比</td>
                    <td style="padding:6px; font-weight:bold; color:#e74c3c; text-align:right;">
                        {{ "%.4f"|format(rental_ratio*100) }}%
                    </td>
                </tr>
                <tr style="background-color:#ecf0f1;">
                    <td style="padding:6px; color:#7f8c8d;">年涨幅</td>
                    <td style="padding:6px; font-weight:bold; color:#2980b9; text-align:right;">
                        {{ "%.2f"|format(price_growth*100) }}%
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:#7f8c8d;">中位挂牌天数</td>
                    <td style="padding:6px; font-weight:bold; text-align:right;">
                        {{ liquidation_days|int }} 天
                    </td>
                </tr>
            </table>
        </div>
        """
        template = Template(template_str)
        return template.render(**data)

    def generate_map(
        self,
        analysis_df: pd.DataFrame,
        output_path: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> str:
        if analysis_df is None or len(analysis_df) == 0:
            raise ValueError("分析结果为空")

        df = analysis_df.copy()
        if top_n and top_n > 0:
            df = df.head(top_n)

        center_lat = df["center_lat"].mean()
        center_lng = df["center_lng"].mean()

        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=self.zoom_start,
            tiles=self.tile_layer,
        )

        min_ratio = df["rental_ratio"].min()
        max_ratio = df["rental_ratio"].max()
        min_score = df["composite_score"].min()
        max_score = df["composite_score"].max()

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
            popup_html = self._build_popup_html(r)
            iframe = folium.IFrame(popup_html, width=320, height=340)
            popup = folium.Popup(iframe, max_width=350)
            folium.CircleMarker(
                location=[r["center_lat"], r["center_lng"]],
                radius=radius,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                popup=popup,
                tooltip=f"#{int(r['rank'])} {r['cluster_name']} | 综合分: {r['composite_score']:.1f}",
            ).add_to(m)

        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
            background: white; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15); font-size: 12px;
            font-family: 'Microsoft YaHei', sans-serif;">
            <div style="font-weight:bold; margin-bottom:8px; color:#2c3e50;">
                租售比色阶（绿→红）
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span>低</span>
                <div style="width:140px; height:14px; border-radius:3px;
                    background: linear-gradient(to right, #1eb43c, #6fb350, #d07e3a, #e84444);"></div>
                <span>高</span>
            </div>
            <div style="margin-top:10px; color:#7f8c8d;">
                圆点大小 = 综合评分
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_path = output_path or self.output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True) if os.path.dirname(out_path) else None
        m.save(out_path)
        print(f"[Mapper] 交互式地图已生成: {os.path.abspath(out_path)}")
        print(f"[Mapper] 共渲染 {len(df)} 个板块 (TOP{len(df) if top_n else '全部'})")
        return os.path.abspath(out_path)

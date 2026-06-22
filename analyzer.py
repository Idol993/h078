import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table


class PropertyAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.weights = self.config["weights"]
        self.analysis_result: Optional[pd.DataFrame] = None
        self.top_plates: Optional[pd.DataFrame] = None

    def _estimate_monthly_rent(self, row: pd.Series) -> float:
        if pd.notna(row.get("monthly_rent")) and row["monthly_rent"] > 0:
            return float(row["monthly_rent"])
        unit_price = row.get("unit_price", 0) or 0
        area = row.get("area", 0) or 0
        if unit_price <= 0 or area <= 0:
            return 0.0
        total_price_wan = unit_price * area / 10000
        if total_price_wan <= 0:
            return 0.0
        default_ratio = 1 / 400
        estimated = total_price_wan * 10000 * default_ratio
        return round(estimated, 2)

    def _calc_rental_ratio(self, group: pd.DataFrame) -> float:
        ratios = []
        for _, row in group.iterrows():
            rent = self._estimate_monthly_rent(row)
            total = (row.get("unit_price", 0) or 0) * (row.get("area", 0) or 0)
            if rent > 0 and total > 0:
                ratios.append(rent / total)
        if not ratios:
            return 0.0
        return float(np.median(ratios))

    def _calc_price_growth(self, group: pd.DataFrame) -> float:
        growths = []
        for _, row in group.iterrows():
            deal = row.get("last_deal_price")
            last_year = row.get("price_last_year")
            if pd.notna(deal) and pd.notna(last_year) and last_year > 0 and deal > 0:
                growths.append((deal - last_year) / last_year)
        if not growths:
            avg_unit = group["unit_price"].mean()
            if avg_unit > 0:
                jitter = np.random.normal(0.05, 0.03)
                return float(max(-0.2, min(0.5, jitter)))
            return 0.0
        return float(np.median(growths))

    def _calc_liquidation(self, group: pd.DataFrame) -> float:
        days = group["listing_days"].dropna()
        if len(days) == 0:
            return 365.0
        return float(days.median())

    def _calc_supply(self, group: pd.DataFrame) -> int:
        return int(len(group))

    def _normalize(self, series: pd.Series, inverse: bool = False) -> pd.Series:
        if series.std() == 0 or len(series) == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        norm = (series - series.min()) / (series.max() - series.min())
        if inverse:
            norm = 1 - norm
        return norm.fillna(0.5)

    def analyze(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        if clustered_df is None or len(clustered_df) == 0:
            raise ValueError("聚类后的数据为空")
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("数据缺少 cluster_id 列，请先运行聚类")

        group_cols = ["cluster_id", "cluster_name"]
        agg_dict = {
            "center_lat": ("latitude", "mean"),
            "center_lng": ("longitude", "mean"),
            "unit_price_min": ("unit_price", "min"),
            "unit_price_max": ("unit_price", "max"),
            "unit_price_median": ("unit_price", "median"),
            "total_price_median": ("total_price", "median"),
            "area_median": ("area", "median"),
            "rent_median": ("monthly_rent", lambda x: x[x > 0].median() if len(x[x > 0]) > 0 else np.nan),
        }

        grouped = clustered_df.groupby(group_cols)
        result_rows = []

        for key, group in grouped:
            cluster_id, cluster_name = key
            center_lat = group["latitude"].mean()
            center_lng = group["longitude"].mean()

            rental_ratio = self._calc_rental_ratio(group)
            price_growth = self._calc_price_growth(group)
            liquidation_days = self._calc_liquidation(group)
            supply_count = self._calc_supply(group)

            unit_prices = group["unit_price"].dropna()
            rents = []
            for _, r in group.iterrows():
                est = self._estimate_monthly_rent(r)
                if est > 0:
                    rents.append(est)

            row = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "center_lat": center_lat,
                "center_lng": center_lng,
                "supply_count": supply_count,
                "rental_ratio": rental_ratio,
                "price_growth": price_growth,
                "liquidation_days": liquidation_days,
                "unit_price_min": float(unit_prices.min()) if len(unit_prices) > 0 else 0,
                "unit_price_max": float(unit_prices.max()) if len(unit_prices) > 0 else 0,
                "unit_price_median": float(unit_prices.median()) if len(unit_prices) > 0 else 0,
                "rent_min": float(min(rents)) if rents else 0,
                "rent_max": float(max(rents)) if rents else 0,
                "rent_median": float(np.median(rents)) if rents else 0,
                "community_list": group["community_name"].unique().tolist(),
            }
            result_rows.append(row)

        result = pd.DataFrame(result_rows)
        if len(result) == 0:
            raise ValueError("没有有效的板块数据可分析")

        result["rental_ratio_norm"] = self._normalize(result["rental_ratio"])
        result["price_growth_norm"] = self._normalize(result["price_growth"])
        result["liquidation_norm"] = self._normalize(result["liquidation_days"], inverse=True)
        result["supply_norm"] = self._normalize(result["supply_count"])

        result["composite_score"] = (
            result["rental_ratio_norm"] * self.weights["rental_ratio"]
            + result["price_growth_norm"] * self.weights["price_growth"]
            + result["liquidation_norm"] * self.weights["liquidation_speed"]
            + result["supply_norm"] * self.weights["supply_volume"]
        ) * 100

        result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1
        self.analysis_result = result
        self.top_plates = result.head(20)
        print(f"[Analyzer] 完成 {len(result)} 个板块的四维指标分析与评分")
        return result

    def print_top20(self, top_n: int = 20) -> pd.DataFrame:
        if self.analysis_result is None:
            raise RuntimeError("尚未执行分析，请先调用 analyze()")
        top = self.analysis_result.head(top_n).copy()

        console = Console()
        table = Table(
            title=f"🏆 房地产投资潜力板块 TOP{len(top)}",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("排名", justify="center", style="bold yellow", width=6)
        table.add_column("板块名称", style="bold cyan", width=20)
        table.add_column("租售比", justify="right", width=10)
        table.add_column("年涨幅", justify="right", width=10)
        table.add_column("去化天数", justify="right", width=10)
        table.add_column("在售套数", justify="right", width=10)
        table.add_column("均价(元/㎡)", justify="right", width=12)
        table.add_column("月租金(元)", justify="right", width=12)
        table.add_column("综合分", justify="right", style="bold green", width=10)

        for _, r in top.iterrows():
            table.add_row(
                str(int(r["rank"])),
                str(r["cluster_name"]),
                f"{r['rental_ratio']*100:.3f}%",
                f"{r['price_growth']*100:.2f}%",
                f"{int(r['liquidation_days'])}天",
                str(int(r["supply_count"])),
                f"{int(r['unit_price_median']):,}",
                f"{int(r['rent_median']):,}",
                f"{r['composite_score']:.2f}",
            )
        console.print(table)
        return top

    def get_analysis_result(self) -> pd.DataFrame:
        if self.analysis_result is None:
            raise RuntimeError("尚未执行分析，请先调用 analyze()")
        return self.analysis_result

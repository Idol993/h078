import os
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
        self.weights_cfg = self.config["weights"]
        self.analysis_cfg = self.config.get("analysis", {})
        self.skip_missing_growth = self.analysis_cfg.get("skip_missing_price_growth", True)
        self.min_rent_ratio = self.analysis_cfg.get("min_rent_sample_ratio", 0.3)
        self.min_growth_ratio = self.analysis_cfg.get("min_growth_sample_ratio", 0.2)
        self.renormalize_weights = self.analysis_cfg.get("renormalize_weights", True)
        self.analysis_result: Optional[pd.DataFrame] = None
        self.top_plates: Optional[pd.DataFrame] = None
        self.active_weights: Dict[str, float] = {}
        self.warnings: List[str] = []

    def _resolve_weights(self, growth_available: bool) -> Dict[str, float]:
        w = dict(self.weights_cfg)
        if not growth_available and self.skip_missing_growth:
            removed = w.pop("price_growth", 0)
            self.warnings.append(
                f"⚠️ CSV中缺少可用的年涨幅数据，已按配置跳过该指标 (原权重 {removed})"
            )
            if self.renormalize_weights:
                total = sum(w.values())
                if total > 0:
                    w = {k: v / total for k, v in w.items()}
                    self.warnings.append(
                        f"📊 剩余指标权重已归一化: "
                        + ", ".join([f"{k}={v:.2f}" for k, v in w.items()])
                    )
        return w

    def _calc_rental_ratio(self, group: pd.DataFrame) -> Tuple[float, int, int, bool]:
        valid = 0
        total = len(group)
        ratios = []
        for _, row in group.iterrows():
            rent = row.get("monthly_rent")
            unit_price = row.get("unit_price", 0) or 0
            area = row.get("area", 0) or 0
            total_value = unit_price * area
            if (
                pd.notna(rent)
                and rent is not None
                and float(rent) > 0
                and total_value > 0
            ):
                ratios.append(float(rent) / total_value)
                valid += 1
        sufficient = total > 0 and (valid / total) >= self.min_rent_ratio
        if not ratios:
            return float("nan"), valid, total, sufficient
        return float(np.median(ratios)), valid, total, sufficient

    def _calc_price_growth(self, group: pd.DataFrame) -> Tuple[float, int, int, bool]:
        valid = 0
        total = len(group)
        growths = []
        for _, row in group.iterrows():
            deal = row.get("last_deal_price")
            last_year = row.get("price_last_year")
            if (
                pd.notna(deal)
                and pd.notna(last_year)
                and deal is not None
                and last_year is not None
                and float(last_year) > 0
                and float(deal) > 0
            ):
                growths.append((float(deal) - float(last_year)) / float(last_year))
                valid += 1
        sufficient = total > 0 and (valid / total) >= self.min_growth_ratio
        if not growths:
            return float("nan"), valid, total, sufficient
        return float(np.median(growths)), valid, total, sufficient

    def _calc_liquidation(self, group: pd.DataFrame) -> float:
        days = group["listing_days"].dropna()
        if len(days) == 0:
            return float("nan")
        return float(days.median())

    def _calc_supply(self, group: pd.DataFrame) -> int:
        return int(len(group))

    def _normalize(
        self, series: pd.Series, inverse: bool = False
    ) -> pd.Series:
        valid = series.dropna()
        if len(valid) == 0 or valid.std() == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        vmin, vmax = valid.min(), valid.max()
        if vmax == vmin:
            return pd.Series([0.5] * len(series), index=series.index)
        result = (series - vmin) / (vmax - vmin)
        if inverse:
            result = 1 - result
        return result.fillna(0.5)

    def analyze(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        if clustered_df is None or len(clustered_df) == 0:
            raise ValueError("聚类后的数据为空")
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("数据缺少 cluster_id 列，请先运行聚类")

        self.warnings = []
        group_cols = ["cluster_id", "cluster_name"]
        grouped = clustered_df.groupby(group_cols)

        result_rows = []
        any_growth_valid = False

        for key, group in grouped:
            cluster_id, cluster_name = key
            center_lat = group["latitude"].mean()
            center_lng = group["longitude"].mean()

            rental_ratio, rent_valid, rent_total, rent_sufficient = self._calc_rental_ratio(group)
            price_growth, growth_valid, growth_total, growth_sufficient = self._calc_price_growth(group)
            if not np.isnan(price_growth):
                any_growth_valid = True
            liquidation_days = self._calc_liquidation(group)
            supply_count = self._calc_supply(group)

            unit_prices = group["unit_price"].dropna()
            rent_series = pd.to_numeric(group["monthly_rent"], errors="coerce").dropna()
            rent_series = rent_series[rent_series > 0]

            row = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "center_lat": center_lat,
                "center_lng": center_lng,
                "supply_count": supply_count,
                "rental_ratio": rental_ratio,
                "rental_ratio_valid": rent_valid,
                "rental_ratio_total": rent_total,
                "rental_ratio_sufficient": rent_sufficient,
                "price_growth": price_growth,
                "price_growth_valid": growth_valid,
                "price_growth_total": growth_total,
                "price_growth_sufficient": growth_sufficient,
                "liquidation_days": liquidation_days,
                "unit_price_min": float(unit_prices.min()) if len(unit_prices) > 0 else 0,
                "unit_price_max": float(unit_prices.max()) if len(unit_prices) > 0 else 0,
                "unit_price_median": float(unit_prices.median()) if len(unit_prices) > 0 else 0,
                "rent_min": float(rent_series.min()) if len(rent_series) > 0 else 0,
                "rent_max": float(rent_series.max()) if len(rent_series) > 0 else 0,
                "rent_median": float(rent_series.median()) if len(rent_series) > 0 else 0,
                "community_list": group["community_name"].unique().tolist(),
            }
            latlngs = group[["latitude", "longitude"]].values.tolist()
            row["boundary_points"] = latlngs
            result_rows.append(row)

        result = pd.DataFrame(result_rows)
        if len(result) == 0:
            raise ValueError("没有有效的板块数据可分析")

        self.active_weights = self._resolve_weights(any_growth_valid)

        result["rental_ratio_norm"] = self._normalize(result["rental_ratio"])
        if "price_growth" in self.active_weights:
            result["price_growth_norm"] = self._normalize(result["price_growth"])
        else:
            result["price_growth_norm"] = 0.0
        result["liquidation_norm"] = self._normalize(
            result["liquidation_days"], inverse=True
        )
        result["supply_norm"] = self._normalize(result["supply_count"])

        def calc_score(r):
            s = 0.0
            if "rental_ratio" in self.active_weights and not np.isnan(r["rental_ratio"]):
                s += r["rental_ratio_norm"] * self.active_weights["rental_ratio"]
            if (
                "price_growth" in self.active_weights
                and not np.isnan(r["price_growth"])
            ):
                s += r["price_growth_norm"] * self.active_weights["price_growth"]
            if (
                "liquidation_speed" in self.active_weights
                and not np.isnan(r["liquidation_days"])
            ):
                s += r["liquidation_norm"] * self.active_weights["liquidation_speed"]
            if "supply_volume" in self.active_weights:
                s += r["supply_norm"] * self.active_weights["supply_volume"]
            return s * 100

        result["composite_score"] = result.apply(calc_score, axis=1)
        result = result.sort_values("composite_score", ascending=False).reset_index(
            drop=True
        )
        result["rank"] = result.index + 1
        self.analysis_result = result
        self.top_plates = result.head(20)

        if self.warnings:
            console = Console()
            for w in self.warnings:
                console.print(f"[yellow]{w}[/yellow]")

        print(f"[Analyzer] 完成 {len(result)} 个板块的四维指标分析与评分")
        return result

    def print_top20(self, top_n: int = 20) -> pd.DataFrame:
        if self.analysis_result is None:
            raise RuntimeError("尚未执行分析，请先调用 analyze()")
        top = self.analysis_result.head(top_n).copy()

        console = Console()
        active_str = ", ".join(
            [f"{k}={v:.2f}" for k, v in self.active_weights.items()]
        )
        title = f"🏆 房地产投资潜力板块 TOP{len(top)}  (权重: {active_str})"
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("排名", justify="center", style="bold yellow", width=6)
        table.add_column("板块名称", style="bold cyan", width=18)
        table.add_column("租售比", justify="right", width=12)
        table.add_column("租金样本", justify="center", width=10)
        table.add_column("年涨幅", justify="right", width=10)
        table.add_column("涨幅样本", justify="center", width=10)
        table.add_column("去化天数", justify="right", width=10)
        table.add_column("在售套数", justify="right", width=10)
        table.add_column("均价(元/㎡)", justify="right", width=12)
        table.add_column("综合分", justify="right", style="bold green", width=10)

        for _, r in top.iterrows():
            if np.isnan(r["rental_ratio"]):
                rr_str = "[dim]N/A[/dim]"
            else:
                rr_str = f"{r['rental_ratio']*100:.3f}%"
            rent_n = int(r["rental_ratio_valid"])
            rent_t = int(r["rental_ratio_total"])
            if r["rental_ratio_sufficient"]:
                rent_sample_str = f"[green]{rent_n}/{rent_t}[/green]"
            elif rent_n > 0:
                rent_sample_str = f"[yellow]{rent_n}/{rent_t}[/yellow]"
            else:
                rent_sample_str = f"[red]{rent_n}/{rent_t}[/red]"

            if np.isnan(r["price_growth"]):
                pg_str = "[dim]N/A[/dim]"
            else:
                pg_str = f"{r['price_growth']*100:.2f}%"
            pg_n = int(r["price_growth_valid"])
            pg_t = int(r["price_growth_total"])
            if "price_growth" not in self.active_weights:
                pg_sample_str = "[dim]跳过[/dim]"
            elif r["price_growth_sufficient"]:
                pg_sample_str = f"[green]{pg_n}/{pg_t}[/green]"
            elif pg_n > 0:
                pg_sample_str = f"[yellow]{pg_n}/{pg_t}[/yellow]"
            else:
                pg_sample_str = f"[red]{pg_n}/{pg_t}[/red]"

            liq_str = (
                f"{int(r['liquidation_days'])}天"
                if not np.isnan(r["liquidation_days"])
                else "[dim]N/A[/dim]"
            )
            table.add_row(
                str(int(r["rank"])),
                str(r["cluster_name"]),
                rr_str,
                rent_sample_str,
                pg_str,
                pg_sample_str,
                liq_str,
                str(int(r["supply_count"])),
                f"{int(r['unit_price_median']):,}",
                f"{r['composite_score']:.2f}",
            )
        console.print(table)
        return top

    def get_analysis_result(self) -> pd.DataFrame:
        if self.analysis_result is None:
            raise RuntimeError("尚未执行分析，请先调用 analyze()")
        return self.analysis_result

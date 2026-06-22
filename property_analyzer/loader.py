import os
import yaml
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


class PropertyLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.column_mapping = self.config["column_mapping"]
        self.city_boundary = self.config["city_boundary"]
        self.min_lat = self.city_boundary["min_lat"]
        self.max_lat = self.city_boundary["max_lat"]
        self.min_lng = self.city_boundary["min_lng"]
        self.max_lng = self.city_boundary["max_lng"]
        self.max_listing_days = self.config["data_cleaning"]["max_listing_days"]
        self.min_valid_price = self.config["data_cleaning"]["min_valid_price"]
        self.cleaned_data: Optional[pd.DataFrame] = None

    def _detect_source(self, columns: List[str]) -> str:
        all_cols_lower = {c.strip().lower() for c in columns}
        source_scores = {}
        for source, mappings in self.column_mapping.items():
            score = 0
            for std_name, aliases in mappings.items():
                for alias in aliases:
                    if alias.lower() in all_cols_lower:
                        score += 1
            source_scores[source] = score
        if max(source_scores.values()) == 0:
            return "lianjia"
        return max(source_scores, key=source_scores.get)

    def _map_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        mappings = self.column_mapping[source]
        col_lower_map = {c.strip().lower(): c for c in df.columns}
        rename_map = {}
        for std_name, aliases in mappings.items():
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in col_lower_map:
                    rename_map[col_lower_map[alias_lower]] = std_name
                    break
        df = df.rename(columns=rename_map)
        return df

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "community_name",
            "latitude",
            "longitude",
            "unit_price",
            "area",
            "listing_days",
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        if "total_price" not in df.columns:
            if "unit_price" in df.columns and "area" in df.columns:
                df["total_price"] = df["unit_price"] * df["area"] / 10000
        if "monthly_rent" not in df.columns:
            df["monthly_rent"] = np.nan
        if "last_deal_price" not in df.columns:
            df["last_deal_price"] = np.nan
        if "price_last_year" not in df.columns:
            df["price_last_year"] = np.nan
        if "layout" not in df.columns:
            df["layout"] = "未知"
        return df

    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            "latitude",
            "longitude",
            "unit_price",
            "total_price",
            "area",
            "listing_days",
            "monthly_rent",
            "last_deal_price",
            "price_last_year",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df[df["unit_price"].notna() & (df["unit_price"] > self.min_valid_price)]
        df = df[df["latitude"].notna() & df["longitude"].notna()]

        df = df[
            (df["latitude"] >= self.min_lat)
            & (df["latitude"] <= self.max_lat)
            & (df["longitude"] >= self.min_lng)
            & (df["longitude"] <= self.max_lng)
        ]
        df = df[df["listing_days"].notna()]
        df = df[df["listing_days"] <= self.max_listing_days]

        df["listing_days"] = df["listing_days"].astype(int)
        df = df.dropna(subset=["community_name"])
        df["community_name"] = df["community_name"].astype(str).str.strip()
        df = df[df["community_name"].str.len() > 0]

        cleaned_count = len(df)
        print(f"[Loader] 数据清洗: 原始 {initial_count} 条 -> 清洗后 {cleaned_count} 条 (剔除 {initial_count - cleaned_count})")
        return df.reset_index(drop=True)

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"[Loader] 加载 CSV: {csv_path} ({len(df)} 条记录, 列名: {list(df.columns)})")

        source = self._detect_source(df.columns)
        print(f"[Loader] 自动识别数据源: {source}")
        df = self._map_columns(df, source)
        df = self._ensure_required_columns(df)
        df = self._convert_numeric(df)
        df = self._clean_data(df)
        self.cleaned_data = df
        return df

    def get_cleaned_data(self) -> pd.DataFrame:
        if self.cleaned_data is None:
            raise RuntimeError("尚未加载数据，请先调用 load_csv()")
        return self.cleaned_data

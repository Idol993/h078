import yaml
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from sklearn.cluster import DBSCAN
from collections import Counter


class PropertyCluster:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.eps = self.config["cluster"]["eps"]
        self.min_samples = self.config["cluster"]["min_samples"]
        self.clustered_data: Optional[pd.DataFrame] = None
        self.cluster_info: Dict[int, Dict] = {}

    def _extract_name_prefix(self, name: str) -> str:
        if not isinstance(name, str) or len(name) == 0:
            return "未知"
        name = name.strip()
        suffixes = [
            "花园", "小区", "家园", "公寓", "苑", "城", "府", "邸", "庭",
            "阁", "院", "轩", "居", "楼", "邨", "坊", "街", "道", "路",
            "大厦", "广场", "中心", "里", "弄", "号", "一期", "二期", "三期",
            "东区", "西区", "南区", "北区", "A区", "B区", "C区", "D区",
        ]
        pattern = "(" + "|".join(re.escape(s) for s in suffixes) + ").*$"
        prefix = re.sub(pattern, "", name)
        if len(prefix) < 2:
            prefix = name[: min(4, len(name))]
        return prefix

    def _generate_cluster_name(self, community_names: List[str]) -> str:
        prefixes = [self._extract_name_prefix(n) for n in community_names if isinstance(n, str)]
        if not prefixes:
            return "未知板块"
        counter = Counter(prefixes)
        top_prefix, count = counter.most_common(1)[0]
        if count >= max(2, len(prefixes) * 0.3):
            return f"{top_prefix}板块"
        full_names_counter = Counter([n for n in community_names if isinstance(n, str)])
        top_name = full_names_counter.most_common(1)[0][0]
        return f"{self._extract_name_prefix(top_name)}板块"

    def run_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("输入数据为空")
        required = ["latitude", "longitude", "community_name"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"缺少列: {col}")

        coords = df[["latitude", "longitude"]].values
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(coords)

        df = df.copy()
        df["cluster_id"] = labels
        valid_clusters = [l for l in set(labels) if l != -1]
        noise_count = list(labels).count(-1)
        print(f"[Cluster] DBSCAN 聚类: {len(valid_clusters)} 个有效板块, 噪声点 {noise_count} 个")

        df = df[df["cluster_id"] != -1].reset_index(drop=True)
        self.cluster_info = {}

        for cid in valid_clusters:
            cluster_df = df[df["cluster_id"] == cid]
            name = self._generate_cluster_name(cluster_df["community_name"].tolist())
            info = {
                "cluster_id": cid,
                "cluster_name": name,
                "center_lat": cluster_df["latitude"].mean(),
                "center_lng": cluster_df["longitude"].mean(),
                "property_count": len(cluster_df),
                "community_names": cluster_df["community_name"].unique().tolist(),
            }
            self.cluster_info[cid] = info
            df.loc[df["cluster_id"] == cid, "cluster_name"] = name

        df["cluster_name"] = df["cluster_name"].fillna("未知板块")
        self.clustered_data = df
        print(f"[Cluster] 已为 {len(df)} 条记录分配板块归属")
        return df

    def get_cluster_info(self) -> Dict[int, Dict]:
        if not self.cluster_info:
            raise RuntimeError("尚未执行聚类，请先调用 run_clustering()")
        return self.cluster_info

    def get_clustered_data(self) -> pd.DataFrame:
        if self.clustered_data is None:
            raise RuntimeError("尚未执行聚类，请先调用 run_clustering()")
        return self.clustered_data

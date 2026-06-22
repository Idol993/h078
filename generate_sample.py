import pandas as pd
import numpy as np

np.random.seed(42)

centers = [
    {"name": "光谷天地", "lat": 30.482, "lng": 114.410, "base_price": 22000, "quality": 1.5},
    {"name": "金融港", "lat": 30.455, "lng": 114.385, "base_price": 19500, "quality": 1.3},
    {"name": "关山大道", "lat": 30.505, "lng": 114.395, "base_price": 26000, "quality": 1.6},
    {"name": "武昌火车站", "lat": 30.530, "lng": 114.320, "base_price": 18000, "quality": 0.9},
    {"name": "楚河汉街", "lat": 30.550, "lng": 114.340, "base_price": 32000, "quality": 1.8},
    {"name": "洪山广场", "lat": 30.545, "lng": 114.330, "base_price": 28000, "quality": 1.4},
    {"name": "街道口", "lat": 30.530, "lng": 114.360, "base_price": 25000, "quality": 1.3},
    {"name": "中南财大", "lat": 30.480, "lng": 114.370, "base_price": 20000, "quality": 1.1},
    {"name": "白沙洲", "lat": 30.480, "lng": 114.290, "base_price": 14000, "quality": 0.7},
    {"name": "南湖新城", "lat": 30.495, "lng": 114.330, "base_price": 17000, "quality": 1.0},
    {"name": "四新大道", "lat": 30.520, "lng": 114.260, "base_price": 15500, "quality": 0.8},
    {"name": "钟家村", "lat": 30.550, "lng": 114.265, "base_price": 19000, "quality": 1.0},
    {"name": "汉口火车站", "lat": 30.620, "lng": 114.250, "base_price": 21000, "quality": 1.2},
    {"name": "江汉路", "lat": 30.580, "lng": 114.280, "base_price": 29000, "quality": 1.7},
    {"name": "武广万松园", "lat": 30.590, "lng": 114.265, "base_price": 31000, "quality": 1.6},
    {"name": "后湖大道", "lat": 30.650, "lng": 114.270, "base_price": 16500, "quality": 0.9},
    {"name": "盘龙城", "lat": 30.710, "lng": 114.240, "base_price": 9500, "quality": 0.5},
    {"name": "青山红钢城", "lat": 30.630, "lng": 114.380, "base_price": 13500, "quality": 0.7},
    {"name": "徐东商圈", "lat": 30.580, "lng": 114.350, "base_price": 23000, "quality": 1.4},
    {"name": "东西湖吴家山", "lat": 30.620, "lng": 114.130, "base_price": 10500, "quality": 0.6},
    {"name": "沌口开发区", "lat": 30.480, "lng": 114.160, "base_price": 12500, "quality": 0.8},
    {"name": "江夏纸坊", "lat": 30.350, "lng": 114.320, "base_price": 9800, "quality": 0.5},
]

suffix_pool = [
    "花园", "小区", "家园", "公寓", "雅苑", "新城", "华府", "府邸",
    "华庭", "国际", "广场", "中心", "名邸", "丽舍", "春天", "阳光",
    "金座", "银座", "翠湖", "山庄", "里", "苑",
]

layouts = ["1室1厅", "2室1厅", "2室2厅", "3室1厅", "3室2厅", "4室2厅", "5室3厅"]
layout_weights = [5, 18, 22, 25, 18, 10, 2]
layout_area_range = {
    "1室1厅": (40, 65),
    "2室1厅": (65, 85),
    "2室2厅": (80, 95),
    "3室1厅": (90, 110),
    "3室2厅": (105, 130),
    "4室2厅": (125, 160),
    "5室3厅": (155, 220),
}

records = []
n_per_center = 20000 // len(centers) + 50

for center in centers:
    for i in range(np.random.randint(n_per_center - 50, n_per_center + 150)):
        offset_lat = np.random.normal(0, 0.0025)
        offset_lng = np.random.normal(0, 0.0025)
        lat = center["lat"] + offset_lat
        lng = center["lng"] + offset_lng

        dist = np.sqrt(offset_lat**2 + offset_lng**2)
        price_decay = max(0.75, 1.0 - dist * 80)
        unit_price = center["base_price"] * price_decay * np.random.normal(1, 0.08)
        unit_price = max(5000, min(50000, unit_price))

        layout = np.random.choice(layouts, p=[w / sum(layout_weights) for w in layout_weights])
        area_min, area_max = layout_area_range[layout]
        area = np.random.uniform(area_min, area_max)
        total_price = unit_price * area / 10000

        rent_ratio = np.random.normal(1 / 380 * center["quality"], 0.0005)
        monthly_rent = max(500, unit_price * area * rent_ratio)

        listing_days = int(np.random.exponential(80 / max(0.5, center["quality"])))
        listing_days = min(600, listing_days)

        growth = np.random.normal(0.05 * center["quality"], 0.04)
        price_last_year = unit_price / (1 + growth)
        last_deal_price = unit_price * np.random.normal(1, 0.03)

        cn_idx = np.random.randint(0, len(suffix_pool))
        community = f"{center['name'][:2]}{suffix_pool[cn_idx]}{np.random.randint(1, 20)}"

        if np.random.random() < 0.01:
            unit_price = 0
        if np.random.random() < 0.005:
            lat = 50.0
            lng = 130.0
        if np.random.random() < 0.008:
            listing_days = 700 + np.random.randint(0, 500)

        records.append({
            "小区名称": community,
            "纬度": round(lat, 6),
            "经度": round(lng, 6),
            "单价(元/㎡)": int(unit_price),
            "总价": round(total_price, 2),
            "建筑面积(㎡)": round(area, 2),
            "户型": layout,
            "挂牌天数": listing_days,
            "月租金": int(monthly_rent),
            "近期成交价": int(last_deal_price),
            "去年同期价": int(price_last_year),
        })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
out_path = "sample_property_data.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"生成模拟数据: {out_path} ({len(df)} 条记录)")
print(df.head())

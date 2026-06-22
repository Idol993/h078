import os
import sys
import click
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

CACHE_DIR = Path(".property_cache")
CACHE_DIR.mkdir(exist_ok=True)

CLEANED_CSV = CACHE_DIR / "cleaned.csv"
CLUSTERED_CSV = CACHE_DIR / "clustered.csv"
ANALYSIS_CSV = CACHE_DIR / "analysis.csv"


@click.group(
    help="""
🏡 房地产投资潜力分析工具

支持链式调用：
  1. property load data.csv      → 加载清洗CSV
  2. property analyze            → 四维指标分析+TOP20
  3. property map                → 生成交互式热力图
  4. property all data.csv       → 一键跑完全流程
""",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    pass


@cli.command("load")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "-c", default="config.yaml", show_default=True, help="配置文件路径")
def cmd_load(csv_path: str, config: str):
    """加载CSV，自动识别列名映射，执行数据清洗"""
    from loader import PropertyLoader

    console.print(Panel.fit(f"📥 加载并清洗数据: {csv_path}", border_style="cyan"))
    loader = PropertyLoader(config_path=config)
    cleaned = loader.load_csv(csv_path)
    cleaned.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
    console.print(f"✅ 清洗完成，缓存至: [green]{CLEANED_CSV}[/green] (共 {len(cleaned)} 条)")


@cli.command("analyze")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="配置文件路径")
@click.option("--top", "-t", default=20, show_default=True, help="显示TOP N板块")
@click.option("--export", "-e", default=None, type=click.Path(), help="导出分析结果CSV路径")
def cmd_analyze(config: str, top: int, export: str):
    """DBSCAN地理聚类 + 四维指标加权评分，输出TOP20潜力板块"""
    from loader import PropertyLoader
    from cluster import PropertyCluster
    from analyzer import PropertyAnalyzer

    if not CLEANED_CSV.exists():
        console.print("[red]❌ 未找到清洗后的数据，请先执行 property load <csv>[/red]")
        sys.exit(1)

    console.print(Panel.fit("🧠 地理聚类 + 四维指标分析", border_style="magenta"))
    cleaned = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")
    console.print(f"📊 读取清洗数据: {len(cleaned)} 条")

    cluster = PropertyCluster(config_path=config)
    clustered = cluster.run_clustering(cleaned)
    clustered.to_csv(CLUSTERED_CSV, index=False, encoding="utf-8-sig")
    console.print(f"✅ 聚类缓存至: [green]{CLUSTERED_CSV}[/green]")

    analyzer = PropertyAnalyzer(config_path=config)
    result = analyzer.analyze(clustered)
    result.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    console.print(f"✅ 分析缓存至: [green]{ANALYSIS_CSV}[/green]")

    analyzer.print_top20(top_n=top)

    if export:
        result.head(top).to_csv(export, index=False, encoding="utf-8-sig")
        console.print(f"✅ 导出TOP{top}结果: [green]{export}[/green]")


@cli.command("map")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="配置文件路径")
@click.option("--output", "-o", default=None, type=click.Path(), help="输出HTML路径")
@click.option("--top", "-t", default=None, type=int, help="仅渲染TOP N板块，默认全部")
def cmd_map(config: str, output: str, top: int):
    """生成folium交互式HTML热力图，点击查看板块详情"""
    from mapper import PropertyMapper

    if not ANALYSIS_CSV.exists():
        console.print("[red]❌ 未找到分析结果，请先执行 property analyze[/red]")
        sys.exit(1)

    console.print(Panel.fit("🗺️ 生成交互式热力地图", border_style="green"))
    analysis = pd.read_csv(ANALYSIS_CSV, encoding="utf-8-sig")

    mapper = PropertyMapper(config_path=config)
    out = mapper.generate_map(analysis, output_path=output, top_n=top)
    console.print(f"🎉 地图生成成功! 请用浏览器打开: [link=file:///{out}]{out}[/link]")


@cli.command("all")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "-c", default="config.yaml", show_default=True, help="配置文件路径")
@click.option("--top", "-t", default=20, show_default=True, help="显示TOP N板块")
@click.option("--map-output", "-o", default=None, type=click.Path(), help="地图HTML输出路径")
@click.option("--export", "-e", default=None, type=click.Path(), help="导出分析结果CSV路径")
def cmd_all(csv_path: str, config: str, top: int, map_output: str, export: str):
    """一键执行: 加载 → 聚类分析 → 输出排行榜 → 生成地图"""
    from loader import PropertyLoader
    from cluster import PropertyCluster
    from analyzer import PropertyAnalyzer
    from mapper import PropertyMapper

    console.print(Panel.fit("🚀 一键全流程分析", border_style="bold yellow"))

    console.rule("[cyan]Step 1/4: 加载与清洗[/cyan]")
    loader = PropertyLoader(config_path=config)
    cleaned = loader.load_csv(csv_path)
    cleaned.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")

    console.rule("[magenta]Step 2/4: DBSCAN地理聚类[/magenta]")
    cluster = PropertyCluster(config_path=config)
    clustered = cluster.run_clustering(cleaned)
    clustered.to_csv(CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    console.rule("[blue]Step 3/4: 四维指标与加权评分[/blue]")
    analyzer = PropertyAnalyzer(config_path=config)
    result = analyzer.analyze(clustered)
    result.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    analyzer.print_top20(top_n=top)

    if export:
        result.head(top).to_csv(export, index=False, encoding="utf-8-sig")
        console.print(f"✅ 导出TOP{top}结果: [green]{export}[/green]")

    console.rule("[green]Step 4/4: 生成交互式地图[/green]")
    mapper = PropertyMapper(config_path=config)
    out = mapper.generate_map(result, output_path=map_output)
    console.rule("[bold green]全部完成![/bold green]")
    console.print(f"🎉 地图文件: [link=file:///{out}]{out}[/link]")


if __name__ == "__main__":
    cli()

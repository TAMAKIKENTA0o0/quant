import nbformat as nbf
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "huaxia_fund_analysis_adata.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

cells = [
    md(
        """
        # 华夏 ETF 基金分析、配置、卖出与回测流程

        这个 notebook 使用 `adata` 获取场内 ETF 数据，围绕“华夏基金里面怎么选、怎么配、怎么卖”搭建一套可复用的量化分析流程。

        重要边界：
        - `adata.fund` 当前主要覆盖场内 ETF 行情，因此本流程默认分析“华夏系 ETF”，适合可交易组合、行业/主题轮动、核心卫星配置等场景。
        - 如果后续要覆盖开放式公募基金，需要另接基金净值、基金经理、持仓、规模、费率等数据源；本 notebook 已把筛选/评分/回测接口写成可替换结构。
        - 以下内容是研究框架，不构成投资建议。请按自己的风险承受能力、交易成本、税费和资金规模调整参数。
        """
    ),
    md(
        """
        ## 1. 环境与参数

        第一次运行如缺包，可取消下一格的安装命令注释。参数集中放在这里，便于快速调整分析范围、流动性门槛、组合数量和风控规则。
        """
    ),
    code(
        """
        # %pip install adata pandas numpy matplotlib seaborn scipy openpyxl requests -q

        import json
        import math
        import time
        import warnings
        from pathlib import Path

        import adata
        import numpy as np
        import pandas as pd
        import requests
        import matplotlib.pyplot as plt
        import seaborn as sns

        warnings.filterwarnings("ignore")
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.width", 160)

        DATA_DIR = Path("data_huaxia_adata")
        PRICE_DIR = DATA_DIR / "prices"
        DATA_DIR.mkdir(exist_ok=True)
        PRICE_DIR.mkdir(exist_ok=True)

        CONFIG = {
            "manager_keyword": "华夏",
            "start_date": "20180101",
            "end_date": "",                 # 留空表示到最新交易日
            "min_history_days": 180,
            "min_avg_amount": 20_000_000,    # 近 60 日日均成交额，单位：元
            "top_n": 8,
            "max_weight": 0.25,
            "max_category_weight": 0.45,
            "rebalance_freq": "ME",          # ME: 月度；W-FRI: 周度；QE: 季度
            "risk_free_rate": 0.02,
            "fee_rate": 0.001,               # 单边交易成本估计
            "stop_loss": -0.08,
            "trailing_stop": -0.12,
            "trend_fast": 60,
            "trend_slow": 120,
        }

        print("adata version:", getattr(adata, "__version__", "unknown"))
        """
    ),
    md(
        """
        ## 2. 数据获取：华夏 ETF 基金池

        先尝试用 `adata.fund.info.all_etf_exchange_traded_info()` 获取全市场 ETF 列表。若上游返回非 JSONP 导致 `adata` 解析失败，则使用同一个东方财富接口的 JSON 兜底，并缓存到本地。
        """
    ),
    code(
        """
        def fetch_all_etf_info_with_adata(retries=3, wait=1):
            last_err = None
            for i in range(retries):
                try:
                    df = adata.fund.info.all_etf_exchange_traded_info(wait_time=wait)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df.assign(source="adata")
                except Exception as exc:
                    last_err = exc
                    time.sleep(wait * (i + 1))
            raise RuntimeError(f"adata ETF 列表接口失败: {last_err}")


        def fetch_all_etf_info_fallback():
            rows = []
            url = "https://68.push2delay.eastmoney.com/api/qt/clist/get"
            headers = {"User-Agent": "Mozilla/5.0"}
            params_base = {
                "pz": 100,
                "po": 1,
                "np": 1,
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": 2,
                "invt": 2,
                "wbp2u": "|0|0|0|web",
                "fid": "f3",
                "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
                "fields": "f12,f14,f2",
            }
            for page in range(1, 60):
                params = {**params_base, "pn": page}
                last_err = None
                for attempt in range(3):
                    try:
                        r = requests.get(url, params=params, headers=headers, timeout=10)
                        r.raise_for_status()
                        break
                    except Exception as exc:
                        last_err = exc
                        time.sleep(1 + attempt)
                else:
                    raise RuntimeError(f"ETF 列表兜底接口失败，第 {page} 页: {last_err}")
                data = r.json().get("data")
                if not data or not data.get("diff"):
                    break
                for item in data["diff"]:
                    rows.append({
                        "fund_code": str(item.get("f12", "")).zfill(6),
                        "short_name": item.get("f14"),
                        "net_value": item.get("f2"),
                    })
            return pd.DataFrame(rows).assign(source="eastmoney_fallback")


        def load_all_etf_info(force_refresh=False):
            cache = DATA_DIR / "all_etf_info.csv"
            if cache.exists() and not force_refresh:
                return pd.read_csv(cache, dtype={"fund_code": str})
            try:
                df = fetch_all_etf_info_with_adata()
            except Exception as exc:
                print(exc)
                df = fetch_all_etf_info_fallback()
            df["fund_code"] = df["fund_code"].astype(str).str.zfill(6)
            df.to_csv(cache, index=False, encoding="utf-8-sig")
            return df


        all_etf = load_all_etf_info(force_refresh=False)
        huaxia_pool = (
            all_etf[all_etf["short_name"].astype(str).str.contains(CONFIG["manager_keyword"], na=False)]
            .drop_duplicates("fund_code")
            .sort_values("fund_code")
            .reset_index(drop=True)
        )

        print("全市场 ETF 数量:", len(all_etf))
        print("华夏 ETF 数量:", len(huaxia_pool))
        display(huaxia_pool.head(30))
        """
    ),
    md(
        """
        ## 3. 基金分类

        ETF 名称里通常包含资产类别、行业或主题关键词。这里用规则先粗分，便于后续做分散化配置和类别上限。真实投研中可以继续接入跟踪指数、申万行业、基金合同、持仓等更细数据。
        """
    ),
    code(
        """
        CATEGORY_RULES = [
            ("债券/现金", ["债", "货币", "现金", "短融", "政金债", "国债", "信用债", "可转债"]),
            ("海外/跨境", ["纳斯达克", "标普", "恒生", "港股", "日经", "德国", "法国", "沙特", "东南亚", "海外", "QDII"]),
            ("宽基", ["沪深300", "上证50", "中证500", "中证1000", "创业板", "科创50", "A500", "MSCI", "上证综指", "深证"]),
            ("金融地产", ["银行", "证券", "保险", "金融", "地产"]),
            ("科技成长", ["芯片", "半导体", "人工智能", "云计算", "软件", "信息", "通信", "5G", "机器人", "科创", "游戏"]),
            ("医药消费", ["医药", "医疗", "创新药", "生物", "消费", "食品", "酒", "家电", "旅游"]),
            ("新能源制造", ["新能源", "光伏", "电池", "汽车", "智能车", "高端装备", "军工", "机械"]),
            ("周期资源", ["有色", "稀有金属", "煤炭", "钢铁", "化工", "能源", "油气", "资源"]),
            ("红利价值", ["红利", "价值", "低波", "央企", "国企", "高股息"]),
        ]

        def classify_fund(name):
            name = str(name)
            for category, keywords in CATEGORY_RULES:
                if any(k in name for k in keywords):
                    return category
            return "其他主题"

        huaxia_pool["category"] = huaxia_pool["short_name"].apply(classify_fund)
        display(huaxia_pool.groupby("category").size().sort_values(ascending=False).to_frame("count"))
        display(huaxia_pool)
        """
    ),
    md(
        """
        ## 4. 下载与整理历史行情

        使用 `adata.fund.market.get_market_etf()` 下载每只 ETF 日线。函数带本地缓存和重试，避免重复打接口；行情字段包括开高低收、成交量、成交额等。
        """
    ),
    code(
        """
        def get_etf_price(code, start_date=None, end_date=None, force_refresh=False):
            start_date = start_date or CONFIG["start_date"]
            end_date = end_date if end_date is not None else CONFIG["end_date"]
            cache = PRICE_DIR / f"{code}_{start_date}_{end_date or 'latest'}.csv"
            if cache.exists() and not force_refresh:
                return pd.read_csv(cache, dtype={"fund_code": str}, parse_dates=["trade_date"])

            last_err = None
            for i in range(3):
                try:
                    df = adata.fund.market.get_market_etf(
                        fund_code=str(code).zfill(6),
                        k_type=1,
                        start_date=start_date,
                        end_date=end_date or "",
                    )
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df["fund_code"] = df["fund_code"].astype(str).str.zfill(6)
                        df["trade_date"] = pd.to_datetime(df["trade_date"])
                        for col in ["open", "high", "low", "close", "volume", "amount", "change", "change_pct"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        df = df.sort_values("trade_date").drop_duplicates(["fund_code", "trade_date"])
                        df.to_csv(cache, index=False, encoding="utf-8-sig")
                        return df
                except Exception as exc:
                    last_err = exc
                    time.sleep(1 + i)
            print(f"{code} 下载失败: {last_err}")
            return pd.DataFrame()


        price_frames = []
        for i, row in huaxia_pool.iterrows():
            df = get_etf_price(row["fund_code"])
            if not df.empty:
                df["short_name"] = row["short_name"]
                df["category"] = row["category"]
                price_frames.append(df)
            if (i + 1) % 20 == 0:
                print(f"已处理 {i + 1}/{len(huaxia_pool)}")

        prices_long = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
        print("行情记录数:", len(prices_long), "覆盖 ETF:", prices_long["fund_code"].nunique() if not prices_long.empty else 0)
        display(prices_long.tail())
        """
    ),
    code(
        """
        close = prices_long.pivot(index="trade_date", columns="fund_code", values="close").sort_index()
        amount = prices_long.pivot(index="trade_date", columns="fund_code", values="amount").sort_index()
        returns = close.pct_change()

        meta = huaxia_pool.set_index("fund_code")[["short_name", "category", "net_value"]]
        print("价格矩阵:", close.shape, "日期:", close.index.min(), "->", close.index.max())
        """
    ),
    md(
        """
        ## 5. 单基金指标与评分

        评分思路：
        - 先用硬门槛过滤：上市时间、近 60 日流动性。
        - 再综合收益动量、夏普、最大回撤、波动率、流动性。
        - 分数只用于横向排序，不代表未来收益承诺。
        """
    ),
    code(
        """
        def max_drawdown(nav):
            nav = pd.Series(nav).dropna()
            if nav.empty:
                return np.nan
            dd = nav / nav.cummax() - 1
            return dd.min()

        def annualized_return(ret):
            ret = pd.Series(ret).dropna()
            if ret.empty:
                return np.nan
            return (1 + ret).prod() ** (252 / len(ret)) - 1

        def annualized_vol(ret):
            return pd.Series(ret).dropna().std() * np.sqrt(252)

        def sharpe_ratio(ret, rf=0.02):
            ret = pd.Series(ret).dropna()
            vol = annualized_vol(ret)
            if not vol or np.isnan(vol):
                return np.nan
            return (annualized_return(ret) - rf) / vol

        def zscore(s, winsor=0.03):
            s = pd.Series(s).astype(float)
            lo, hi = s.quantile(winsor), s.quantile(1 - winsor)
            s = s.clip(lo, hi)
            std = s.std()
            if std == 0 or np.isnan(std):
                return s * 0
            return (s - s.mean()) / std

        def calc_metrics(asof=None):
            c = close.loc[:asof] if asof is not None else close
            r = c.pct_change()
            rows = []
            for code in c.columns:
                px = c[code].dropna()
                if len(px) < 2:
                    continue
                ret = r[code].dropna()
                amt = amount[code].loc[px.index].dropna() if code in amount else pd.Series(dtype=float)
                nav = px / px.iloc[0]
                rows.append({
                    "fund_code": code,
                    "short_name": meta.loc[code, "short_name"] if code in meta.index else code,
                    "category": meta.loc[code, "category"] if code in meta.index else "未知",
                    "first_date": px.index.min(),
                    "last_date": px.index.max(),
                    "history_days": len(px),
                    "return_1m": px.iloc[-1] / px.iloc[-21] - 1 if len(px) > 21 else np.nan,
                    "return_3m": px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else np.nan,
                    "return_6m": px.iloc[-1] / px.iloc[-126] - 1 if len(px) > 126 else np.nan,
                    "return_1y": px.iloc[-1] / px.iloc[-252] - 1 if len(px) > 252 else np.nan,
                    "ann_return": annualized_return(ret),
                    "ann_vol": annualized_vol(ret),
                    "sharpe": sharpe_ratio(ret, CONFIG["risk_free_rate"]),
                    "max_drawdown": max_drawdown(nav),
                    "avg_amount_60d": amt.tail(60).mean() if not amt.empty else np.nan,
                    "close": px.iloc[-1],
                })
            out = pd.DataFrame(rows)
            if out.empty:
                return out
            valid = (
                (out["history_days"] >= CONFIG["min_history_days"])
                & (out["avg_amount_60d"] >= CONFIG["min_avg_amount"])
            )
            out["eligible"] = valid
            out["score"] = (
                0.25 * zscore(out["return_1y"].fillna(out["return_6m"]))
                + 0.20 * zscore(out["return_6m"])
                + 0.20 * zscore(out["sharpe"])
                + 0.15 * zscore(out["max_drawdown"])       # 回撤越接近 0 越好
                - 0.10 * zscore(out["ann_vol"])
                + 0.10 * zscore(np.log1p(out["avg_amount_60d"]))
            )
            out.loc[~out["eligible"], "score"] = np.nan
            return out.sort_values("score", ascending=False)

        metrics = calc_metrics()
        display(metrics.head(25))
        """
    ),
    md(
        """
        ## 6. 怎么选：候选池与相关性

        初筛取分数靠前的基金，但需要看相关性，避免买了一堆名字不同、实际暴露高度相似的 ETF。
        """
    ),
    code(
        """
        candidates = metrics[metrics["eligible"]].head(max(CONFIG["top_n"] * 2, 12)).copy()
        display(candidates[["fund_code", "short_name", "category", "score", "return_6m", "return_1y", "sharpe", "max_drawdown", "avg_amount_60d"]])

        candidate_codes = candidates["fund_code"].tolist()
        corr = returns[candidate_codes].tail(252).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
        plt.title("候选 ETF 近一年收益相关性")
        plt.show()
        """
    ),
    md(
        """
        ## 7. 怎么配：风险预算 + 类别约束

        这里采用简单、稳健、可解释的配置方法：
        - 从分数排名靠前的 ETF 中逐个纳入。
        - 单只权重按近一年波动率倒数分配。
        - 单只和单类别设置上限，避免组合被单一主题绑架。
        """
    ),
    code(
        """
        def cap_and_normalize(weights, max_weight=0.25, max_iter=20):
            w = pd.Series(weights, dtype=float).clip(lower=0)
            if w.sum() <= 0:
                return w
            w = w / w.sum()
            for _ in range(max_iter):
                over = w > max_weight
                if not over.any():
                    break
                excess = (w[over] - max_weight).sum()
                w[over] = max_weight
                under = ~over
                if w[under].sum() > 0:
                    w[under] += excess * w[under] / w[under].sum()
            return w / w.sum()

        def enforce_category_cap(weights, categories, cap=0.45, max_iter=20):
            w = pd.Series(weights, dtype=float)
            categories = pd.Series(categories)
            for _ in range(max_iter):
                cat_sum = w.groupby(categories).sum()
                over_cats = cat_sum[cat_sum > cap]
                if over_cats.empty:
                    break
                for cat, total in over_cats.items():
                    idx = categories[categories == cat].index
                    w.loc[idx] *= cap / total
                spare = 1 - w.sum()
                under_idx = categories[~categories.isin(over_cats.index)].index
                if spare > 1e-8 and len(under_idx) and w.loc[under_idx].sum() > 0:
                    w.loc[under_idx] += spare * w.loc[under_idx] / w.loc[under_idx].sum()
            return w / w.sum()

        def build_portfolio(metric_df, top_n=None):
            top_n = top_n or CONFIG["top_n"]
            selected = metric_df[metric_df["eligible"]].sort_values("score", ascending=False).head(top_n).copy()
            if selected.empty:
                return selected
            vol = selected.set_index("fund_code")["ann_vol"].replace(0, np.nan)
            raw_w = (1 / vol).replace([np.inf, -np.inf], np.nan).fillna(0)
            w = cap_and_normalize(raw_w, CONFIG["max_weight"])
            cats = selected.set_index("fund_code")["category"]
            w = enforce_category_cap(w, cats, CONFIG["max_category_weight"])
            selected["target_weight"] = selected["fund_code"].map(w)
            return selected.sort_values("target_weight", ascending=False)

        portfolio = build_portfolio(metrics)
        display(portfolio[["fund_code", "short_name", "category", "target_weight", "score", "ann_vol", "sharpe", "max_drawdown", "avg_amount_60d"]])

        plt.figure(figsize=(9, 4))
        portfolio.set_index("short_name")["target_weight"].sort_values().plot(kind="barh")
        plt.title("当前目标组合权重")
        plt.xlabel("weight")
        plt.show()
        """
    ),
    md(
        """
        ## 8. 怎么卖：卖出与调仓信号

        卖出规则建议分三类：
        - 风控卖出：持有期跌破止损，或从持有后高点回撤超过阈值。
        - 趋势卖出：短均线下穿长均线，说明趋势结构变弱。
        - 组合卖出：评分跌出候选池、流动性恶化，或月度调仓被更优标的替代。

        下面用“假设当前持仓等于上一节目标组合”做当前卖出信号示例；实盘可把 `entry_date`、`entry_price`、真实仓位替换成自己的持仓记录。
        """
    ),
    code(
        """
        def sell_signals(holdings, entry_lookback=63):
            rows = []
            for _, h in holdings.iterrows():
                code = h["fund_code"]
                px = close[code].dropna()
                if len(px) < max(CONFIG["trend_slow"], entry_lookback) + 5:
                    continue
                entry_date = px.index[-entry_lookback]
                entry_price = px.loc[entry_date]
                latest = px.iloc[-1]
                since_entry = px.loc[entry_date:]
                pnl = latest / entry_price - 1
                trail_dd = latest / since_entry.cummax().iloc[-1] - 1
                ma_fast = px.rolling(CONFIG["trend_fast"]).mean()
                ma_slow = px.rolling(CONFIG["trend_slow"]).mean()
                trend_bad = bool(ma_fast.iloc[-1] < ma_slow.iloc[-1])
                rank = metrics.reset_index(drop=True).query("fund_code == @code").index
                rank = int(rank[0] + 1) if len(rank) else None
                score_bad = rank is None or rank > CONFIG["top_n"] * 2
                liquidity_bad = h["avg_amount_60d"] < CONFIG["min_avg_amount"]
                reasons = []
                if pnl <= CONFIG["stop_loss"]:
                    reasons.append("止损")
                if trail_dd <= CONFIG["trailing_stop"]:
                    reasons.append("移动止盈/回撤")
                if trend_bad:
                    reasons.append("趋势转弱")
                if score_bad:
                    reasons.append("评分跌出候选池")
                if liquidity_bad:
                    reasons.append("流动性不足")
                rows.append({
                    "fund_code": code,
                    "short_name": h["short_name"],
                    "target_weight": h.get("target_weight", np.nan),
                    "entry_date": entry_date.date(),
                    "entry_price": entry_price,
                    "latest_price": latest,
                    "holding_return": pnl,
                    "trailing_drawdown": trail_dd,
                    "ma_fast": ma_fast.iloc[-1],
                    "ma_slow": ma_slow.iloc[-1],
                    "rank": rank,
                    "action": "SELL/REDUCE" if reasons else "HOLD",
                    "reason": ",".join(reasons),
                })
            return pd.DataFrame(rows)

        current_sell_table = sell_signals(portfolio)
        display(current_sell_table)
        """
    ),
    md(
        """
        ## 9. 回测系统：月度滚动选基与配置

        回测逻辑：
        - 每个调仓日只使用当日之前可见的数据计算指标，避免未来函数。
        - 从华夏 ETF 池中筛选合格基金，按评分选前 N。
        - 用波动率倒数配置，并加单只/类别上限。
        - 持有到下一调仓日，扣除换手成本。
        """
    ),
    code(
        """
        def get_rebalance_dates(price_index, freq="M", warmup_days=252):
            freq_alias = {"M": "ME", "Q": "QE"}.get(freq, freq)
            idx = pd.DatetimeIndex(price_index).sort_values()
            idx = idx[idx >= idx[min(warmup_days, len(idx) - 1)]]
            if len(idx) == 0:
                return []
            grouped = pd.Series(index=idx, data=idx).resample(freq_alias).last().dropna()
            return list(pd.DatetimeIndex(grouped.values))

        def backtest_strategy():
            daily_ret = returns.copy()
            rebalance_dates = get_rebalance_dates(close.index, CONFIG["rebalance_freq"], warmup_days=252)
            weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            holdings_log = []
            prev_w = pd.Series(0.0, index=close.columns)

            for dt in rebalance_dates:
                m = calc_metrics(asof=dt)
                p = build_portfolio(m)
                if p.empty:
                    continue
                w = pd.Series(0.0, index=close.columns)
                w.loc[p["fund_code"]] = p.set_index("fund_code")["target_weight"]
                turnover = (w - prev_w).abs().sum()
                weights.loc[dt:, :] = w.values
                log = p[["fund_code", "short_name", "category", "target_weight", "score"]].copy()
                log["rebalance_date"] = dt
                log["turnover"] = turnover
                holdings_log.append(log)
                prev_w = w

            # 当天收盘形成权重，下一交易日开始享受收益。
            shifted_w = weights.shift(1).fillna(0)
            gross_ret = (shifted_w * daily_ret).sum(axis=1).fillna(0)

            # 成本在调仓日的下一天扣除，近似处理。
            turnover_daily = weights.diff().abs().sum(axis=1).fillna(0)
            cost = turnover_daily * CONFIG["fee_rate"]
            net_ret = gross_ret - cost
            nav = (1 + net_ret).cumprod()
            log_df = pd.concat(holdings_log, ignore_index=True) if holdings_log else pd.DataFrame()
            return nav, net_ret, weights, log_df

        strategy_nav, strategy_ret, bt_weights, bt_log = backtest_strategy()

        equal_weight_ret = returns[close.columns].mean(axis=1).fillna(0)
        equal_weight_nav = (1 + equal_weight_ret).cumprod()

        bench_code = "510330" if "510330" in close.columns else close.columns[0]
        bench_nav = close[bench_code].dropna() / close[bench_code].dropna().iloc[0]

        bt_summary = pd.DataFrame({
            "strategy": strategy_nav,
            "huaxia_equal_weight": equal_weight_nav.reindex(strategy_nav.index).ffill(),
            f"benchmark_{bench_code}": bench_nav.reindex(strategy_nav.index).ffill(),
        }).dropna()

        plt.figure(figsize=(12, 5))
        bt_summary.plot(ax=plt.gca())
        plt.title("回测净值曲线")
        plt.ylabel("NAV")
        plt.show()

        display(bt_log.tail(20))
        """
    ),
    code(
        """
        def performance_table(ret_map):
            rows = []
            for name, ret in ret_map.items():
                ret = pd.Series(ret).dropna()
                nav = (1 + ret).cumprod()
                rows.append({
                    "name": name,
                    "ann_return": annualized_return(ret),
                    "ann_vol": annualized_vol(ret),
                    "sharpe": sharpe_ratio(ret, CONFIG["risk_free_rate"]),
                    "max_drawdown": max_drawdown(nav),
                    "calmar": annualized_return(ret) / abs(max_drawdown(nav)) if max_drawdown(nav) < 0 else np.nan,
                    "win_rate": (ret > 0).mean(),
                    "final_nav": nav.iloc[-1] if len(nav) else np.nan,
                })
            return pd.DataFrame(rows)

        bench_ret = bench_nav.pct_change().reindex(strategy_ret.index).fillna(0)
        perf = performance_table({
            "strategy": strategy_ret,
            "huaxia_equal_weight": equal_weight_ret.reindex(strategy_ret.index).fillna(0),
            f"benchmark_{bench_code}": bench_ret,
        })
        display(perf)

        rolling_dd = bt_summary / bt_summary.cummax() - 1
        plt.figure(figsize=(12, 4))
        rolling_dd.plot(ax=plt.gca())
        plt.title("回撤曲线")
        plt.ylabel("drawdown")
        plt.show()
        """
    ),
    md(
        """
        ## 10. 输出结果

        将当前基金池指标、目标组合、卖出信号、回测绩效和调仓日志导出，方便复盘。
        """
    ),
    code(
        """
        output_xlsx = DATA_DIR / "huaxia_fund_analysis_output.xlsx"
        with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
            huaxia_pool.to_excel(writer, sheet_name="huaxia_pool", index=False)
            metrics.to_excel(writer, sheet_name="metrics", index=False)
            portfolio.to_excel(writer, sheet_name="target_portfolio", index=False)
            current_sell_table.to_excel(writer, sheet_name="sell_signals", index=False)
            perf.to_excel(writer, sheet_name="backtest_perf", index=False)
            bt_log.to_excel(writer, sheet_name="rebalance_log", index=False)

        print("已导出:", output_xlsx.resolve())
        """
    ),
    md(
        """
        ## 11. 下一步可增强方向

        - 加入开放式基金净值、规模、费率、基金经理任期、机构持有人比例等数据，扩大到完整华夏公募基金池。
        - 组合层面增加目标波动率、最大回撤约束、行业暴露约束、换手率约束。
        - 卖出规则可以拆为“清仓”和“减仓”，并用真实持仓成本、交易滑点、申赎/交易限制做精细化模拟。
        - 回测可以加入 Walk-forward 参数稳定性检查，避免对某段历史过拟合。
        """
    ),
]

nb["cells"] = cells
nbf.write(nb, OUT)
print(OUT)

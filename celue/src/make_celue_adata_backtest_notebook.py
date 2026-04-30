import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "celue_adata_backtest.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb["cells"] = [
    md(
        """
        # 基于 adata 重做 `celue.py` 小市值策略并回测

        原 `celue.py` 是聚宽事件驱动小市值策略，核心逻辑如下：

        - 股票池：`399101.XSHE` 中小综指成分股。
        - 过滤：新股、ST、科创板/创业板/北交所、停牌、涨跌停、高价股。
        - 选股：先按流通市值升序取 200，再按总市值升序取 100，再按行业去重，目标持仓 10 只。
        - 调仓：每周二调仓，等权买入。
        - 风控：1 月和 4 月空仓；个股跌破成本 0.91 止损；市场大跌时清仓。

        `adata` 与聚宽差异：

        - `adata` 能拿当前指数成分、股票日线、股本变化、东方财富行业/板块。
        - `adata` 没有聚宽的点时指数成分、历史 ST、历史停牌状态、完整估值快照和分钟交易撮合。
        - 因此本 notebook 是“日频研究版复刻”，会有当前成分股带来的幸存者偏差，结果适合研究框架，不应直接当作实盘收益预期。
        """
    ),
    md("## 1. 环境与参数"),
    code(
        """
        # %pip install adata pandas numpy matplotlib seaborn openpyxl nbformat nbclient ipykernel -q

        import json
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
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.width", 180)

        DATA_DIR = Path("data_celue_adata")
        PRICE_DIR = DATA_DIR / "prices"
        SHARE_DIR = DATA_DIR / "shares"
        PLATE_DIR = DATA_DIR / "plates"
        for p in [DATA_DIR, PRICE_DIR, SHARE_DIR, PLATE_DIR]:
            p.mkdir(exist_ok=True)

        CONFIG = {
            "index_code": "399101",          # 原策略基准/股票池：中小综指
            "start_date": "2025-01-01",     # 可改为 2020-01-01，但首次下载会更慢
            "end_date": "",                 # 留空表示到最新
            "initial_cash": 1_000_000,
            "stock_num": 10,
            "prefilter_n": 60,               # 为控制 adata 下载量，先用当前流通市值预筛；调大更接近原策略
            "candidate_by_float_cap": 200,
            "candidate_by_total_cap": 100,
            "max_price": 100,
            "new_stock_days": 375,
            "rebalance_weekday": 1,          # Monday=0, Tuesday=1
            "empty_months": {1, 4},          # 原策略 1 月、4 月空仓
            "fee_rate": 0.0003,              # 佣金+滑点近似；印花税简化进卖出换手
            "sell_fee_extra": 0.001,
            "stoploss_limit": 0.91,
            "market_stoploss": 0.93,
            "use_industry_dedup": True,
        }

        print("adata version:", getattr(adata, "__version__", "unknown"))
        """
    ),
    md("## 2. 数据层：用 adata 获取股票池、行情、股本与行业"),
    code(
        """
        def cache_csv(path, loader, dtype=None, parse_dates=None, force=False):
            path = Path(path)
            if path.exists() and not force:
                return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
            df = loader()
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return df


        def load_all_stocks():
            def _loader():
                df = adata.stock.info.all_code()
                df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
                return df
            return cache_csv(DATA_DIR / "all_stocks.csv", _loader, dtype={"stock_code": str}, parse_dates=["list_date"])


        def load_index_constituent(index_code):
            def _loader():
                df = adata.stock.info.index_constituent(index_code)
                df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
                return df
            return cache_csv(DATA_DIR / f"index_{index_code}_constituent.csv", _loader, dtype={"stock_code": str})


        def get_market_cached(code, start_date, end_date=""):
            code = str(code).zfill(6)
            suffix = end_date or "latest"
            path = PRICE_DIR / f"{code}_{start_date}_{suffix}.csv"
            if path.exists():
                return pd.read_csv(path, dtype={"stock_code": str}, parse_dates=["trade_date"])
            last_err = "empty response"
            for i in range(3):
                try:
                    df = adata.stock.market.get_market(code, start_date=start_date, end_date=end_date or None, k_type=1, adjust_type=1)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
                        df["trade_date"] = pd.to_datetime(df["trade_date"])
                        for col in ["open", "close", "high", "low", "volume", "amount", "change_pct", "change", "turnover_ratio", "pre_close"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        df = df.sort_values("trade_date").drop_duplicates(["stock_code", "trade_date"])
                        df.to_csv(path, index=False, encoding="utf-8-sig")
                        return df
                except Exception as exc:
                    last_err = exc
                    time.sleep(1 + i)
            print(f"行情下载失败 {code}: {last_err}")
            return pd.DataFrame()


        def get_shares_cached(code):
            code = str(code).zfill(6)
            path = SHARE_DIR / f"{code}_shares.csv"
            if path.exists():
                return pd.read_csv(path, dtype={"stock_code": str}, parse_dates=["change_date"])
            try:
                df = adata.stock.info.get_stock_shares(code, is_history=True)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
                    df["change_date"] = pd.to_datetime(df["change_date"])
                    for col in ["total_shares", "limit_shares", "list_a_shares"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.sort_values("change_date")
                    df.to_csv(path, index=False, encoding="utf-8-sig")
                    return df
            except Exception as exc:
                print(f"股本下载失败 {code}: {exc}")
            return pd.DataFrame(columns=["stock_code", "change_date", "total_shares", "limit_shares", "list_a_shares", "change_reason"])


        def get_industry_cached(code):
            code = str(code).zfill(6)
            path = PLATE_DIR / f"{code}_plate.csv"
            if path.exists():
                df = pd.read_csv(path, dtype={"stock_code": str})
            else:
                try:
                    df = adata.stock.info.get_plate_east(code, plate_type=1)
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame()
                except Exception:
                    df = pd.DataFrame()
                if df.empty:
                    df = pd.DataFrame([{"stock_code": code, "plate_name": "未知行业", "plate_type": "行业"}])
                df.to_csv(path, index=False, encoding="utf-8-sig")
            industry_rows = df[df.get("plate_type", "") == "行业"] if "plate_type" in df.columns else df
            if industry_rows.empty:
                return "未知行业"
            return str(industry_rows.iloc[0].get("plate_name", "未知行业"))
        """
    ),
    md("## 3. 构建股票池：复刻原策略过滤逻辑"),
    code(
        """
        all_stocks = load_all_stocks()
        index_df = load_index_constituent(CONFIG["index_code"])

        universe = (
            index_df.merge(all_stocks[["stock_code", "list_date"]], on="stock_code", how="left")
            .rename(columns={"short_name": "name"})
            .drop_duplicates("stock_code")
        )
        universe["list_date"] = pd.to_datetime(universe["list_date"], errors="coerce")

        def base_filter(df):
            s = df["stock_code"].astype(str)
            name = df["name"].astype(str)
            keep = (
                ~s.str.startswith(("30", "68", "8", "4"))
                & ~name.str.contains("ST|退|\\*", regex=True, na=False)
            )
            return df[keep].copy()

        universe = base_filter(universe)
        print("原指数成分数:", len(index_df), "基础过滤后:", len(universe))
        display(universe.head())
        """
    ),
    md(
        """
        ## 4. 预筛并下载行情

        原策略每次从完整中小综指中排序。为了让 notebook 首次运行可控，这里先用东方财富全市场快照一次性拿当前流通市值做 `prefilter_n` 预筛，再用 `adata` 下载这些股票的历史行情和股本做回测。想更接近原版，可以把 `prefilter_n` 调大到 400、800 或全量。
        """
    ),
    code(
        """
        def fetch_current_a_snapshot(force=False):
            path = DATA_DIR / "current_a_snapshot.csv"
            if path.exists() and not force:
                return pd.read_csv(path, dtype={"stock_code": str})

            rows = []
            url = "https://82.push2delay.eastmoney.com/api/qt/clist/get"
            params_base = {
                "pn": 1,
                "pz": 100,
                "po": 1,
                "np": 1,
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": 2,
                "invt": 2,
                "fid": "f3",
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
                "fields": "f12,f14,f2,f20,f21,f23",
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            for page in range(1, 80):
                params = {**params_base, "pn": page}
                last_err = None
                for attempt in range(3):
                    try:
                        r = requests.get(url, params=params, headers=headers, timeout=12)
                        r.raise_for_status()
                        data = r.json().get("data")
                        break
                    except Exception as exc:
                        last_err = exc
                        time.sleep(1 + attempt)
                else:
                    raise RuntimeError(f"全市场快照失败 page={page}: {last_err}")
                if not data or not data.get("diff"):
                    break
                for item in data["diff"]:
                    rows.append({
                        "stock_code": str(item.get("f12", "")).zfill(6),
                        "name_snapshot": item.get("f14"),
                        "latest_close": item.get("f2"),
                        "latest_total_cap": item.get("f20"),
                        "latest_float_cap": item.get("f21"),
                        "pb": item.get("f23"),
                    })
            df = pd.DataFrame(rows)
            for col in ["latest_close", "latest_total_cap", "latest_float_cap", "pb"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return df

        snapshot = fetch_current_a_snapshot(force=False)
        prefilter = (
            universe.merge(snapshot, on="stock_code", how="left")
            .dropna(subset=["latest_float_cap"])
            .sort_values("latest_float_cap")
            .head(CONFIG["prefilter_n"])
            .reset_index(drop=True)
        )
        display(prefilter.head(20))
        print("进入历史回测下载的股票数:", len(prefilter))
        """
    ),
    code(
        """
        price_frames = []
        for i, code_ in enumerate(prefilter["stock_code"], start=1):
            df = get_market_cached(code_, CONFIG["start_date"], CONFIG["end_date"])
            if not df.empty:
                price_frames.append(df)
            if i % 30 == 0:
                print("行情处理:", i, "/", len(prefilter))

        prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
        prices = prices.merge(prefilter[["stock_code", "name", "list_date"]], on="stock_code", how="left")
        print("行情记录:", len(prices), "股票数:", prices["stock_code"].nunique())
        display(prices.tail())

        close = prices.pivot(index="trade_date", columns="stock_code", values="close").sort_index()
        open_ = prices.pivot(index="trade_date", columns="stock_code", values="open").sort_index()
        change_pct = prices.pivot(index="trade_date", columns="stock_code", values="change_pct").sort_index()
        turnover_ratio = prices.pivot(index="trade_date", columns="stock_code", values="turnover_ratio").sort_index()
        ret = close.pct_change().fillna(0)
        trade_dates = close.index
        """
    ),
    md("## 5. 股本、行业与市值快照"),
    code(
        """
        share_history = {code_: get_shares_cached(code_) for code_ in prefilter["stock_code"]}

        def shares_asof(code, asof, field):
            df = share_history.get(code)
            if df is None or df.empty:
                return np.nan
            d = df[df["change_date"] <= pd.Timestamp(asof)]
            if d.empty:
                d = df.head(1)
            val = pd.to_numeric(d[field], errors="coerce").dropna()
            return val.iloc[-1] if len(val) else np.nan

        industry_map = {}
        if CONFIG["use_industry_dedup"]:
            for i, code_ in enumerate(prefilter["stock_code"], start=1):
                industry_map[code_] = get_industry_cached(code_)
                if i % 40 == 0:
                    print("行业处理:", i, "/", len(prefilter))
        else:
            industry_map = {code_: "未启用行业去重" for code_ in prefilter["stock_code"]}

        meta = prefilter.set_index("stock_code")[["name", "list_date"]].copy()
        meta["industry"] = pd.Series(industry_map)
        display(meta.head(20))
        """
    ),
    md("## 6. 选股函数：按原策略重写"),
    code(
        """
        def previous_trade_date(dt):
            idx = trade_dates[trade_dates < pd.Timestamp(dt)]
            return idx[-1] if len(idx) else None

        def is_rebalance_day(dt):
            return pd.Timestamp(dt).weekday() == CONFIG["rebalance_weekday"]

        def is_empty_month(dt):
            return pd.Timestamp(dt).month in CONFIG["empty_months"]

        def select_stocks(asof):
            asof = pd.Timestamp(asof)
            px = close.loc[asof].dropna()
            rows = []
            for code_, price in px.items():
                if code_ not in meta.index:
                    continue
                list_date = meta.loc[code_, "list_date"]
                if pd.notna(list_date) and (asof - list_date).days < CONFIG["new_stock_days"]:
                    continue
                if price > CONFIG["max_price"]:
                    continue
                # 近似涨跌停过滤：非持仓买入时避开前一交易日接近涨跌停的股票。
                cp = change_pct.loc[asof, code_] if code_ in change_pct.columns else np.nan
                if pd.notna(cp) and (cp >= 9.5 or cp <= -9.5):
                    continue
                float_shares = shares_asof(code_, asof, "list_a_shares")
                total_shares = shares_asof(code_, asof, "total_shares")
                if pd.isna(float_shares) or pd.isna(total_shares):
                    continue
                rows.append({
                    "stock_code": code_,
                    "name": meta.loc[code_, "name"],
                    "industry": meta.loc[code_, "industry"],
                    "close": price,
                    "float_cap": price * float_shares,
                    "total_cap": price * total_shares,
                    "turnover_ratio": turnover_ratio.loc[asof, code_] if code_ in turnover_ratio.columns else np.nan,
                })
            df = pd.DataFrame(rows)
            if df.empty:
                return []
            df = df.sort_values("float_cap").head(CONFIG["candidate_by_float_cap"])
            df = df.sort_values("total_cap").head(CONFIG["candidate_by_total_cap"])
            selected = []
            used_industries = set()
            for _, row in df.iterrows():
                ind = row["industry"]
                if CONFIG["use_industry_dedup"] and ind in used_industries:
                    continue
                selected.append(row["stock_code"])
                used_industries.add(ind)
                if len(selected) >= CONFIG["stock_num"] * 2:
                    break
            return selected

        sample_asof = trade_dates[min(120, len(trade_dates)-1)]
        sample_selected = select_stocks(sample_asof)
        pd.DataFrame({"stock_code": sample_selected}).join(meta, on="stock_code")
        """
    ),
    md("## 7. 回测引擎：日频复刻调仓、空仓月和止损"),
    code(
        """
        def run_backtest():
            cash = CONFIG["initial_cash"]
            positions = {}  # code -> shares
            entry_price = {}
            logs = []
            nav_rows = []

            for i, dt in enumerate(trade_dates):
                dt = pd.Timestamp(dt)
                # 用当日收盘价估值。
                pos_value = 0.0
                for code_, shares in list(positions.items()):
                    price = close.loc[dt, code_] if code_ in close.columns else np.nan
                    if pd.notna(price):
                        pos_value += shares * price
                total_value = cash + pos_value

                # 市场大跌止损：复刻原策略 close/open 均值 <= 0.93。
                ratio = (close.loc[dt] / open_.loc[dt]).replace([np.inf, -np.inf], np.nan)
                market_ratio = ratio.dropna().mean()
                if positions and pd.notna(market_ratio) and market_ratio <= CONFIG["market_stoploss"]:
                    sell_value = 0
                    for code_, shares in list(positions.items()):
                        price = close.loc[dt, code_]
                        sell_value += shares * price
                    cost = sell_value * (CONFIG["fee_rate"] + CONFIG["sell_fee_extra"])
                    cash += sell_value - cost
                    logs.append({"date": dt, "action": "market_stoploss", "codes": ",".join(positions.keys()), "value": sell_value, "cost": cost})
                    positions.clear()
                    entry_price.clear()

                # 个股止损：收盘价 < 成本 * 0.91。
                for code_, shares in list(positions.items()):
                    price = close.loc[dt, code_]
                    if pd.notna(price) and price < entry_price.get(code_, price) * CONFIG["stoploss_limit"]:
                        sell_value = shares * price
                        cost = sell_value * (CONFIG["fee_rate"] + CONFIG["sell_fee_extra"])
                        cash += sell_value - cost
                        del positions[code_]
                        entry_price.pop(code_, None)
                        logs.append({"date": dt, "action": "stock_stoploss", "codes": code_, "value": sell_value, "cost": cost})

                # 周二调仓；选择信号只用前一交易日数据，避免看未来。
                if is_rebalance_day(dt):
                    signal_dt = previous_trade_date(dt)
                    if signal_dt is not None:
                        if is_empty_month(dt):
                            targets = []
                            reason = "empty_month"
                        else:
                            targets = select_stocks(signal_dt)[:CONFIG["stock_num"]]
                            reason = "weekly_rebalance"

                        # 卖出非目标。
                        for code_, shares in list(positions.items()):
                            if code_ not in targets:
                                price = close.loc[dt, code_]
                                if pd.notna(price):
                                    sell_value = shares * price
                                    cost = sell_value * (CONFIG["fee_rate"] + CONFIG["sell_fee_extra"])
                                    cash += sell_value - cost
                                    logs.append({"date": dt, "action": "sell_" + reason, "codes": code_, "value": sell_value, "cost": cost})
                                del positions[code_]
                                entry_price.pop(code_, None)

                        # 重新估值，等权买入缺口。
                        current_value = cash + sum(
                            shares * close.loc[dt, code_]
                            for code_, shares in positions.items()
                            if code_ in close.columns and pd.notna(close.loc[dt, code_])
                        )
                        if targets:
                            target_value_each = current_value / CONFIG["stock_num"]
                            for code_ in targets:
                                price = close.loc[dt, code_] if code_ in close.columns else np.nan
                                if pd.isna(price) or price <= 0:
                                    continue
                                current_shares = positions.get(code_, 0.0)
                                current_position_value = current_shares * price
                                diff_value = target_value_each - current_position_value
                                if diff_value > current_value * 0.001:
                                    buy_cash = min(cash, diff_value)
                                    cost = buy_cash * CONFIG["fee_rate"]
                                    shares_to_buy = (buy_cash - cost) / price
                                    if shares_to_buy > 0:
                                        cash -= buy_cash
                                        positions[code_] = current_shares + shares_to_buy
                                        entry_price.setdefault(code_, price)
                                        logs.append({"date": dt, "action": "buy_" + reason, "codes": code_, "value": buy_cash, "cost": cost})

                pos_value = sum(
                    shares * close.loc[dt, code_]
                    for code_, shares in positions.items()
                    if code_ in close.columns and pd.notna(close.loc[dt, code_])
                )
                total_value = cash + pos_value
                nav_rows.append({
                    "date": dt,
                    "cash": cash,
                    "position_value": pos_value,
                    "total_value": total_value,
                    "holding_count": len(positions),
                    "holdings": ",".join(positions.keys()),
                })

            nav = pd.DataFrame(nav_rows).set_index("date")
            log = pd.DataFrame(logs)
            nav["nav"] = nav["total_value"] / CONFIG["initial_cash"]
            nav["ret"] = nav["nav"].pct_change().fillna(0)
            return nav, log

        nav, trade_log = run_backtest()
        display(nav.tail())
        display(trade_log.tail(20))
        """
    ),
    md("## 8. 绩效分析"),
    code(
        """
        def max_drawdown(nav_series):
            dd = nav_series / nav_series.cummax() - 1
            return dd.min()

        def ann_return(ret_series):
            ret_series = pd.Series(ret_series).dropna()
            if len(ret_series) == 0:
                return np.nan
            return (1 + ret_series).prod() ** (252 / len(ret_series)) - 1

        def ann_vol(ret_series):
            return pd.Series(ret_series).dropna().std() * np.sqrt(252)

        def sharpe(ret_series, rf=0.02):
            ar = ann_return(ret_series)
            av = ann_vol(ret_series)
            return (ar - rf) / av if av and not np.isnan(av) else np.nan

        # 基准：399101 指数；若接口失败，则用回测股票池等权作替代。
        try:
            idx = adata.stock.market.get_market_index(CONFIG["index_code"], start_date=CONFIG["start_date"], k_type=1)
            idx["trade_date"] = pd.to_datetime(idx["trade_date"])
            idx = idx.set_index("trade_date").sort_index()
            idx_close_col = "close" if "close" in idx.columns else "price"
            benchmark_nav = idx[idx_close_col].reindex(nav.index).ffill()
            benchmark_nav = benchmark_nav / benchmark_nav.dropna().iloc[0]
        except Exception as exc:
            print("指数行情失败，使用股票池等权替代:", exc)
            ew_ret = ret.reindex(nav.index).mean(axis=1).fillna(0)
            benchmark_nav = (1 + ew_ret).cumprod()

        result = pd.DataFrame({"strategy": nav["nav"], "benchmark": benchmark_nav}).dropna()
        perf = pd.DataFrame([
            {
                "name": col,
                "ann_return": ann_return(result[col].pct_change().fillna(0)),
                "ann_vol": ann_vol(result[col].pct_change().fillna(0)),
                "sharpe": sharpe(result[col].pct_change().fillna(0)),
                "max_drawdown": max_drawdown(result[col]),
                "final_nav": result[col].iloc[-1],
            }
            for col in result.columns
        ])
        display(perf)

        plt.figure(figsize=(12, 5))
        result.plot(ax=plt.gca())
        plt.title("celue adata 日频回测净值")
        plt.ylabel("NAV")
        plt.show()

        plt.figure(figsize=(12, 4))
        (result / result.cummax() - 1).plot(ax=plt.gca())
        plt.title("回撤")
        plt.ylabel("drawdown")
        plt.show()
        """
    ),
    md("## 9. 持仓与交易复盘"),
    code(
        """
        latest_holdings = []
        if len(nav):
            last_codes = [x for x in nav.iloc[-1]["holdings"].split(",") if x]
            for code_ in last_codes:
                latest_holdings.append({
                    "stock_code": code_,
                    "name": meta.loc[code_, "name"] if code_ in meta.index else "",
                    "industry": meta.loc[code_, "industry"] if code_ in meta.index else "",
                    "last_close": close[code_].dropna().iloc[-1] if code_ in close.columns else np.nan,
                })
        latest_holdings = pd.DataFrame(latest_holdings)
        display(latest_holdings)

        action_count = trade_log["action"].value_counts().to_frame("count") if not trade_log.empty else pd.DataFrame()
        display(action_count)
        """
    ),
    md("## 10. 导出结果"),
    code(
        """
        out_xlsx = DATA_DIR / "celue_adata_backtest_output.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            universe.to_excel(writer, sheet_name="universe", index=False)
            prefilter.to_excel(writer, sheet_name="prefilter", index=False)
            meta.reset_index().to_excel(writer, sheet_name="meta", index=False)
            nav.reset_index().to_excel(writer, sheet_name="nav", index=False)
            trade_log.to_excel(writer, sheet_name="trade_log", index=False)
            perf.to_excel(writer, sheet_name="performance", index=False)
            latest_holdings.to_excel(writer, sheet_name="latest_holdings", index=False)

        print("notebook 输出文件:", out_xlsx.resolve())
        """
    ),
]

nbf.write(nb, OUT)
print(OUT)

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from forest4.backtest.engine import run_backtest
from forest4.backtest.grid import run_grid
from forest4.backtest.risk import RiskManager
from forest4.config import BacktestSettings
from forest4.examples.synthetic import generate_ohlc

st.set_page_config(page_title="FOREST 4.0 Dashboard", layout="wide")


# --------------------
# Pomocnicze
# --------------------
def _rename_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica nazwy kolumn na: open, high, low, close."""
    mapping = {c.lower(): c for c in df.columns}
    # dopuszczalne warianty
    alias = {
        "o": "open",
        "op": "open",
        "open": "open",
        "h": "high",
        "hi": "high",
        "high": "high",
        "l": "low",
        "lo": "low",
        "low": "low",
        "c": "close",
        "cl": "close",
        "close": "close",
    }
    out = {}
    for k, v in alias.items():
        if k in mapping:
            out[mapping[k]] = v
    df = df.rename(columns=out)
    return df


def load_csv(file) -> pd.DataFrame:
    """Wczytuje CSV i ustawia DatetimeIndex (kolumna time/date/datetime lub 1. kolumna)."""
    raw = file.read()
    df = pd.read_csv(io.BytesIO(raw))
    df = _rename_ohlc(df)

    # Kolumna czasu
    time_candidates = [c for c in ["time", "date", "datetime"] if c in df.columns]
    if time_candidates:
        tcol = time_candidates[0]
        df[tcol] = pd.to_datetime(df[tcol])
        df = df.set_index(tcol)
    else:
        # próba z pierwszą kolumną
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
        except Exception:
            pass

    req = {"open", "high", "low", "close"}
    if not req.issubset(set(df.columns)):
        missing = ", ".join(sorted(req - set(df.columns)))
        raise ValueError(f"Brak wymaganych kolumn: {missing}")

    df = df[["open", "high", "low", "close"]].copy()
    df.index = pd.DatetimeIndex(df.index, name="time")
    return df.astype(float)


def dd_series(equity: pd.Series) -> pd.Series:
    """Drawdown jako seria [0..N], wartości ujemne."""
    equity = equity.astype(float)
    peak = equity.cummax()
    return equity / peak - 1.0


def equity_metrics(equity: pd.Series) -> dict[str, float]:
    """CAGR, max_dd (rel), Sharpe (z dziennych zwrotów), RAR."""
    eq = equity.dropna().astype(float)
    if eq.empty:
        return {"equity_end": 0.0, "max_dd": 0.0, "cagr": 0.0, "sharpe": 0.0, "rar": 0.0}

    eq0, eqN = float(eq.iloc[0]), float(eq.iloc[-1])
    equity_end = eqN
    # CAGR ~ w skali lat; jeśli mamy DatetimeIndex to policzymy dni
    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
        days = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr = (eqN / max(eq0, 1e-9)) ** (365.25 / days) - 1.0
    else:
        # fallback (brak czasu) — „roczne” traktujmy umownie
        cagr = (eqN / max(eq0, 1e-9)) - 1.0

    dd = dd_series(eq)
    max_dd = float(dd.min())  # ujemny

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0:
        # przyjmijmy ~252 sesje rocznie
        sharpe = (rets.mean() / rets.std()) * (252.0**0.5)
    else:
        sharpe = 0.0

    rar = (cagr / abs(max_dd)) if max_dd < 0 else float("inf")
    return {
        "equity_end": equity_end,
        "max_dd": max_dd,
        "cagr": cagr,
        "sharpe": sharpe,
        "rar": rar,
    }


def heatmap(grid_df: pd.DataFrame, metric: str, dd_lim: float | None = None) -> Any:
    """Wykres heatmapy (fast vs slow) dla metryk z grida."""
    df = grid_df.copy()
    if dd_lim is not None:
        df = df[df["max_dd"] >= -abs(dd_lim)]
    # oczekujemy kolumn: fast, slow, metric
    pivot = df.pivot(index="fast", columns="slow", values=metric)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", origin="lower")
    fig.update_layout(title=f"Heat-map: {metric}", xaxis_title="slow", yaxis_title="fast")
    return fig


# --------------------
# UI — zakładki
# --------------------
st.title("FOREST 4.0 – Backtest & Grid")


tab_bt, tab_grid, tab_heat = st.tabs(["Back-test", "Grid Runner", "Grid Heat-map"])

with tab_bt:
    st.subheader("Pojedynczy backtest (EMA cross)")

    left, right = st.columns([2, 1], gap="large")
    with left:
        up = st.file_uploader("CSV z kolumnami: time(index), open, high, low, close", type=["csv"])

    with right:
        st.markdown("**Nie masz danych?** Wygeneruj demo:")
        periods = st.slider("Okresy (rows)", min_value=50, max_value=5000, value=500, step=50)
        freq = st.selectbox("Częstotliwość", ["D", "H", "15min", "30min"])
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        start_price = st.number_input("Cena startowa", min_value=0.01, value=100.0, step=1.0)
        gen_btn = st.button("Generate demo OHLC")

    # wybór danych
    df: pd.DataFrame | None = None
    src = ""
    if gen_btn:
        df = generate_ohlc(periods=periods, freq=freq, seed=int(seed), start_price=float(start_price))
        src = f"demo ({len(df):,} wierszy)"
    elif up is not None:
        df = load_csv(up)
        src = f"CSV ({len(df):,} wierszy)"

    st.write(f"Źródło danych: **{src or '—'}**")

    fast = st.slider("EMA fast", min_value=2, max_value=100, value=12, step=1)
    slow = st.slider("EMA slow", min_value=fast + 1, max_value=250, value=26, step=1)

    init_cap = st.number_input("Kapitał początkowy", min_value=1000.0, value=100_000.0, step=1000.0)

    with st.expander("Ryzyko i koszty (opcjonalnie)", expanded=False):
        risk_per = st.slider("Risk per trade", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
        atr_period = st.number_input("ATR period", min_value=2, value=14, step=1)
        atr_mult = st.number_input("ATR multiple", min_value=0.5, value=2.0, step=0.5)
        fee_perc = st.number_input("Prowizja (ułamek)", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")
        slippage_perc = st.number_input("Poślizg (ułamek)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

    run_btn = st.button("Run back-test", type="primary", disabled=df is None)

    if run_btn and df is not None:
        # ustawienia strategii
        settings = BacktestSettings()
        # spróbuj nadpisać fast/slow, jeśli struktura na to pozwala
        try:
            settings.strategy.params.update({"fast": int(fast), "slow": int(slow)})
        except Exception:
            pass

        rm = RiskManager(
            initial_capital=float(init_cap),
            risk_per_trade=float(risk_per),
            fee_perc=float(fee_perc),
            slippage_perc=float(slippage_perc),
        )

        with st.spinner("Uruchamiam backtest..."):
            res = run_backtest(
                df,
                settings,
                risk=rm,
                atr_period=int(atr_period),
                atr_multiple=float(atr_mult),
            )
            eq = res.equity_curve

        st.success("Done ✅")
        cols = st.columns([2, 1])
        with cols[0]:
            st.plotly_chart(px.line(eq, title="Equity"), use_container_width=True)
        with cols[1]:
            st.area_chart(dd_series(eq), use_container_width=True)
            m = equity_metrics(eq)
            st.write(
                pd.DataFrame(
                    {
                        "equity_end": [m["equity_end"]],
                        "max_dd": [m["max_dd"]],
                        "cagr": [m["cagr"]],
                        "sharpe": [m["sharpe"]],
                        "rar": [m["rar"]],
                    }
                )
            )

        csv = eq.to_csv(index=True).encode("utf-8")
        st.download_button("Download equity.csv", data=csv, file_name="equity.csv", mime="text/csv")


with tab_grid:
    st.subheader("Grid Runner")
    st.write("Uruchom siatkę parametrów (fast/slow) i zapisz wyniki do pliku.")
    up2 = st.file_uploader("CSV (OHLC) pod grid-search", type=["csv"], key="grid_csv")
    if up2 is not None:
        df2 = load_csv(up2)

        c1, c2, c3 = st.columns(3)
        with c1:
            fmin = st.number_input("fast min", value=8, step=1)
            fmax = st.number_input("fast max", value=20, step=1)
            fstep = st.number_input("fast step", value=2, step=1)
        with c2:
            smin = st.number_input("slow min", value=22, step=1)
            smax = st.number_input("slow max", value=60, step=1)
            sstep = st.number_input("slow step", value=2, step=1)
        with c3:
            n_jobs = st.number_input("CPU jobs (<=0 -> auto)", value=0, step=1)

        export_name = st.text_input("Nazwa pliku wyników (parquet/csv)", value="grid.parquet")
        run_g = st.button("Run grid")
        if run_g:
            with st.spinner("Grid running..."):
                out = run_grid(
                    df2,
                    fast_range=range(int(fmin), int(fmax) + 1, int(fstep)),
                    slow_range=range(int(smin), int(smax) + 1, int(sstep)),
                    n_jobs=int(n_jobs),
                    export_path=export_name,
                )
            st.success(f"Zapisano {len(out)} wierszy do {export_name}")


with tab_heat:
    st.subheader("Grid Heat-map")
    up3 = st.file_uploader("Wyniki grida (parquet/csv)", type=["parquet", "csv"], key="heat_csv")
    metric = st.selectbox("Metryka", ["equity_end", "max_dd", "rar", "sharpe"])
    dd_lim = st.number_input("Filtr: minimalny (relatywny) max DD (np. -0.35)", value=-1.0, step=0.05)
    if up3 is not None:
        name = up3.name.lower()
        if name.endswith(".parquet"):
            gdf = pd.read_parquet(up3)
        else:
            gdf = pd.read_csv(up3)
        fig = heatmap(gdf, metric=metric, dd_lim=dd_lim if dd_lim > -1.0 else None)
        st.plotly_chart(fig, use_container_width=True)


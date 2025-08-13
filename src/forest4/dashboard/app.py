from __future__ import annotations

import inspect
import io
import math
from pathlib import Path
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

# ======================================================================
# Pomocnicze: walidacja, wczytywanie, metryki, wykrywanie luk, benchmark
# ======================================================================


def _rename_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica nazwy kolumn na: open, high, low, close."""
    mapping = {c.lower(): c for c in df.columns}
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
    return df.rename(columns=out)


def _normalize_dtindex_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Wymusza tz-naive (usuwa strefę czasową) dla spójnego cięcia po dacie."""
    if getattr(idx, "tz", None) is not None:
        try:
            idx = idx.tz_convert("UTC").tz_localize(None)  # tz-aware -> UTC -> naive
        except Exception:
            try:
                idx = idx.tz_localize(None)  # fallback
            except Exception:
                pass
    # bezpiecznie ustaw nazwę
    idx = pd.DatetimeIndex(idx, name="time")
    return idx


def _prepare_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Standaryzuje OHLC + DatetimeIndex + walidacja spójności."""
    df = _rename_ohlc(df)

    # Kolumna czasu -> index
    time_candidates = [c for c in ["time", "date", "datetime"] if c in df.columns]
    if time_candidates:
        tcol = time_candidates[0]
        df[tcol] = pd.to_datetime(df[tcol], errors="raise", utc=False)
        df = df.set_index(tcol)
    else:
        # spróbuj z pierwszą kolumną
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="raise", utc=False)
            df = df.set_index(df.columns[0])
        except Exception as e:
            st.error("Nie udało się ustawić indeksu czasowego. Dodaj kolumnę 'time/date/datetime'.")
            raise e

    req = {"open", "high", "low", "close"}
    if not req.issubset(set(df.columns)):
        missing = ", ".join(sorted(req - set(df.columns)))
        raise ValueError(f"Brak wymaganych kolumn: {missing}")

    # tylko wymagane kolumny, typy i sortowanie
    df = df[["open", "high", "low", "close"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = _normalize_dtindex_naive(pd.DatetimeIndex(df.index))
    df = df.sort_index()
    # usuń duplikaty znaczników czasu
    df = df[~df.index.duplicated(keep="last")]

    # NaN – informacja i czyszczenie
    if df.isna().any().any():
        st.warning("Wykryto brakujące wartości w OHLC – wiersze z NaN zostaną usunięte.")
        df = df.dropna(subset=["open", "high", "low", "close"])

    # Walidacja spójności OHLC
    too_low = df["low"] > df[["open", "high", "close"]].max(axis=1)
    too_high = df["high"] < df[["open", "low", "close"]].min(axis=1)
    if bool(too_low.any() or too_high.any()):
        bad = int((too_low | too_high).sum())
        raise ValueError(
            f"Nieprawidłowe OHLC w {bad} wierszach (sprawdź relacje high/low vs open/close)."
        )

    return df.astype(float)


@st.cache_data(show_spinner=False)
def _load_table_cached(raw: bytes, fname: str) -> pd.DataFrame:
    """Cache'owane wczytywanie CSV/Parquet/Excel + przygotowanie OHLC."""
    ext = Path(fname).suffix.lower()
    bio = io.BytesIO(raw)

    if ext == ".csv":
        df = pd.read_csv(bio)
    elif ext == ".parquet":
        try:
            df = pd.read_parquet(bio)
        except Exception as e:
            st.error("Aby wczytać Parquet wymagany jest pakiet pyarrow lub fastparquet.")
            raise e
    elif ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(bio)
        except Exception as e:
            st.error("Aby wczytać Excel wymagany jest pakiet openpyxl lub xlrd.")
            raise e
    else:
        raise ValueError(f"Nieobsługiwane rozszerzenie pliku: {ext}")

    return _prepare_ohlc(df)


def load_uploaded(file) -> pd.DataFrame:
    """Wczytuje plik z uploader'a (csv/parquet/xlsx) i przygotowuje OHLC."""
    raw = file.read()
    name = getattr(file, "name", "upload.csv")
    return _load_table_cached(raw, name)


def dd_series(equity: pd.Series) -> pd.Series:
    """Drawdown jako seria [0..N], wartości ujemne."""
    equity = equity.astype(float)
    peak = equity.cummax()
    return equity / peak - 1.0


def _infer_periods_per_year(idx: pd.Index) -> int:
    """Heurystyka annualizacji dla Sharpe'a: D≈252; H≈24*252; M≈(24*60/m)*252."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return 252
    dt = (idx[1:] - idx[:-1]).median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        return 252
    # Doba lub więcej -> dzienne
    if dt >= pd.Timedelta(hours=23):
        return 252
    # Minutowe/godzinowe
    mins = max(1, int(round(dt / pd.Timedelta(minutes=1))))
    per_day = max(1, (24 * 60) // mins)
    return per_day * 252


def equity_metrics(equity: pd.Series) -> dict[str, float]:
    """CAGR, max_dd (rel), Sharpe (z annualizacją wg częstotliwości), RAR."""
    eq = equity.dropna().astype(float)
    if eq.empty:
        return {"equity_end": 0.0, "max_dd": 0.0, "cagr": 0.0, "sharpe": 0.0, "rar": 0.0}

    eq0, eqN = float(eq.iloc[0]), float(eq.iloc[-1])
    equity_end = eqN

    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
        days = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr = (eqN / max(eq0, 1e-9)) ** (365.25 / days) - 1.0
    else:
        cagr = (eqN / max(eq0, 1e-9)) - 1.0

    dd = dd_series(eq)
    max_dd = float(dd.min())  # ujemny

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0:
        per_year = _infer_periods_per_year(eq.index)
        sharpe = (rets.mean() / rets.std()) * math.sqrt(per_year)
    else:
        sharpe = 0.0

    rar = (cagr / abs(max_dd)) if max_dd < 0 else float("inf")
    return {"equity_end": equity_end, "max_dd": max_dd, "cagr": cagr, "sharpe": sharpe, "rar": rar}


def detect_gaps(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Wykryj luki czasowe (odstęp >> typowy interwał)."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return []
    dt = (idx[1:] - idx[:-1]).median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        return []
    thr = dt * 1.5
    gaps = [(p, c) for p, c in zip(idx[:-1], idx[1:]) if (c - p) > thr]
    return gaps


def buyhold_equity_aligned(
    eq: pd.Series, price: pd.Series, initial_capital: float
) -> pd.Series | None:
    """Benchmark Buy&Hold dopasowany do indeksu equity (również gdy equity ma indeks liczbowy)."""
    if price is None or price.empty or eq is None or eq.empty:
        return None
    price = price.dropna().astype(float)

    # Dopasowanie indeksów
    if isinstance(eq.index, pd.DatetimeIndex) and isinstance(price.index, pd.DatetimeIndex):
        price_aligned = price.reindex(eq.index, method="pad").dropna()
        # jeśli skróciło serię – dopasuj też equity
        if len(price_aligned) != len(eq):
            eq = eq.reindex(price_aligned.index)
        base = float(price_aligned.iloc[0])
        bh = initial_capital * (price_aligned / base)
        bh.name = "buy_hold"
        return bh

    # Gdy equity ma indeks liczbowy – użyj pierwszych N próbek ceny
    n = len(eq)
    if len(price) >= n:
        price_aligned = price.iloc[:n].reset_index(drop=True)
        base = float(price_aligned.iloc[0])
        # utwórz ten sam indeks co equity
        bh = pd.Series(
            initial_capital * (price_aligned / base).values, index=eq.index, name="buy_hold"
        )
        return bh

    return None


# ============================================================
# Param bridge – pewne przekazanie fast/slow do silnika
# ============================================================


def _apply_strategy_params(settings: BacktestSettings, fast: int, slow: int) -> None:
    """Ustawia parametry fast/slow w możliwych miejscach struktur konfiguracyjnych."""
    # 1) próba: settings.strategy.params (dict)
    try:
        strat = getattr(settings, "strategy", None)
        params = getattr(strat, "params", None)
        if isinstance(params, dict):
            params.update({"fast": int(fast), "slow": int(slow)})
    except Exception:
        pass

    # 2) alternatywne nazwy atrybutów
    alt_f = ["fast", "ema_fast", "short", "short_ema", "fast_len", "n_fast"]
    alt_s = ["slow", "ema_slow", "long", "long_ema", "slow_len", "n_slow"]
    for name in alt_f:
        if hasattr(strat, name):
            try:
                setattr(strat, name, int(fast))
            except Exception:
                pass
    for name in alt_s:
        if hasattr(strat, name):
            try:
                setattr(strat, name, int(slow))
            except Exception:
                pass

    # 3) ostatecznie – top-level BacktestSettings
    for name in alt_f:
        if hasattr(settings, name):
            try:
                setattr(settings, name, int(fast))
            except Exception:
                pass
    for name in alt_s:
        if hasattr(settings, name):
            try:
                setattr(settings, name, int(slow))
            except Exception:
                pass


def call_run_backtest(
    df: pd.DataFrame,
    settings: BacktestSettings,
    risk_mgr: RiskManager,
    atr_period: int,
    atr_multiple: float,
    fast: int | None = None,
    slow: int | None = None,
):
    """Bezpieczne wywołanie run_backtest z dopasowaniem nazw argumentów."""
    # Zapewnij, że ustawienia zawierają nasze fast/slow
    if fast is not None and slow is not None:
        _apply_strategy_params(settings, int(fast), int(slow))

    sig = inspect.signature(run_backtest)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    # ryzyko
    if "risk" in params:
        kwargs["risk"] = risk_mgr
    elif "risk_mgr" in params:
        kwargs["risk_mgr"] = risk_mgr

    # ATR
    if "atr_period" in params:
        kwargs["atr_period"] = int(atr_period)
    if "atr_multiple" in params:
        kwargs["atr_multiple"] = float(atr_multiple)

    # Jeśli silnik akceptuje fast/slow wprost – przekaż
    if fast is not None:
        for k in ("fast", "ema_fast", "short", "fast_len", "n_fast"):
            if k in params:
                kwargs[k] = int(fast)
                break
    if slow is not None:
        for k in ("slow", "ema_slow", "long", "slow_len", "n_slow"):
            if k in params:
                kwargs[k] = int(slow)
                break

    return run_backtest(df, settings, **kwargs)


def call_run_grid(
    df: pd.DataFrame,
    fmin: int,
    fmax: int,
    fstep: int,
    smin: int,
    smax: int,
    sstep: int,
    n_jobs: int | None,
    export_name: str | None,
    base_settings: BacktestSettings | None,
    risk_mgr: RiskManager | None,
    atr_period: int | None,
    atr_multiple: float | None,
    fee_perc: float | None,
    slippage_perc: float | None,
) -> pd.DataFrame:
    """Bezpieczne wywołanie run_grid z mapowaniem nazw argumentów i typów."""
    sig = inspect.signature(run_grid)
    params = sig.parameters

    fast_vals = list(range(int(fmin), int(fmax) + 1, int(fstep)))
    slow_vals = list(range(int(smin), int(smax) + 1, int(sstep)))

    kwargs: dict[str, Any] = {}
    if "fast_values" in params and "slow_values" in params:
        kwargs["fast_values"] = fast_vals
        kwargs["slow_values"] = slow_vals
    elif "fast_range" in params and "slow_range" in params:
        kwargs["fast_range"] = range(int(fmin), int(fmax) + 1, int(fstep))
        kwargs["slow_range"] = range(int(smin), int(smax) + 1, int(sstep))
    else:
        try:
            return run_grid(df, fast_vals, slow_vals)  # type: ignore[misc]
        except TypeError:
            pass

    if base_settings is not None and "base" in params:
        kwargs["base"] = base_settings
    if risk_mgr is not None and "risk" in params:
        kwargs["risk"] = risk_mgr
    if atr_period is not None and "atr_period" in params:
        kwargs["atr_period"] = int(atr_period)
    if atr_multiple is not None and "atr_multiple" in params:
        kwargs["atr_multiple"] = float(atr_multiple)
    if fee_perc is not None and "fee_perc" in params:
        kwargs["fee_perc"] = float(fee_perc)
    if slippage_perc is not None and "slippage_perc" in params:
        kwargs["slippage_perc"] = float(slippage_perc)

    if n_jobs is not None:
        if "n_jobs" in params:
            kwargs["n_jobs"] = int(n_jobs)
        elif "jobs" in params:
            kwargs["jobs"] = int(n_jobs)

    if export_name:
        if "export_path" in params:
            kwargs["export_path"] = export_name
        elif "export_name" in params:
            kwargs["export_name"] = export_name

    return run_grid(df, **kwargs)  # type: ignore[misc]


# ===========================
# UI — zakładki
# ===========================

st.title("FOREST 4.0 – Backtest & Grid (stable)")

# Przycisk resetu sesji
if st.button("🔁 Resetuj sesję"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

tab_bt, tab_grid = st.tabs(["Back-test", "Grid Runner"])

# Pamięć sesji
for key in ["bt_df", "bt_eq", "bt_metrics", "grid_df", "bt_df_filtered"]:
    st.session_state.setdefault(key, None)

# ===== Back-test =====
with tab_bt:
    st.subheader("Pojedynczy backtest (EMA cross)")

    left, right = st.columns([2, 1], gap="large")
    with left:
        up = st.file_uploader(
            "Dane OHLC (CSV/Parquet/Excel): time(index), open, high, low, close",
            type=["csv", "parquet", "xlsx", "xls"],
            key="bt_csv",
        )

    with right:
        st.markdown("**Nie masz danych?** Wygeneruj demo:")
        periods = st.slider("Okresy (rows)", min_value=50, max_value=5000, value=500, step=50)
        freq = st.selectbox("Częstotliwość", ["D", "H", "15min", "30min"])
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        start_price = st.number_input("Cena startowa", min_value=0.01, value=100.0, step=1.0)
        gen_btn = st.button("Generate demo OHLC", key="gen_demo_btn")

    df: pd.DataFrame | None = None
    src = ""
    if gen_btn:
        df = generate_ohlc(
            periods=periods, freq=freq, seed=int(seed), start_price=float(start_price)
        )
        src = f"demo ({len(df):,} wierszy)"
        st.session_state.bt_df = df
    elif up is not None:
        try:
            df = load_uploaded(up)
        except Exception as e:
            st.error(f"Błąd wczytywania: {e}")
            df = None
        else:
            src = f"{Path(getattr(up, 'name', 'upload.csv')).suffix.upper()} ({len(df):,} wierszy)"
            st.session_state.bt_df = df
    else:
        df = st.session_state.bt_df

    st.write(f"Źródło danych: **{src or ('—' if df is None else f'memory ({len(df):,})')}**")

    # Zakres dat + presety
    df_view = None
    if df is not None and not df.empty:
        # upewnij się, że indeks jest tz-naive
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = _normalize_dtindex_naive(df.index)

        min_d, max_d = df.index.min().date(), df.index.max().date()
        dflt = (min_d, max_d)
        # Streamlit bywa kapryśny: może zwrócić tuple zamiast dwóch wartości
        date_input_val = st.date_input("Zakres dat", value=dflt, min_value=min_d, max_value=max_d)
        if isinstance(date_input_val, tuple) and len(date_input_val) == 2:
            date_from, date_to = date_input_val
        else:
            # pojedyncza data – traktuj jako full range
            date_from, date_to = dflt

        if date_from > date_to:
            st.error("Data 'od' nie może być większa niż 'do'.")
        else:
            df_view = df.loc[pd.Timestamp(date_from) : pd.Timestamp(date_to)]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("⏳ Ostatni rok"):
                end = df.index.max()
                start = end - pd.DateOffset(years=1)
                df_view = df.loc[start:end]
        with c2:
            if st.button("📅 YTD"):
                year_start = pd.Timestamp(pd.Timestamp(df.index.max()).year, 1, 1)
                df_view = df.loc[year_start : df.index.max()]
        with c3:
            if st.button("🗓️ Ostatnie 90 dni"):
                end = df.index.max()
                start = end - pd.Timedelta(days=90)
                df_view = df.loc[start:end]
        with c4:
            show_gaps = st.checkbox("Pokaż luki (weekendy/święta)")

        # Wykrywanie luk
        if "show_gaps" in locals() and show_gaps and df_view is not None and len(df_view) > 2:
            gaps = detect_gaps(df_view.index)
            if gaps:
                st.info(f"Wykryto {len(gaps)} luk. Przykłady: {gaps[:3]}")
            else:
                st.success("Brak istotnych luk czasowych w wybranym zakresie.")

        st.session_state.bt_df_filtered = df_view

    fast = st.slider("EMA fast", min_value=2, max_value=200, value=12, step=1)
    slow = st.slider(
        "EMA slow", min_value=max(fast + 1, 3), max_value=400, value=max(26, fast + 1), step=1
    )
    init_cap = st.number_input("Kapitał początkowy", min_value=1000.0, value=100_000.0, step=1000.0)

    with st.expander("Ryzyko i koszty (opcjonalnie)", expanded=False):
        risk_per = st.slider(
            "Risk per trade", min_value=0.001, max_value=0.05, value=0.01, step=0.001
        )
        atr_period = st.number_input("ATR period", min_value=2, value=14, step=1)
        atr_mult = st.number_input("ATR multiple", min_value=0.5, value=2.0, step=0.5)
        fee_perc = st.number_input(
            "Prowizja (ułamek)", min_value=0.0, value=0.0005, step=0.0001, format="%.4f"
        )
        slippage_perc = st.number_input(
            "Poślizg (ułamek)", min_value=0.0, value=0.0, step=0.0001, format="%.4f"
        )

    show_bench = st.checkbox("Pokaż benchmark Buy&Hold na wykresie", value=True)
    run_btn = st.button(
        "Run back-test",
        type="primary",
        disabled=(st.session_state.bt_df_filtered is None),
        key="run_bt_btn",
    )

    if run_btn and st.session_state.bt_df_filtered is not None:
        data_for_bt = st.session_state.bt_df_filtered.copy()

        # Ustaw parametry strategii (bridge)
        settings = BacktestSettings()
        _apply_strategy_params(settings, int(fast), int(slow))

        rm = RiskManager(
            initial_capital=float(init_cap),
            risk_per_trade=float(risk_per),
            fee_perc=float(fee_perc),
            slippage_perc=float(slippage_perc),
        )

        with st.spinner("Uruchamiam backtest..."):
            res = call_run_backtest(
                df=data_for_bt,
                settings=settings,
                risk_mgr=rm,
                atr_period=int(atr_period),
                atr_multiple=float(atr_mult),
                fast=int(fast),
                slow=int(slow),
            )
            eq = res.equity_curve

        st.success("Done ✅")
        st.session_state.bt_eq = eq
        st.session_state.bt_metrics = equity_metrics(eq)

    # prezentacja wyników (jeśli są w pamięci)
    if st.session_state.bt_eq is not None and st.session_state.bt_df_filtered is not None:
        eq = st.session_state.bt_eq
        df_used = st.session_state.bt_df_filtered

        # Benchmark – bezpieczne dopasowanie do indeksu equity
        eq_bench = (
            buyhold_equity_aligned(eq, df_used.get("close"), float(init_cap))
            if show_bench
            else None
        )

        # Wykres equity (strategia + (opcjonalnie) bench)
        plot_df = pd.DataFrame({"strategy": eq})
        if eq_bench is not None:
            try:
                plot_df["buy_hold"] = eq_bench
            except Exception:
                pass
        fig = px.line(plot_df, title="Equity (strategia vs. benchmark)")
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown + metryki
        cols = st.columns([2, 1])
        with cols[0]:
            st.area_chart(dd_series(eq), use_container_width=True)
        with cols[1]:
            m = st.session_state.bt_metrics or equity_metrics(eq)
            st.write(pd.DataFrame([m]))

        # Pobieranie
        st.download_button(
            "Pobierz equity (strategia).csv",
            data=eq.to_csv(index=True).encode("utf-8"),
            file_name="equity_strategy.csv",
            mime="text/csv",
        )
        if show_bench and eq_bench is not None:
            both = plot_df.copy()
            st.download_button(
                "Pobierz equity (strategy_vs_bench).csv",
                data=both.to_csv(index=True).encode("utf-8"),
                file_name="equity_strategy_vs_bench.csv",
                mime="text/csv",
            )

# ===== Grid Runner =====
with tab_grid:
    st.subheader("Grid Runner")
    st.write("Uruchom siatkę parametrów (fast/slow) i zapisz wyniki do pliku.")
    up2 = st.file_uploader(
        "Dane OHLC (CSV/Parquet/Excel) pod grid-search",
        type=["csv", "parquet", "xlsx", "xls"],
        key="grid_csv",
    )

    if up2 is not None:
        try:
            df2 = load_uploaded(up2)
        except Exception as e:
            st.error(f"Błąd wczytywania: {e}")
            df2 = None

        if df2 is not None:
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

            with st.expander("Ryzyko i koszty (opcjonalnie)", expanded=False):
                risk_per_g = st.slider(
                    "Risk per trade (grid)", min_value=0.001, max_value=0.05, value=0.01, step=0.001
                )
                atr_period_g = st.number_input("ATR period (grid)", min_value=2, value=14, step=1)
                atr_mult_g = st.number_input(
                    "ATR multiple (grid)", min_value=0.5, value=2.0, step=0.5
                )
                fee_perc_g = st.number_input(
                    "Prowizja (ułamek, grid)",
                    min_value=0.0,
                    value=0.0005,
                    step=0.0001,
                    format="%.4f",
                )
                slippage_perc_g = st.number_input(
                    "Poślizg (ułamek, grid)", min_value=0.0, value=0.0, step=0.0001, format="%.4f"
                )

            export_name = st.text_input(
                "Nazwa pliku wyników (parquet/csv, opcjonalnie)", value="grid.parquet"
            )
            run_g = st.button("Run grid", key="run_grid_btn")

            if run_g:
                if fmin >= fmax or smin >= smax or fstep <= 0 or sstep <= 0:
                    st.error("Zakresy muszą mieć min < max oraz dodatni krok.")
                else:
                    base = BacktestSettings()
                    # dla przejrzystości — domyślne fast/slow są w run_grid
                    rm_g = RiskManager(
                        initial_capital=100_000.0,
                        risk_per_trade=float(risk_per_g),
                        fee_perc=float(fee_perc_g),
                        slippage_perc=float(slippage_perc_g),
                    )
                    with st.spinner("Grid running..."):
                        out_df = call_run_grid(
                            df=df2,
                            fmin=int(fmin),
                            fmax=int(fmax),
                            fstep=int(fstep),
                            smin=int(smin),
                            smax=int(smax),
                            sstep=int(sstep),
                            n_jobs=int(n_jobs),
                            export_name=(export_name or None),
                            base_settings=base,
                            risk_mgr=rm_g,
                            atr_period=int(atr_period_g),
                            atr_multiple=float(atr_mult_g),
                            fee_perc=float(fee_perc_g),
                            slippage_perc=float(slippage_perc_g),
                        )
                    st.success(f"Zakończono – {len(out_df):,} wierszy.")
                    st.session_state.grid_df = out_df

    # prezentacja wyników grida (jeśli są w pamięci)
    if st.session_state.grid_df is not None:
        gdf = st.session_state.grid_df
        st.dataframe(gdf)
        st.download_button(
            "Pobierz grid jako CSV",
            data=gdf.to_csv(index=False).encode("utf-8"),
            file_name="grid.csv",
            mime="text/csv",
            key="dl_grid_csv",
        )

        # (opcjonalnie) prosta heatmapa metryki
        with st.expander("Heatmap (beta)"):
            hm_on = st.checkbox("Pokaż heatmapę CAGR (fast vs slow)", value=False)
            if hm_on and {"fast", "slow", "cagr"}.issubset(set(gdf.columns)):
                try:
                    pivot = gdf.pivot(index="slow", columns="fast", values="cagr")
                    fig_hm = px.imshow(
                        pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        color_continuous_scale="RdYlGn",
                        labels={"color": "CAGR"},
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception as e:
                    st.warning(f"Nie udało się narysować heatmapy: {e}")

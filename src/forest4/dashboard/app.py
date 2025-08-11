from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

# 💡 najważniejsze: importy BEZWZGLĘDNE
from forest4.backtest.engine import run_backtest
from forest4.backtest.grid import run_grid
from forest4.config import BacktestSettings

st.set_page_config(page_title="FOREST 4.0 Dashboard", layout="wide")


def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    data = file.read()
    df = pd.read_csv(io.BytesIO(data))
    return df


def metrics(equity: pd.Series) -> pd.Series:
    roll = equity.cummax()
    dd = (equity - roll) / roll
    return dd


def heatmap(df: pd.DataFrame, metric: str, dd_lim: float | None = None):
    data = df.copy()
    if dd_lim is not None and "max_dd" in data.columns:
        data = data[data["max_dd"] <= dd_lim]
    pivot = data.pivot(index="fast", columns="slow", values=metric)
    fig = px.imshow(pivot, aspect="auto", origin="lower", title=f"Heatmap: {metric}")
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.title("FOREST 4.0 – Backtest & Grid")
    tab1, tab2, tab3 = st.tabs(["Back-test", "Grid Runner", "Grid Heat-map"])

    with tab1:
        st.header("Pojedynczy backtest (EMA cross)")
        file = st.file_uploader(
            "CSV z kolumnami: time(index), open, high, low, close", type=["csv"]
        )
        fast = st.slider("EMA fast", min_value=2, max_value=50, value=12, step=1)
        slow = st.slider("EMA slow", min_value=5, max_value=200, value=26, step=1)
        capital = st.number_input(
            "Kapitał początkowy", min_value=1000.0, value=100_000.0, step=1000.0
        )

        if st.button("Run back-test") and file is not None:
            df = load_csv(file)
            settings = BacktestSettings()
            settings.strategy.fast = fast
            settings.strategy.slow = slow
            settings.risk.initial_capital = capital
            res = run_backtest(df, settings)
            st.subheader("Equity")
            st.plotly_chart(
                px.line(res.equity_curve, title="Equity Curve"), use_container_width=True
            )
            st.subheader("Drawdown")
            st.plotly_chart(
                px.area(metrics(res.equity_curve), title="Drawdown"),
                use_container_width=True,
            )

    with tab2:
        st.header("Grid Runner")
        file2 = st.file_uploader("CSV (Grid)", type=["csv"], key="grid")
        fast_min, fast_max = st.slider("fast range", min_value=2, max_value=50, value=(5, 15))
        slow_min, slow_max = st.slider("slow range", min_value=10, max_value=200, value=(20, 60))
        if st.button("Run grid") and file2 is not None:
            df2 = load_csv(file2)
            base = BacktestSettings()
            resdf = run_grid(
                df2, base, range(fast_min, fast_max + 1), range(slow_min, slow_max + 1)
            )
            st.session_state["latest_grid"] = resdf
            st.success(f"Grid finished. Rows: {len(resdf)}")
            st.dataframe(resdf, use_container_width=True)

            # Pobierz wyniki jako CSV
            csv_bytes = resdf.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Pobierz wyniki (CSV)",
                data=csv_bytes,
                file_name="grid_results.csv",
                mime="text/csv",
            )

    with tab3:
        st.header("Heat-map")
        metric = st.selectbox(
            "Metryka", ["final_equity", "max_dd", "final_return", "cagr", "sharpe"]
        )
        dd_lim = st.slider(
            "Max DD limit (opcjonalnie)", min_value=0.0, max_value=1.0, value=1.0, step=0.01
        )
        if "latest_grid" in st.session_state:
            heatmap(st.session_state["latest_grid"], metric=metric, dd_lim=dd_lim)
        else:
            st.info("Najpierw uruchom grid w zakładce 'Grid Runner'.")


# Wrapper CLI: pozwala odpalić to jako `poetry run forest4-dashboard`
def main() -> int:
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve()
    cmd = [
        "streamlit",
        "run",
        str(here),
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    return subprocess.call(cmd)


# Gdy Streamlit uruchamia ten plik bezpośrednio (tak jak w Dockerze), po prostu renderuj UI.
if __name__ == "__main__":
    render()

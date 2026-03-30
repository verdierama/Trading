"""
Crypto Down Channel Scanner (Logarithmic Scale)
================================================
Uses Yahoo Finance - scans cryptos for descending channels on log scale.
Generates individual plots + summary dashboard.

Requirements:
    pip install yfinance numpy scipy pandas matplotlib tqdm
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import warnings
import os
import time
import math
import json

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Lire les symboles depuis le fichier JSON
with open('tickers\\crypto_symbols.json', 'r') as f:
    CRYPTO_SYMBOLS = json.load(f)

# ──────────────────────────────────────────────    # ◄◄ NOUVEAU
# STANDBY FILTER                                    # ◄◄ NOUVEAU
# ──────────────────────────────────────────────    # ◄◄ NOUVEAU
STANDBY_FILE = "tickers\\crypto_symbols_standby.json" # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
def load_standby_list(filepath=STANDBY_FILE):         # ◄◄ NOUVEAU
    """                                               # ◄◄ NOUVEAU
    Charge la liste des tickers a ignorer.            # ◄◄ NOUVEAU
    Format attendu : ["BTC-USD", "ETH-USD", ...]     # ◄◄ NOUVEAU
    """                                               # ◄◄ NOUVEAU
    if not os.path.exists(filepath):                  # ◄◄ NOUVEAU
        logger.info(                                  # ◄◄ NOUVEAU
            "Fichier standby non trouve: {} "         # ◄◄ NOUVEAU
            "-> aucun ticker exclu.".format(filepath) # ◄◄ NOUVEAU
        )                                             # ◄◄ NOUVEAU
        return set()                                  # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
    try:                                              # ◄◄ NOUVEAU
        with open(filepath, "r", encoding="utf-8") as f:  # ◄◄ NOUVEAU
            data = json.load(f)                       # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
        standby = set(                                # ◄◄ NOUVEAU
            s.strip().upper() for s in data           # ◄◄ NOUVEAU
            if isinstance(s, str)                     # ◄◄ NOUVEAU
        )                                             # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
        if standby:                                   # ◄◄ NOUVEAU
            logger.info(                              # ◄◄ NOUVEAU
                "Standby: {} tickers exclus: {}".format(  # ◄◄ NOUVEAU
                    len(standby),                     # ◄◄ NOUVEAU
                    ", ".join(sorted(standby))        # ◄◄ NOUVEAU
                )                                     # ◄◄ NOUVEAU
            )                                         # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
        return standby                                # ◄◄ NOUVEAU
                                                      # ◄◄ NOUVEAU
    except Exception as e:                            # ◄◄ NOUVEAU
        logger.warning(                               # ◄◄ NOUVEAU
            "Erreur lecture {}: {}".format(filepath, e)  # ◄◄ NOUVEAU
        )                                             # ◄◄ NOUVEAU
        return set()                                  # ◄◄ NOUVEAU


# ──────────────────────────────────────────────
# ADDITIONAL CONDITION CHECK
# ──────────────────────────────────────────────
def check_price_near_upper_channel(current_log_price, current_upper, current_lower):
    """
    Verifie si le cours est a moins de H/4 du haut du canal descendant.
    H = hauteur du canal (upper - lower en log)
    Le cours doit etre dans le quart superieur du canal.
    """
    H = current_upper - current_lower
    threshold = current_upper - (H / 4.0)
    return current_log_price >= threshold

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
class Config(object):
    SYMBOLS = CRYPTO_SYMBOLS
    TIMEFRAME = "1d"
    MIN_CANDLES = 50
    MIN_SWING_POINTS = 3
    SWING_ORDER = 10
    MAX_CHANNEL_WIDTH = 0.60
    MIN_CHANNEL_WIDTH = 0.05
    SLOPE_TOLERANCE = 0.40
    MAX_DEVIATION = 0.03
    PRICE_IN_CHANNEL_TOLERANCE = 0.02
    MIN_NEGATIVE_SLOPE = -1e-5
    MAX_VIOLATION_RATIO = 0.20
    REFINE_ITERATIONS = 3
    PLOT_RESULTS = True
    SHOW_PLOTS = True
    SAVE_PLOTS = True
    SHOW_INDIVIDUAL = True
    SHOW_DASHBOARD = True
    OUTPUT_DIR = "channel_plots"
    CSV_OUTPUT = "down_channel_results.csv"

# ──────────────────────────────────────────────
# DATA FETCHING (Yahoo Finance)
# ──────────────────────────────────────────────
class DataFetcher(object):
    def __init__(self, config):
        self.config = config

    def fetch_ohlcv(self, symbol):
        """Fetch OHLCV data with hybrid start logic."""
        try:
            fixed_start = datetime(2025, 9, 1)
            min_days = 180

            # 🔹 Récupération max historique
            df = yf.download(
                symbol,
                period="max",
                interval=self.config.TIMEFRAME,
                progress=False,
                auto_adjust=True,
            )

            if df is None or len(df) < self.config.MIN_CANDLES:
                return None

            # Nettoyage colonnes
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [str(c).lower().strip() for c in df.columns]

            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    return None

            df = df[required].astype(float).dropna()
            df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

            if len(df) < self.config.MIN_CANDLES:
                return None

            # 🔹 Gestion des dates
            df.index = pd.to_datetime(df.index)
            first_date = df.index.min()
            last_date = df.index.max()

            # Cas 1 : historique ancien → on coupe à 01/09/2025
            if first_date <= fixed_start:
                df = df[df.index >= fixed_start]

            # Cas 2 : crypto récente
            else:
                duration_days = (last_date - first_date).days

                if duration_days < min_days:
                    logger.debug(
                        f"{symbol} rejeté : historique trop court ({duration_days} jours)"
                    )
                    return None

            # Vérification finale
            if len(df) < self.config.MIN_CANDLES:
                return None

            return df

        except Exception as e:
            logger.debug("Error fetching {}: {}".format(symbol, e))
            return None

# ──────────────────────────────────────────────
# CHANNEL DETECTION ENGINE
# ──────────────────────────────────────────────
class ChannelDetector(object):
    def __init__(self, config):
        self.config = config

    def find_swing_points(self, series, order):
        highs = argrelextrema(series, np.greater_equal, order=order)[0]
        lows = argrelextrema(series, np.less_equal, order=order)[0]
        return highs, lows

    def fit_trendline(self, indices, values):
        if len(indices) < 2:
            return None
        idx_float = indices.astype(float)
        result = linregress(idx_float, values)
        return {
            "slope": result.slope,
            "intercept": result.intercept,
            "r_squared": result.rvalue ** 2,
            "stderr": result.stderr,
        }

    def refine_swing_points(self, indices, log_prices, line_type="upper"):
        idx = indices.copy()
        for _ in range(self.config.REFINE_ITERATIONS):
            if len(idx) < self.config.MIN_SWING_POINTS:
                break
            fit = self.fit_trendline(idx, log_prices[idx])
            if fit is None:
                break
            fitted_vals = fit["slope"] * idx.astype(float) + fit["intercept"]
            residuals = log_prices[idx] - fitted_vals

            if line_type == "upper":
                mask = residuals > -self.config.MAX_DEVIATION
            else:
                mask = residuals < self.config.MAX_DEVIATION

            if mask.sum() < self.config.MIN_SWING_POINTS:
                break
            idx = idx[mask]
        return idx

    def detect_channel(self, df):

        body_high = np.maximum(df["open"].values, df["close"].values)
        body_low = np.minimum(df["open"].values, df["close"].values)

        log_body_high = np.log(body_high)
        log_body_low = np.log(body_low)

        log_high = np.log(df["high"].values)
        log_low = np.log(df["low"].values)
        log_close = np.log(df["close"].values)

        n = len(df)
        indices = np.arange(n, dtype=float)

        swing_high_idx, _ = self.find_swing_points(
            log_body_high, self.config.SWING_ORDER
        )

        _, swing_low_idx = self.find_swing_points(
            log_body_low, self.config.SWING_ORDER
        )

        if len(swing_high_idx) < self.config.MIN_SWING_POINTS:
            return None

        if len(swing_low_idx) < self.config.MIN_SWING_POINTS:
            return None

        swing_high_idx = self.refine_swing_points(
            swing_high_idx, log_body_high, "upper"
        )

        swing_low_idx = self.refine_swing_points(
            swing_low_idx, log_body_low, "lower"
        )

        if len(swing_high_idx) < self.config.MIN_SWING_POINTS:
            return None

        if len(swing_low_idx) < self.config.MIN_SWING_POINTS:
            return None

        first_high = swing_high_idx[0]
        first_low = swing_low_idx[0]

        if first_low <= first_high:

            valid_lows = swing_low_idx[swing_low_idx > first_high]

            if len(valid_lows) == 0:
                return None

            swing_low_idx = valid_lows

        upper_fit = self.fit_trendline(
            swing_high_idx, log_body_high[swing_high_idx]
        )

        if upper_fit is None:
            return None

        upper_slope = upper_fit["slope"]
        upper_intercept = upper_fit["intercept"]

        if upper_slope > self.config.MIN_NEGATIVE_SLOPE:
            return None

        m = upper_slope

        support_intercepts = log_body_low[swing_low_idx] - m * swing_low_idx

        lower_intercept = np.min(support_intercepts)

        lower_slope = m

        upper_line = m * indices + upper_intercept
        lower_line = m * indices + lower_intercept

        above_upper = log_body_high > (
            upper_line + self.config.MAX_DEVIATION
        )

        if np.any(above_upper):
            return None

        below_lower = log_body_low < (
            lower_line - self.config.MAX_DEVIATION
        )

        violation_ratio = below_lower.sum() / float(n)

        if violation_ratio > self.config.MAX_VIOLATION_RATIO:
            return None

        channel_width = np.mean(upper_line - lower_line)

        if channel_width < self.config.MIN_CHANNEL_WIDTH:
            return None

        if channel_width > self.config.MAX_CHANNEL_WIDTH:
            return None

        current_log_price = log_close[-1]
        current_upper = upper_line[-1]
        current_lower = lower_line[-1]

        H = current_upper - current_lower

        if current_log_price > current_upper:
            return None

        if current_log_price < current_upper - H/4:
            return None

        close_prices = df["close"].values
        if len(close_prices) >= 9:
            ma9 = pd.Series(close_prices).rolling(9).mean().values
            current_ma9 = np.log(ma9[-1])

            if current_log_price >= current_ma9:
                return None

            if current_ma9 > current_upper:
                return None
            if current_ma9 < current_upper - H/4:
                return None
        else:
            return None

        containment = 1.0 - violation_ratio
        avg_r2 = upper_fit["r_squared"]

        score = (avg_r2 * 0.6 + containment * 0.4) * 100.0

        current_width = current_upper - current_lower

        if current_width > 0:
            position = (current_log_price - current_lower) / current_width
        else:
            position = 0.5

        log_return_100d = upper_slope * 100.0
        decline_100d_pct = (np.exp(log_return_100d) - 1.0) * 100.0

        future_n = 20

        future_idx = np.arange(n, n + future_n, dtype=float)

        future_upper = m * future_idx + upper_intercept
        future_lower = m * future_idx + lower_intercept

        return {
            "upper_slope": upper_slope,
            "lower_slope": lower_slope,
            "upper_intercept": upper_intercept,
            "lower_intercept": lower_intercept,
            "upper_r2": upper_fit["r_squared"],
            "lower_r2": upper_fit["r_squared"],
            "channel_width_log": channel_width,
            "channel_width_pct": (np.exp(channel_width) - 1.0) * 100.0,
            "violation_ratio": violation_ratio,
            "slope_diff_ratio": 0.0,
            "score": round(score, 1),
            "position_in_channel": round(position, 2),
            "decline_100d_pct": round(decline_100d_pct, 1),
            "swing_high_idx": swing_high_idx,
            "swing_low_idx": swing_low_idx,
            "upper_line": upper_line,
            "lower_line": lower_line,
            "future_upper": future_upper,
            "future_lower": future_lower,
            "future_n": future_n,
            "num_swing_highs": len(swing_high_idx),
            "num_swing_lows": len(swing_low_idx),
            "ma9": ma9,
        }

# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────
class ChannelPlotter(object):
    def __init__(self, config):
        self.config = config
        if config.SAVE_PLOTS:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def format_price(self, price):
        """Format price for display."""
        if price >= 10000:
            return "${:,.0f}".format(price)
        elif price >= 1:
            return "${:,.2f}".format(price)
        elif price >= 0.01:
            return "${:,.4f}".format(price)
        else:
            return "${:,.8f}".format(price)

    def plot_individual(self, symbol, df, channel):
        n = len(df)
        future_n = channel["future_n"]
        total_len = n + future_n

        log_close = np.log(df["close"].values)
        log_high = np.log(df["high"].values)
        log_low = np.log(df["low"].values)
        log_open = np.log(df["open"].values)
        volume = df["volume"].values

        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[3, 1], hspace=0.05, figure=fig
        )
        ax_price = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)

        x_data = np.arange(n)
        x_future = np.arange(n, total_len)

        colors = []
        for i in range(n):
            if df["close"].values[i] >= df["open"].values[i]:
                colors.append("#26a69a")
            else:
                colors.append("#ef5350")

        for i in range(n):
            ax_price.plot(
                [i, i], [log_low[i], log_high[i]],
                color=colors[i], linewidth=0.8, alpha=0.6
            )

        body_top = np.maximum(log_open, log_close)
        body_bot = np.minimum(log_open, log_close)
        ax_price.bar(
            x_data, body_top - body_bot, bottom=body_bot,
            width=0.7, color=colors, alpha=0.85, edgecolor="none"
        )

        ax_price.plot(
            x_data, channel["upper_line"],
            color="#ff5252", linewidth=2.5, linestyle="--",
            label="Upper Resistance (R2={:.2f})".format(channel["upper_r2"])
        )
        ax_price.plot(
            x_data, channel["lower_line"],
            color="#69f0ae", linewidth=2.5, linestyle="--",
            label="Lower Support (R2={:.2f})".format(channel["lower_r2"])
        )

        ax_price.fill_between(
            x_data, channel["lower_line"], channel["upper_line"],
            color="dodgerblue", alpha=0.06
        )

        ax_price.plot(
            x_future, channel["future_upper"],
            color="#ff5252", linewidth=1.5, linestyle=":",
            alpha=0.5
        )
        ax_price.plot(
            x_future, channel["future_lower"],
            color="#69f0ae", linewidth=1.5, linestyle=":",
            alpha=0.5
        )
        ax_price.fill_between(
            x_future, channel["future_lower"], channel["future_upper"],
            color="dodgerblue", alpha=0.03
        )

        midline = (channel["upper_line"] + channel["lower_line"]) / 2.0
        ax_price.plot(
            x_data, midline,
            color="#42a5f5", linewidth=1, linestyle=":",
            alpha=0.4, label="Midline"
        )

        ma9 = channel.get("ma9", None)
        if ma9 is not None:
            log_ma9 = np.log(
                pd.Series(df["close"].values).rolling(9).mean().values
            )
            ax_price.plot(
                x_data, log_ma9,
                color="#FFD700",
                linewidth=1.5,
                linestyle="-",
                alpha=0.8,
                label="MA9"
            )

        sh = channel["swing_high_idx"]
        sl = channel["swing_low_idx"]
        ax_price.scatter(
            sh, log_high[sh], marker="v", color="#ff1744",
            s=100, zorder=5, edgecolors="white", linewidths=0.5,
            label="Swing Highs ({})".format(len(sh))
        )
        ax_price.scatter(
            sl, log_low[sl], marker="^", color="#00e676",
            s=100, zorder=5, edgecolors="white", linewidths=0.5,
            label="Swing Lows ({})".format(len(sl))
        )

        current_price = df["close"].values[-1]
        ax_price.scatter(
            n - 1, log_close[-1], marker="D", color="#ffd740",
            s=150, zorder=6, edgecolors="black", linewidths=1.5,
            label="Current: {}".format(self.format_price(current_price))
        )

        ax_price.axhline(
            y=log_close[-1], color="#ffd740",
            linewidth=0.8, linestyle="-.", alpha=0.4
        )

        pos = channel["position_in_channel"]
        current_upper = channel["upper_line"][-1]
        current_lower = channel["lower_line"][-1]

        bar_x = n + future_n + 2
        ax_price.plot(
            [bar_x, bar_x], [current_lower, current_upper],
            color="white", linewidth=3, alpha=0.3
        )
        marker_y = current_lower + pos * (current_upper - current_lower)
        ax_price.scatter(
            bar_x, marker_y, marker=">",
            color="#ffd740", s=200, zorder=7
        )

        ax_price.annotate(
            self.format_price(np.exp(current_upper)),
            xy=(n - 1, current_upper),
            xytext=(n + 2, current_upper),
            color="#ff5252", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#ff5252", alpha=0.5)
        )
        ax_price.annotate(
            self.format_price(np.exp(current_lower)),
            xy=(n - 1, current_lower),
            xytext=(n + 2, current_lower),
            color="#69f0ae", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#69f0ae", alpha=0.5)
        )

        vol_colors = []
        for i in range(n):
            if df["close"].values[i] >= df["open"].values[i]:
                vol_colors.append("#26a69a")
            else:
                vol_colors.append("#ef5350")

        ax_vol.bar(x_data, volume, width=0.7, color=vol_colors, alpha=0.6)

        if n > 20:
            vol_ma = pd.Series(volume).rolling(20).mean().values
            ax_vol.plot(x_data, vol_ma, color="#42a5f5", linewidth=1.2, alpha=0.8)

        dates = df.index
        tick_step = max(1, n // 10)
        tick_positions = list(range(0, n, tick_step))
        tick_labels = [dates[i].strftime("%b %d\n%Y") for i in tick_positions]
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, rotation=0, fontsize=8)
        plt.setp(ax_price.get_xticklabels(), visible=False)

        ax_price2 = ax_price.twinx()
        yticks = ax_price.get_yticks()
        ax_price2.set_ylim(ax_price.get_ylim())
        ax_price2.set_yticks(yticks)
        price_labels = [self.format_price(np.exp(y)) for y in yticks]
        ax_price2.set_yticklabels(price_labels, fontsize=8)
        ax_price2.tick_params(colors="#aaaaaa")
        ax_price2.set_ylabel("Price (USD)", color="#aaaaaa", fontsize=10)

        dark_bg = "#0d1117"
        panel_bg = "#161b22"

        fig.patch.set_facecolor(dark_bg)
        ax_price.set_facecolor(panel_bg)
        ax_vol.set_facecolor(panel_bg)

        ax_price.tick_params(colors="#aaaaaa")
        ax_vol.tick_params(colors="#aaaaaa")
        ax_price.set_ylabel("Log Price", color="#aaaaaa", fontsize=10)
        ax_vol.set_ylabel("Volume", color="#aaaaaa", fontsize=10)
        ax_vol.set_xlabel("Date", color="#aaaaaa", fontsize=10)

        ax_price.grid(True, alpha=0.08, color="gray")
        ax_vol.grid(True, alpha=0.08, color="gray")

        for spine in ax_price.spines.values():
            spine.set_color("#30363d")
        for spine in ax_vol.spines.values():
            spine.set_color("#30363d")
        for spine in ax_price2.spines.values():
            spine.set_color("#30363d")

        ax_price.legend(
            loc="upper left", fontsize=8,
            facecolor=panel_bg, edgecolor="#30363d",
            labelcolor="#cccccc", framealpha=0.9
        )

        if pos < 0.3:
            pos_label = "NEAR BOTTOM"
            pos_color = "#69f0ae"
        elif pos > 0.7:
            pos_label = "NEAR TOP"
            pos_color = "#ff5252"
        else:
            pos_label = "MID-CHANNEL"
            pos_color = "#42a5f5"

        title_text = (
            "{sym}  |  Descending Channel (Log Scale)  |  "
            "Score: {score}  |  Decline (100d): {dec:.1f}%"
        ).format(
            sym=symbol,
            score=channel["score"],
            dec=channel["decline_100d_pct"]
        )
        subtitle_text = (
            "Width: {w:.1f}%  |  Position: {p:.0%} ({lbl})  |  "
            "Swing H/L: {sh}/{sl}  |  Timeframe: {tf}  |  "
            "Price < MA9"
        ).format(
            w=channel["channel_width_pct"],
            p=pos, lbl=pos_label,
            sh=len(sh), sl=len(sl),
            tf=self.config.TIMEFRAME
        )

        fig.suptitle(
            title_text,
            color="white", fontsize=14, fontweight="bold",
            y=0.98
        )
        ax_price.set_title(
            subtitle_text,
            color="#aaaaaa", fontsize=10, pad=10
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if self.config.SAVE_PLOTS:
            safe_name = symbol.replace("/", "_").replace("-", "_")
            filepath = os.path.join(
                self.config.OUTPUT_DIR,
                "{}_channel.png".format(safe_name)
            )
            plt.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
            logger.info("  Plot saved: {}".format(filepath))

        if self.config.SHOW_INDIVIDUAL:
            plt.show()
        else:
            plt.close(fig)

    def plot_dashboard(self, all_results):
        num = len(all_results)
        if num == 0:
            return

        cols = min(4, num)
        rows = math.ceil(num / cols)

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5 * cols, 4 * rows),
            squeeze=False
        )

        dark_bg = "#0d1117"
        panel_bg = "#161b22"
        fig.patch.set_facecolor(dark_bg)

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]
            ax.set_facecolor(panel_bg)

            if idx >= num:
                ax.axis("off")
                continue

            item = all_results[idx]
            symbol = item["symbol"]
            df = item["df"]
            channel = item["channel"]

            n = len(df)
            x = np.arange(n)
            log_close = np.log(df["close"].values)
            log_high = np.log(df["high"].values)
            log_low = np.log(df["low"].values)

            ax.plot(x, log_close, color="white", linewidth=0.8, alpha=0.8)

            colors = []
            for i in range(n):
                if df["close"].values[i] >= df["open"].values[i]:
                    colors.append("#26a69a")
                else:
                    colors.append("#ef5350")
            ax.bar(
                x, log_high - log_low, bottom=log_low,
                width=0.8, color=colors, alpha=0.25
            )

            ax.plot(x, channel["upper_line"], "r--", linewidth=1.5)
            ax.plot(x, channel["lower_line"], "g--", linewidth=1.5)
            ax.fill_between(
                x, channel["lower_line"], channel["upper_line"],
                color="dodgerblue", alpha=0.06
            )

            log_ma9 = np.log(
                pd.Series(df["close"].values).rolling(9).mean().values
            )
            ax.plot(x, log_ma9, color="#FFD700", linewidth=1.0,
                    linestyle="-", alpha=0.8)

            sh = channel["swing_high_idx"]
            sl = channel["swing_low_idx"]
            ax.scatter(sh, log_high[sh], marker="v", color="red", s=30, zorder=5)
            ax.scatter(sl, log_low[sl], marker="^", color="lime", s=30, zorder=5)

            ax.scatter(
                n - 1, log_close[-1], marker="D",
                color="#ffd740", s=60, zorder=6, edgecolors="black"
            )

            pos = channel["position_in_channel"]
            if pos < 0.3:
                badge_color = "#69f0ae"
                badge_text = "LOW"
            elif pos > 0.7:
                badge_color = "#ff5252"
                badge_text = "HIGH"
            else:
                badge_color = "#42a5f5"
                badge_text = "MID"

            current_price = df["close"].values[-1]
            title = "{} | {} | S:{}\n{} | Dec100d:{:.0f}% | W:{:.0f}%".format(
                symbol,
                self.format_price(current_price),
                channel["score"],
                badge_text,
                channel["decline_100d_pct"],
                channel["channel_width_pct"]
            )
            ax.set_title(title, color=badge_color, fontsize=9, fontweight="bold")

            ax.grid(True, alpha=0.06, color="gray")
            ax.tick_params(colors="#666666", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

            tick_step = max(1, n // 4)
            tick_pos = list(range(0, n, tick_step))
            tick_lbl = [df.index[i].strftime("%b") for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=7)

        fig.suptitle(
            "CRYPTO DOWN CHANNEL DASHBOARD  |  "
            "{} Channels Found  |  {}".format(
                num,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ),
            color="white", fontsize=16, fontweight="bold",
            y=1.01
        )

        plt.tight_layout()

        if self.config.SAVE_PLOTS:
            filepath = os.path.join(self.config.OUTPUT_DIR, "DASHBOARD.png")
            plt.savefig(
                filepath, dpi=150,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight"
            )
            logger.info("Dashboard saved: {}".format(filepath))

        if self.config.SHOW_DASHBOARD:
            plt.show()
        else:
            plt.close(fig)


# ──────────────────────────────────────────────
# MAIN SCANNER
# ──────────────────────────────────────────────
class DownChannelScanner(object):
    def __init__(self, config=None):
        self.config = config or Config()
        self.fetcher = DataFetcher(self.config)
        self.detector = ChannelDetector(self.config)
        self.plotter = ChannelPlotter(self.config)
        self.standby_symbols = load_standby_list()        # ◄◄ NOUVEAU

    def scan(self):
        symbols = self.config.SYMBOLS
        results = []
        plot_data = []
        skipped_count = 0                                 # ◄◄ NOUVEAU

        logger.info(
            "Scanning {} crypto symbols for descending channels...".format(
                len(symbols)
            )
        )

        for symbol in tqdm(symbols, desc="Scanning cryptos"):

            # ── Check standby ──                         # ◄◄ NOUVEAU
            if symbol.strip().upper() in self.standby_symbols:  # ◄◄ NOUVEAU
                skipped_count += 1                        # ◄◄ NOUVEAU
                continue                                  # ◄◄ NOUVEAU

            df = self.fetcher.fetch_ohlcv(symbol)
            if df is None:
                continue

            channel = self.detector.detect_channel(df)
            if channel is None:
                continue

            current_price = float(df["close"].iloc[-1])
            logger.info(
                "  FOUND: {} | price={} | score={} | "
                "width={:.1f}% | pos={:.0%}".format(
                    symbol,
                    self.plotter.format_price(current_price),
                    channel["score"],
                    channel["channel_width_pct"],
                    channel["position_in_channel"]
                )
            )

            results.append({
                "symbol": symbol,
                "current_price": round(current_price, 6),
                "score": channel["score"],
                "decline_100d_pct": channel["decline_100d_pct"],
                "channel_width_pct": round(channel["channel_width_pct"], 1),
                "position_in_channel": channel["position_in_channel"],
                "upper_slope": round(channel["upper_slope"], 6),
                "lower_slope": round(channel["lower_slope"], 6),
                "upper_r2": round(channel["upper_r2"], 3),
                "lower_r2": round(channel["lower_r2"], 3),
                "swing_highs": channel["num_swing_highs"],
                "swing_lows": channel["num_swing_lows"],
                "violation_ratio": round(channel["violation_ratio"], 3),
            })

            plot_data.append({
                "symbol": symbol,
                "df": df.copy(),
                "channel": channel,
            })

            if self.config.PLOT_RESULTS:
                try:
                    self.plotter.plot_individual(symbol, df, channel)
                except Exception as e:
                    logger.warning("  Plot failed for {}: {}".format(symbol, e))

            time.sleep(0.3)

        # ── Résumé standby ──                            # ◄◄ NOUVEAU
        if skipped_count > 0:                             # ◄◄ NOUVEAU
            logger.info(                                  # ◄◄ NOUVEAU
                "Standby: {} tickers ignores.".format(    # ◄◄ NOUVEAU
                    skipped_count                         # ◄◄ NOUVEAU
                )                                         # ◄◄ NOUVEAU
            )                                             # ◄◄ NOUVEAU

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.sort_values("decline_100d_pct", ascending=True, inplace=True)
            results_df.reset_index(drop=True, inplace=True)

        decline_order = {
            r["symbol"]: r["decline_100d_pct"] for r in results
        }

        plot_data.sort(
            key=lambda x: decline_order.get(x["symbol"], 0)
        )

        if self.config.SHOW_DASHBOARD and len(plot_data) > 0:
            logger.info("Generating summary dashboard...")
            try:
                self.plotter.plot_dashboard(plot_data)
            except Exception as e:
                logger.warning("Dashboard plot failed: {}".format(e))

        return results_df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def main():
    config = Config()

    config.LOOKBACK_DAYS = 180
    config.TIMEFRAME = "1d"
    config.SWING_ORDER = 10
    config.PLOT_RESULTS = True
    config.SAVE_PLOTS = True
    config.SHOW_INDIVIDUAL = True
    config.SHOW_DASHBOARD = True

    print("")
    print("=" * 60)
    print("  CRYPTO DOWN CHANNEL SCANNER (Logarithmic Scale)")
    print("  Data Source: Yahoo Finance")
    print("  Lookback: {} days | Timeframe: {}".format(
        config.LOOKBACK_DAYS, config.TIMEFRAME
    ))
    print("  Symbols to scan: {}".format(len(config.SYMBOLS)))
    print("=" * 60)
    print("")

    scanner = DownChannelScanner(config)
    results = scanner.scan()

    if results.empty:
        print("")
        print("No descending channels found with current parameters.")
        print("Try adjusting: LOOKBACK_DAYS, SWING_ORDER, SLOPE_TOLERANCE")
    else:
        print("")
        print("=" * 100)
        print("  RESULTS: CRYPTO DOWN CHANNELS (LOG SCALE)")
        print("=" * 100)

        display_cols = [
            "symbol", "current_price", "score", "decline_100d_pct",
            "channel_width_pct", "position_in_channel",
            "upper_r2", "lower_r2", "swing_highs", "swing_lows"
        ]
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=True))

        print("=" * 100)
        print("")
        print("Found {} cryptos in a descending channel.".format(len(results)))

        results.to_csv(config.CSV_OUTPUT, index=False)
        logger.info("Results saved to {}".format(config.CSV_OUTPUT))

        print("")
        print("TOP PICKS (highest quality channels):")
        print("-" * 75)
        for _, row in results.head(10).iterrows():
            if row["position_in_channel"] < 0.3:
                pos_label = "NEAR BOTTOM"
            elif row["position_in_channel"] > 0.7:
                pos_label = "NEAR TOP"
            else:
                pos_label = "MID-CHANNEL"

            price = row["current_price"]
            if price >= 1:
                price_str = "${:>10,.2f}".format(price)
            elif price >= 0.01:
                price_str = "${:>10,.4f}".format(price)
            else:
                price_str = "${:>10,.8f}".format(price)

            print(
                "  {:>12s} | {} | Score: {:5.1f} | "
                "Decline: {:6.1f}% | Width: {:5.1f}% | {}".format(
                    row["symbol"], price_str, row["score"],
                    row["decline_100d_pct"], row["channel_width_pct"],
                    pos_label
                )
            )

        if config.SAVE_PLOTS:
            print("")
            print("Individual charts + dashboard saved in: {}/".format(
                config.OUTPUT_DIR
            ))

    print("")
    print("Done!")


if __name__ == "__main__":
    main()
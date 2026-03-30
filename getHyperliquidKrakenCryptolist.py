import requests
import re
import json
import os
import yfinance as yf
import time
import pandas as pd

# Créer les dossiers
os.makedirs("tickers", exist_ok=True)
cache_dir = "tickers/cache_yf"
os.makedirs(cache_dir, exist_ok=True)

# =========================
# Charger mapping manuel
# =========================
mapping_file = "tickers/yahoo_crypto_mapping.json"
if os.path.exists(mapping_file):
    with open(mapping_file, "r") as f:
        yahoo_mapping = json.load(f)
else:
    yahoo_mapping = {}
    print(f"Aucun mapping manuel trouvé à {mapping_file}.")

# =========================
# Charger liste ignore
# =========================
ignore_file = "tickers/yahoo_crypto_ignore.json"
if os.path.exists(ignore_file):
    try:
        with open(ignore_file, "r") as f:
            ignore_tickers = set(json.load(f))
    except Exception as e:
        print(f"⚠️ Erreur lecture ignore file, reset : {e}")
        ignore_tickers = set()
else:
    ignore_tickers = set()
    print(f"Aucun fichier d'exclusion trouvé à {ignore_file}.")

# =========================
# Kraken Perpetuals
# =========================
kraken_url = "https://futures.kraken.com/derivatives/api/v3/instruments"
kraken_data = requests.get(kraken_url).json()
kraken_tickers = []
for inst in kraken_data["instruments"]:
    symbol = inst["symbol"]
    if symbol.startswith(("PF_", "PI_")):
        pair = symbol.split("_")[1]
        pair = re.sub(r'^[0-9]+', '', pair)
        if pair.startswith("XBT"):
            pair = pair.replace("XBT", "BTC")
        if pair.endswith("USD"):
            pair = pair[:-3] + "-USD"
        kraken_tickers.append(pair.upper())

# =========================
# Hyperliquid Perpetuals
# =========================
API_URL = "https://api.hyperliquid.xyz/info"

def fetch_perp_markets():
    try:
        response = requests.post(API_URL, json={"type": "meta"})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Erreur Hyperliquid : {e}")
        return None

def list_hyperliquid_assets():
    data = fetch_perp_markets()
    if not data:
        return []
    tickers = []
    for asset in data.get("universe", []):
        symbol = asset.get("name")
        if symbol:
            t = re.sub(r'^[0-9]+', '', symbol).upper()
            if not t.endswith("-USD"):
                t += "-USD"
            tickers.append(t)
    return tickers

hyperliquid_tickers = list_hyperliquid_assets()

# =========================
# Kraken Margin Assets
# =========================
KRAKEN_SPOT_API = "https://api.kraken.com/0/public/AssetPairs"

def fetch_kraken_margin_assets():
    try:
        response = requests.get(KRAKEN_SPOT_API)
        data = response.json()
        if data["error"]:
            print("Erreur API Kraken Spot :", data["error"])
            return []
        pairs = data["result"]
        margin_assets = set()
        for pair_name, pair_data in pairs.items():
            if "margin_call" in pair_data and "margin_stop" in pair_data:
                base = pair_data.get("base", "").replace("XBT", "BTC").replace("X", "").replace("Z", "")
                if base:
                    margin_assets.add(f"{base}-USD")
        return sorted(margin_assets)
    except Exception as e:
        print("Erreur récupération margin Kraken :", e)
        return []

kraken_margin_tickers = fetch_kraken_margin_assets()

# =========================
# Fusion de toutes les sources
# =========================
all_tickers = sorted(list(set(
    kraken_tickers + hyperliquid_tickers + kraken_margin_tickers
)))
print(f"Kraken perps: {len(kraken_tickers)}, Hyperliquid perps: {len(hyperliquid_tickers)}, Kraken margin: {len(kraken_margin_tickers)}, Total: {len(all_tickers)}")

# =========================
# Fonction Yahoo avec cache et vérif historique
# =========================
def find_yahoo_ticker(base_ticker):
    base_ticker_upper = base_ticker.upper()
    if base_ticker_upper in ignore_tickers:
        return None

    tick = yahoo_mapping.get(base_ticker_upper, base_ticker_upper)
    cache_file = os.path.join(cache_dir, f"{tick}.csv")

    # Charger depuis cache si existe
    if os.path.exists(cache_file):
        hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        try:
            t = yf.Ticker(tick)
            hist = t.history(period="max")
            if hist.empty:
                return None
            hist.to_csv(cache_file)
        except Exception as e:
            print(f"Erreur Yahoo Finance pour {tick} : {e}")
            return None

    # Convertir index en tz-naive
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)

    current_price = hist["Close"].iloc[-1]
    cutoff_date = pd.to_datetime("2025-10-01")  # tz-naive
    hist_before = hist[hist.index < cutoff_date]

    # Vérification : plage [current/2 ; current]
    if not hist_before.empty:
        lower_bound = current_price / 2
        upper_bound = current_price

        mask = (hist_before["Close"] >= lower_bound) & (hist_before["Close"] <= upper_bound)

        if mask.any():
            print(f"⚠️ {tick} ignoré : historique contient valeur entre {lower_bound:.4f} et {upper_bound:.4f}")
            ignore_tickers.add(base_ticker_upper)

            with open(ignore_file, "w") as f:
                json.dump(sorted(list(ignore_tickers)), f, indent=4)

            return None
    return tick

# =========================
# Vérification finale
# =========================
print("Vérification des tickers Yahoo...")
final_tickers = []
for t in all_tickers:
    real_ticker = find_yahoo_ticker(t)
    if real_ticker:
        final_tickers.append(real_ticker)
    time.sleep(0.1)

final_tickers = sorted(list(set(final_tickers)))

# =========================
# Sauvegarde JSON final
# =========================
json_path = "tickers/crypto_symbols.json"
with open(json_path, "w") as f:
    json.dump(final_tickers, f, indent=4)

# Sauvegarde ignore
with open(ignore_file, "w") as f:
    json.dump(sorted(list(ignore_tickers)), f, indent=4)

print(f"JSON final généré avec {len(final_tickers)} tickers : {json_path}")
print(final_tickers[:20])
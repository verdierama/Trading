import os
import glob
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from us_top_1000 import US_TOP_1000
import requests
import warnings
warnings.filterwarnings('ignore')

isin_mapping = pd.read_csv("isin_mapping.csv").set_index("ticker")["isin"].to_dict()

# ============================================================================
# CONFIGURATION - ADAPTÉE POUR WEEKLY
# ============================================================================
CONFIG = {
    'period': '3y',              # 3 ans pour avoir assez de données weekly
    'interval': '1wk',           # *** WEEKLY ***
    'min_lows': 3,
    'min_highs': 2,
    'tolerance': 0.25,
    'pivot_order': 3,            # Réduit pour weekly (3 semaines de chaque côté)
    'min_slope': 0.00001,        # Ajusté pour weekly
    'min_r_squared': 0.60,
    'ath_threshold': 0.90,
    'recovery_drop': 0.5,
    'recovery_threshold': 0.80,
    'min_bars': 50,              # Minimum 50 semaines de données
}

# ============================================================================
# CONFIGURATION API - FINANCIAL MODELING PREP
# ============================================================================
FMP_API_KEY = "G7rCQl7PIiQBE2gkRpXZ81DstqXjYhiO"  # ⚠️ Remplacez par votre clé API gratuite
                                # Inscription: https://site.financialmodelingprep.com/

# ============================================================================
# FONCTION POUR RÉCUPÉRER LES TICKERS US > 1 MILLIARD $
# ============================================================================
def get_us_tickers_above_1b(api_key=FMP_API_KEY, min_market_cap=1_000_000_000):
    """
    Récupère tous les tickers US avec une capitalisation > 1 milliard $
    via l'API Financial Modeling Prep
    
    Args:
        api_key: Clé API FMP (gratuite)
        min_market_cap: Capitalisation minimum en dollars (défaut: 1 milliard)
    
    Returns:
        list: Liste des tickers triés alphabétiquement
    """
    print(f"   📡 Connexion à Financial Modeling Prep API...")
    
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    
    params = {
        "marketCapMoreThan": min_market_cap,
        "exchange": "NYSE,NASDAQ,AMEX",
        "isActivelyTrading": True,
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                # Filtrer les tickers valides (pas de caractères spéciaux problématiques)
                tickers = []
                for stock in data:
                    symbol = stock.get('symbol', '')
                    # Exclure les tickers avec des caractères problématiques pour yfinance
                    if symbol and '.' not in symbol and '-' not in symbol and '^' not in symbol:
                        tickers.append(symbol)
                    elif symbol and '-' in symbol:
                        # Garder certains tickers avec tiret (ex: BRK-B, BF-B)
                        tickers.append(symbol)
                
                tickers = sorted(list(set(tickers)))
                print(f"   ✅ {len(tickers)} tickers US récupérés (cap > ${min_market_cap/1e9:.0f}B)")
                return tickers
            else:
                print(f"   ⚠️ Réponse API vide ou format inattendu")
                print(f"   💡 Vérifiez votre clé API ou les limites de requêtes")
                return []
        
        elif response.status_code == 401:
            print(f"   ❌ Erreur 401: Clé API invalide ou expirée")
            print(f"   💡 Obtenez une clé gratuite sur: https://site.financialmodelingprep.com/")
            return []
        
        elif response.status_code == 403:
            print(f"   ❌ Erreur 403: Limite de requêtes atteinte")
            print(f"   💡 Attendez quelques minutes ou passez à un plan payant")
            return []
        
        else:
            print(f"   ❌ Erreur API: {response.status_code}")
            return []
    
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout: L'API ne répond pas")
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Erreur de connexion: {e}")
        return []


def get_us_tickers_with_fallback(api_key=FMP_API_KEY):
    """
    Récupère les tickers US avec fallback sur une liste statique si l'API échoue
    """
    # Essayer l'API en premier
    tickers = get_us_tickers_above_1b(api_key)
    
    if len(tickers) > 0:
        return tickers
    
    # Fallback: liste statique S&P 500
    print(f"   ⚠️ Fallback sur la liste S&P 500 statique...")
    return SP500_TICKERS_FALLBACK


# ============================================================================
# LISTE FALLBACK S&P 500 (utilisée si l'API échoue)
# ============================================================================
SP500_TICKERS_FALLBACK = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", 
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", 
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", 
    "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", 
    "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", 
    "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", 
    "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", 
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", 
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", 
    "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", 
    "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", 
    "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", 
    "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", 
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", 
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", 
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", 
    "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", 
    "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FLT", 
    "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", 
    "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", 
    "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", 
    "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", 
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", 
    "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", 
    "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", 
    "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", 
    "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", 
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", 
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", 
    "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", 
    "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", 
    "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", 
    "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", 
    "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", 
    "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", 
    "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", 
    "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", 
    "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", 
    "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", 
    "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", 
    "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", 
    "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", 
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", 
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", 
    "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", 
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VICI", "VLO", 
    "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", 
    "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", 
    "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", 
    "ZBRA", "ZTS"
]

# ============================================================================
# LISTES DES INDICES EUROPÉENS (inchangées)
# ============================================================================

# CAC 40 (France - Paris) - Suffixe .PA
CAC40_TICKERS = [
    "AC.PA", "AI.PA", "AIR.PA", "CS.PA", "BNP.PA", "EN.PA", "BVI.PA",
    "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "FGR.PA", "ENGI.PA", "EL.PA", "ERF.PA",
    "ENX.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA",
    "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA",
    "STMPA.PA", "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
]

# SBF 120 (France - Paris) - Inclut CAC40 + 80 autres - Suffixe .PA
SBF120_TICKERS = [
    # CAC 40
    "AC.PA", "AI.PA", "AIR.PA", "CS.PA", "BNP.PA", "EN.PA", "BVI.PA",
    "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "FGR.PA", "ENGI.PA", "EL.PA", "ERF.PA",
    "ENX.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA",
    "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA",
    "STMPA.PA", "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
    # SBF 120 (hors CAC40)
    "ABVX.PA", "AF.PA", "ALO.PA", "ATE.PA", "AMUN.PA", "ARG.PA", "ATO.PA",
    "AYV.PA", "BB.PA", "BIM.PA", "BOL.PA", "CARM.PA", "CLARI.PA", "COFA.PA",
    "COV.PA", "AM.PA", "DSY.PA", "DBG.PA", "EDEN.PA", "DEC.PA",
    "ELIOR.PA", "ELIS.PA", "EMEIS.PA", "ERA.PA", "EXA.PA", "RF.PA", "ETL.PA", "EXENS.PA", "FDJU.PA", "FRVIA.PA", "GFC.PA", "GET.PA",
    "MMT.PA", "ADP.PA", "GTT.PA", "ICAD.PA", "IDL.PA","NK.PA", "IPN.PA", "IPS.PA", "ITP.PA", "LI.PA",
    "FII.PA", "KOF.PA", "LACR.PA", "ALBON.PA", "LOUP.PA", "MEDCL.PA", "MERY.PA", "MF.PA",
    "MMB.PA", "MRN.PA", "NANO.PA", "NEX.PA", "NXI.PA", "ODET.PA", "OPM.PA",
    "PAR.PA", "PLX.PA","RI.PA", "PVL.PA", "RAL.PA", "RBT.PA", "RCO.PA", "RUI.PA", "RXL.PA",
    "DIM.PA", "SCR.PA", "SCHP.PA", "SESG.PA", "SK.PA",
    "SMCP.PA", "SOI.PA", "SOP.PA", "SPIE.PA", "SRP.PA", "STF.PA",
    "SW.PA", "TE.PA", "TEP.PA", "TFI.PA", "TRI.PA", "FR.PA", "UBI.PA", "FR.PA", "VIRP.PA", "VK.PA",
    "VLA.PA", "VRLA.PA", "VCT.PA", "VIRP.PA", "VIRI.PA", "VIV.PA", "VU.PA", "WLN.PA"
]

# DAX 40 (Allemagne - Francfort) - Suffixe .DE
DAX40_TICKERS = [
    "1COV.DE", "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", 
    "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE", "DB1.DE", "DBK.DE", "DHL.DE", 
    "DTE.DE", "DTG.DE", "ENR.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE", 
    "HEN3.DE", "HNR1.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", 
    "P911.DE", "PAH3.DE", "PUM.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE", 
    "SHL.DE", "SIE.DE", "SRT3.DE", "SY1.DE", "VOW3.DE", "ZAL.DE",
]

# MDAX (Allemagne - Mid Cap) - Suffixe .DE
MDAX_TICKERS = [
    "AAD.DE", "ADJ.DE", "AFX.DE", "AIXA.DE", "AT1.DE",
    "BC8.DE", "BOSS.DE", "CEC.DE", "DEQ.DE", "DIC.DE",
    "DUE.DE", "DWNI.DE", "EVD.DE", "EVK.DE", "EVT.DE", "FIE.DE", "FPE3.DE",
    "G1A.DE", "GBF.DE", "GFT.DE", "GIL.DE", "GLJ.DE", "GMM.DE", "HAG.DE",
    "HBH.DE", "HDD.DE", "HLE.DE", "HNL.DE", "HOT.DE", "JEN.DE", "JUN3.DE",
    "KCO.DE", "KGX.DE", "KRN.DE", "KWS.DE", "LEG.DE", "LHA.DE",
    "LXS.DE", "NDX1.DE", "NEM.DE", "NOEJ.DE",
    "PBB.DE", "PSM.DE", "RAA.DE", "RRTL.DE", "S92.DE", "SAX.DE",
    "SBS.DE", "SDF.DE", "SFQ.DE", "SHA0.DE", "SKB.DE",
    "SZG.DE", "SZU.DE", "TEG.DE", "TKA.DE", "TLX.DE",
    "TMV.DE", "8TRA.DE", "TTK.DE", "UN0.DE", "UTDI.DE", "VBK.DE",
    "VIB3.DE", "VNA.DE", "WAC.DE", "WCH.DE"
]

# FTSE 100 (Royaume-Uni - Londres) - Suffixe .L
FTSE100_TICKERS = [
    "AAF.L", "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L", "AUTO.L", "AV.L",
    "AZN.L", "BA.L", "BARC.L", "BATS.L", "BEZ.L", "BKG.L", "BME.L",
    "BNZL.L", "BP.L", "BRBY.L", "BT-A.L", "CCH.L", "CNA.L", "CPG.L", "CRDA.L",
    "CRH.L", "CTEC.L", "DCC.L", "DGE.L", "DPLM.L", "EDV.L", "ENT.L", "EXPN.L",
    "EZJ.L", "FCIT.L", "FLTR.L", "FRAS.L", "FRES.L", "GLEN.L", "GSK.L", "HIK.L",
    "HLMA.L", "HLN.L", "HSBA.L", "HSX.L", "HWDN.L", "IAG.L", "ICG.L", "IHG.L",
    "III.L", "IMB.L", "IMI.L", "INF.L", "ITRK.L", "ITV.L", "JD.L", "KGF.L",
    "LAND.L", "LGEN.L", "LLOY.L", "LMP.L", "LSEG.L", "MKS.L", "MNDI.L", "MNG.L",
    "MRO.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L", "PHNX.L", "PRU.L", "PSH.L",
    "PSN.L", "PSON.L", "REL.L", "RIO.L", "RKT.L", "RMV.L", "RR.L", "RS1.L",
    "RTO.L", "SBRY.L", "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L",
    "SMIN.L", "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L", "STJ.L", "SVT.L",
    "TSCO.L", "TW.L", "ULVR.L", "UTG.L", "UU.L", "VOD.L", "WEIR.L", "WPP.L", "WTB.L"
]

# IBEX 35 (Espagne - Madrid) - Suffixe .MC
IBEX35_TICKERS = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC", "BBVA.MC", 
    "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC", "ENG.MC", "FDR.MC", 
    "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC", "ITX.MC", "LOG.MC", 
    "MAP.MC", "MEL.MC", "MRL.MC", "MTS.MC", "NTGY.MC", "PHM.MC", "RED.MC", 
    "REP.MC", "ROVI.MC", "SAB.MC", "SAN.MC", "SCYR.MC", "SOL.MC", "TEF.MC", 
    "UNI.MC",
]

# FTSE MIB (Italie - Milan) - Suffixe .MI
FTSEMIB_TICKERS = [
    "A2A.MI", "AMP.MI", "AZM.MI", "BAMI.MI", "BGN.MI", "BMED.MI", "BPE.MI", 
    "BZU.MI", "CPR.MI", "DIA.MI", "ENEL.MI", "ENI.MI", "ERG.MI", "FBK.MI", 
    "G.MI", "HER.MI", "IG.MI", "INW.MI", "IP.MI", "ISP.MI", "IVG.MI", "LDO.MI", 
    "MB.MI", "MONC.MI", "NEXI.MI", "PIRC.MI", "PRY.MI", "PST.MI", "RACE.MI", 
    "REC.MI", "SRG.MI", "STLAM.MI", "STMMI.MI", "TEN.MI", "TIT.MI", "TRN.MI", 
    "UCG.MI", "UNI.MI",
]

# SMI (Suisse - Zurich) - Suffixe .SW
SMI_TICKERS = [
    "ABBN.SW", "ADEN.SW", "ALC.SW", "GEBN.SW", "GIVN.SW", "HOLN.SW", 
    "LONN.SW", "NESN.SW", "NOVN.SW", "PGHN.SW", "ROG.SW", "SCMN.SW", 
    "SGSN.SW", "SIKA.SW", "SLHN.SW", "SOON.SW", "SREN.SW", "UBSG.SW", "ZURN.SW",
]

# AEX (Pays-Bas - Amsterdam) - Suffixe .AS
AEX_TICKERS = [
    "ABN.AS", "ADYEN.AS", "AGN.AS", "AKZA.AS", "ASML.AS", "ASRNL.AS", 
    "HEIA.AS", "INGA.AS", "KPN.AS", "MT.AS", "NN.AS", "PHIA.AS", "PRX.AS", 
    "RAND.AS", "REN.AS", "SHELL.AS", "UNA.AS", "WKL.AS","SBMO.AS", "ALFEN.AS", "APAM.AS"
]

# BEL 20 (Belgique - Bruxelles) - Suffixe .BR
BEL20_TICKERS = [
    "ABI.BR", "ACKB.BR", "AED.BR", "AGS.BR", "ARGX.BR", "AZE.BR", "COFB.BR", "COLR.BR", 
    "DIE.BR", "ELI.BR", "GBLB.BR", "KBC.BR", "LOTB.BR", "MELE.BR", "MONT.BR", "PROX.BR", "SOF.BR", "SOLB.BR", 
    "UCB.BR", "UMI.BR", "VGP.BR", "WDP.BR", "SYENS.BR",
]

# OMX Stockholm 30 (Suède - Stockholm) - Suffixe .ST
OMX30_TICKERS = [
    "ABB.ST", "ADDT-B.ST", "ALFA.ST", "ASSA-B.ST", "ATCO-A.ST", "ATCO-B.ST", "AZN.ST",
    "BOL.ST", "ELUX-B.ST", "EPI-A.ST", "EQT.ST", "ERIC-B.ST", "ESSITY-B.ST", "EVO.ST", "GETI-B.ST",
    "HEXA-B.ST", "HM-B.ST", "INDU-C.ST", "INVE-B.ST", "KINV-B.ST", "LIFCO-B.ST", "NIBE-B.ST", "NDA-SE.ST",
    "SAAB-B.ST", "SAND.ST", "SCA-B.ST", "SEB-A.ST", "SHB-A.ST", "SKA-B.ST", "SKF-B.ST", "SSAB-A.ST",
    "SWED-A.ST", "TEL2-B.ST", "TELIA.ST", "VOLV-B.ST"
]

# OMX Copenhagen 25 (Danemark - Copenhague) - Suffixe .CO
OMXC25_TICKERS = [
    "AMBU-B.CO", "BAVA.CO", "CARL-B.CO", "COLO-B.CO", "DANSKE.CO", "DEMANT.CO",
    "DSV.CO", "FLS.CO", "GMAB.CO", "GN.CO", "ISS.CO", "JYSK.CO", "MAERSK-A.CO", "MAERSK-B.CO",
    "NKT.CO", "NOVO-B.CO", "NSIS-B.CO", "ORSTED.CO", "PNDORA.CO", "RBREW.CO", "ROCK-B.CO", "TRYG.CO",
    "VWS.CO", "ZEAL.CO"
]

# OMX Helsinki 25 (Finlande - Helsinki) - Suffixe .HE
OMXH25_TICKERS = [
    "HIAB.HE", "ELISA.HE", "FORTUM.HE", "HUH1V.HE", "KCR.HE",
    "KESKOB.HE", "KNEBV.HE", "METSO.HE", "NESTE.HE", "NOKIA.HE",
    "ORNBV.HE", "OUT1V.HE", "SAMPO.HE", "STERV.HE",
    "TIETO.HE", "UPM.HE", "WRT1V.HE"
]

# OBX (Norvège - Oslo) - Suffixe .OL
OBX_TICKERS = [
    "AKRBP.OL", "AKRBP.OL", "DNB.OL", "EQNR.OL", "FROO.OL",
    "GJF.OL", "MOWI.OL", "NHY.OL", "ORK.OL", "SALM.OL",
    "TEL.OL", "TGS.OL", "TOM.OL", "YAR.OL", "HAV.OL"
]

# PSI 20 (Portugal - Lisbonne) - Suffixe .LS
PSI20_TICKERS = [
    "ALTR.LS", "BCP.LS", "COR.LS", "CTT.LS", "EDP.LS",
    "EDPR.LS", "GALP.LS", "JMT.LS", "NOS.LS", "PHR.LS",
    "RENE.LS", "SEM.LS", "SON.LS"
]

# ATX (Autriche - Vienne) - Suffixe .VI
ATX_TICKERS = [
    "ANDR.VI", "BG.VI", "CAI.VI", "EBS.VI", "EVN.VI",
    "LNZ.VI", "OMV.VI", "POST.VI", "RBI.VI", "SBO.VI",
    "VOE.VI", "VIG.VI"
]

# ISEQ 20 (Irlande - Dublin) - Suffixe .IR
ISEQ20_TICKERS = [
    "A5G.IR", "BIRG.IR", "EG7.IR", "GL9.IR",
    "GVR.IR", "GRP.IR", "IR5B.IR", "IRES.IR", "KMR.IR",
    "KRZ.IR", "KRX.IR", "MLC.IR", "OIZ.IR", "PTSB.IR", "RYA.IR", "UPR.IR"
]

# WIG20 (Pologne - Varsovie) - Suffixe .WA
WIG20_TICKERS = [
    "ALR.WA", "CDR.WA", "CPS.WA", "DNP.WA", "JSW.WA",
    "KGH.WA", "LPP.WA", "MBK.WA", "PEO.WA", "PGE.WA",
    "PGN.WA", "PKN.WA", "PKO.WA", "PZU.WA", "SPL.WA"
]

STOOQ_SUFFIX = {
    '.PA': '.pa',
    '.DE': '.de',
    '.MI': '.mi',
    '.MC': '.mc',
    '.L':  '.uk',   # ici c’est le point critique
    '.SW': '.sw',
    '.AS': '.as',
    '.BR': '.br',
}

COLUMN_MAP = {
    'Data': 'Date',
    'Otwarcie': 'Open',
    'Najwyzszy': 'High',
    'Najnizszy': 'Low',
    'Zamkniecie': 'Close',
    'Wolumen': 'Volume'
}

# ============================================================================
# DICTIONNAIRE DES MARCHÉS
# ============================================================================
MARKETS = {
    # US - Dynamique via API
 'US': {
        'name': 'US Top 1000 (plus grandes capitalisations)',
        'tickers': US_TOP_1000,
        'currency': '$',
        'suffix': '',
        'dynamic': False,  # ← Pas besoin d'API ne marche pas de toute façon, limite de requettes atteinte
    },
    # France
    'FR': {
        'name': 'SBF 120 (France)',
        'tickers': SBF120_TICKERS,
        'currency': '€',
        'suffix': '.PA',
        'dynamic': False,
    },
    'CAC40': {
        'name': 'CAC 40 (France)',
        'tickers': CAC40_TICKERS,
        'currency': '€',
        'suffix': '.PA',
        'dynamic': False,
    },
    # Allemagne
    'DE': {
        'name': 'DAX 40 (Allemagne)',
        'tickers': DAX40_TICKERS,
        'currency': '€',
        'suffix': '.DE',
        'dynamic': False,
    },
    'MDAX': {
        'name': 'MDAX (Allemagne)',
        'tickers': MDAX_TICKERS,
        'currency': '€',
        'suffix': '.DE',
        'dynamic': False,
    },
    # UK
    'UK': {
        'name': 'FTSE 100 (UK)',
        'tickers': FTSE100_TICKERS,
        'currency': '£',
        'suffix': '.L',
        'dynamic': False,
    },
    # Espagne
    'ES': {
        'name': 'IBEX 35 (Espagne)',
        'tickers': IBEX35_TICKERS,
        'currency': '€',
        'suffix': '.MC',
        'dynamic': False,
    },
    # Italie
    'IT': {
        'name': 'FTSE MIB (Italie)',
        'tickers': FTSEMIB_TICKERS,
        'currency': '€',
        'suffix': '.MI',
        'dynamic': False,
    },
    # Suisse
    'CH': {
        'name': 'SMI (Suisse)',
        'tickers': SMI_TICKERS,
        'currency': 'CHF',
        'suffix': '.SW',
        'dynamic': False,
    },
    # Pays-Bas
    'NL': {
        'name': 'AEX (Pays-Bas)',
        'tickers': AEX_TICKERS,
        'currency': '€',
        'suffix': '.AS',
        'dynamic': False,
    },
    # Belgique
    'BE': {
        'name': 'BEL 20 (Belgique)',
        'tickers': BEL20_TICKERS,
        'currency': '€',
        'suffix': '.BR',
        'dynamic': False,
    },
    # Suède
    'SE': {
        'name': 'OMX Stockholm 30 (Suède)',
        'tickers': OMX30_TICKERS,
        'currency': 'SEK',
        'suffix': '.ST',
        'dynamic': False,
    },
    # Danemark
    'DK': {
        'name': 'OMX Copenhagen 25 (Danemark)',
        'tickers': OMXC25_TICKERS,
        'currency': 'DKK',
        'suffix': '.CO',
        'dynamic': False,
    },
    # Finlande
    'FI': {
        'name': 'OMX Helsinki 25 (Finlande)',
        'tickers': OMXH25_TICKERS,
        'currency': '€',
        'suffix': '.HE',
        'dynamic': False,
    },
    # Norvège
    'NO': {
        'name': 'OBX (Norvège)',
        'tickers': OBX_TICKERS,
        'currency': 'NOK',
        'suffix': '.OL',
        'dynamic': False,
    },
    # Portugal
    'PT': {
        'name': 'PSI 20 (Portugal)',
        'tickers': PSI20_TICKERS,
        'currency': '€',
        'suffix': '.LS',
        'dynamic': False,
    },
    # Autriche
    'AT': {
        'name': 'ATX (Autriche)',
        'tickers': ATX_TICKERS,
        'currency': '€',
        'suffix': '.VI',
        'dynamic': False,
    },
    # Irlande
    'IE': {
        'name': 'ISEQ 20 (Irlande)',
        'tickers': ISEQ20_TICKERS,
        'currency': '€',
        'suffix': '.IR',
        'dynamic': False,
    },
    # Pologne, do not use since I cannot buy!
    'PL': {
        'name': 'WIG20 (Pologne)',
        'tickers': WIG20_TICKERS,
        'currency': 'PLN',
        'suffix': '.WA',
        'dynamic': False,
    },
    # EUR = Tous les marchés européens combinés
    'EUR': {
        'name': 'Europe (All)',
        'tickers': (SBF120_TICKERS + DAX40_TICKERS + FTSE100_TICKERS + 
                   IBEX35_TICKERS + FTSEMIB_TICKERS + SMI_TICKERS + 
                   AEX_TICKERS + BEL20_TICKERS + MDAX_TICKERS + 
                   OMX30_TICKERS + OMXC25_TICKERS + OMXH25_TICKERS + 
                   OBX_TICKERS + PSI20_TICKERS + ATX_TICKERS +
                   ISEQ20_TICKERS),
        'currency': '€',
        'suffix': '',
        'dynamic': False,
    },
}

# ============================================================================
# FONCTION POUR RÉCUPÉRER LES TICKERS
# ============================================================================
def get_tickers(market="US", api_key=FMP_API_KEY):
    """
    Récupère la liste des tickers selon le marché
    
    Args:
        market: Code du marché (US, EUR, FR, DE, UK, ES, IT, CH, NL, BE, CAC40, MDAX)
        api_key: Clé API pour Financial Modeling Prep (utilisée pour US)
    
    Returns:
        tuple: (liste des tickers, nom de l'indice, devise)
    """
    market = market.upper()
    
    if market not in MARKETS:
        available = ', '.join(MARKETS.keys())
        raise ValueError(f"Marché '{market}' non reconnu. Disponibles: {available}")
    
    market_info = MARKETS[market]
    name = market_info['name']
    currency = market_info['currency']
    
    # Récupération dynamique pour le marché US
    if market_info.get('dynamic', False):
        print(f"\n📊 Récupération dynamique des tickers US (cap > $1B)...")
        tickers = get_us_tickers_with_fallback(api_key)
    else:
        tickers = market_info['tickers']
    
    # Supprimer les doublons
    tickers = list(dict.fromkeys(tickers))
    
    print(f"   ✅ {name}: {len(tickers)} tickers")
    
    return tickers, name, currency


def list_available_markets():
    """Affiche les marchés disponibles"""
    print("\n📊 MARCHÉS DISPONIBLES:")
    print("=" * 60)
    for code, info in MARKETS.items():
        dynamic_flag = " [API]" if info.get('dynamic', False) else ""
        ticker_count = "~1500-2000" if info.get('dynamic', False) else str(len(info['tickers']))
        print(f"   {code:8} → {info['name']:30} ({ticker_count:>10} tickers) {info['currency']}{dynamic_flag}")
    print("=" * 60)


def download_stock_data(ticker, market_code, isin=None, config=CONFIG):
    """
    Route automatiquement vers Yahoo (US), Stooq (Europe), puis Euronext (retardé)
    Renvoie weekly OHLC
    """
    errors = []

    # 1️⃣ Essayer Yahoo

    if market == "US":
        #print(f"downloading US market")
        df, err = download_stock_data_yahoo(ticker, config)
        if df is not None:
            return df, "yahoo"
        errors.append(f"Yahoo: {err}")
    else:
        #print(f"downloading EUR market")
        # 1️⃣ Essayer Yahoo
        df, err = download_stock_data_yahoo(ticker, config)
        if df is not None:
            return df, "yahoo"
        errors.append(f"Yahoo: {err}")
        print(f"yahoo download failed")
        
        # 2️⃣ Essayer Stooq (Europe)
        try:
            df, err = download_stock_data_stooq(ticker, config)
            if df is not None:
                return df, "stooq"
            errors.append(f"Stooq: {err}")
            print(f"stooq download failed")
        except Exception as e:
            errors.append(f"Stooq Exception: {str(e)}")

        # 3️⃣ Essayer Euronext (retardé)
        if isin is not None:
            df, err = download_stock_data_euronext(
                ticker=ticker,
                isin=isin,
                years=config.get("years", 5),
                min_bars=config.get("min_bars", 50)
            )
            if df is not None:
                return df, "euronext"
            errors.append(f"Euronext: {err}")
            print(f"euronext download failed")
        else:
            errors.append("Euronext: ISIN manquant")

    # Si tout échoue
    return None, " | ".join(errors)
    
    
    # =============================
# CONFIG
# =============================

CACHE_DIR = "data/cache"
DAILY_CACHE = os.path.join(CACHE_DIR, "euronext_daily")
WEEKLY_CACHE = os.path.join(CACHE_DIR, "euronext_weekly")

os.makedirs(DAILY_CACHE, exist_ok=True)
os.makedirs(WEEKLY_CACHE, exist_ok=True)

REQUEST_SLEEP = 0.8     # important pour 800 tickers
MIN_BARS_WEEKLY = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =============================
# EURONEXT DAILY DOWNLOAD
# =============================

def download_euronext_daily(isin, years=5):
    """
    Télécharge les données DAILY retardées Euronext via ISIN
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years)

    url = (
        "https://live.euronext.com/en/ajax/getHistoricalPrice"
        f"?isin={isin}"
        f"&fromDate={start.strftime('%Y-%m-%d')}"
        f"&toDate={end.strftime('%Y-%m-%d')}"
        "&period=DAILY"
    )

    r = requests.get(url, headers=HEADERS, timeout=20)

    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"

    payload = r.json()

    if "data" not in payload or len(payload["data"]) == 0:
        return None, "Pas de données Euronext"

    df = pd.DataFrame(payload["data"])

    # Normalisation
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df.reset_index(drop=True), None

# =============================
# DAILY → WEEKLY
# =============================

def daily_to_weekly(df_daily):
    df = df_daily.copy()
    df = df.set_index("Date")

    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return weekly.reset_index()

# =============================
# CACHE HELPERS
# =============================

def cache_path(base_dir, key):
    return os.path.join(base_dir, f"{key}.parquet")

def load_cache(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def save_cache(df, path):
    df.to_parquet(path, index=False)

# =============================
# MAIN ENTRY POINT
# =============================

def download_stock_data_euronext(
    ticker,
    isin,
    years=5,
    min_bars=MIN_BARS_WEEKLY
):
    """
    Source Euronext retardée (fallback)
    Retourne WEEKLY
    """

    weekly_cache = cache_path(WEEKLY_CACHE, ticker)

    cached_weekly = load_cache(weekly_cache)
    if cached_weekly is not None and len(cached_weekly) >= min_bars:
        return cached_weekly, "cache"

    # ---- DAILY ----
    daily_cache = cache_path(DAILY_CACHE, isin)

    df_daily = load_cache(daily_cache)

    if df_daily is None:
        df_daily, err = download_euronext_daily(isin, years)
        if df_daily is None:
            return None, err

        save_cache(df_daily, daily_cache)
        time.sleep(REQUEST_SLEEP)

    # ---- WEEKLY ----
    df_weekly = daily_to_weekly(df_daily)

    if len(df_weekly) < min_bars:
        return None, f"Données insuffisantes: {len(df_weekly)} semaines"

    save_cache(df_weekly, weekly_cache)

    return df_weekly, None
    
    
    

# ============================================================================
# RÉCUPÉRATION DES DONNÉES
# ============================================================================

def download_stock_data_yahoo(ticker, config=CONFIG):
    """
    Télécharge les données historiques d'une action EN WEEKLY
    """
    try:
        # *** TÉLÉCHARGEMENT EN WEEKLY ***
        data = yf.download(
            ticker, 
            period=config['period'],      # 5 ans
            interval=config['interval'],  # '1wk' = weekly
            progress=False, 
            auto_adjust=True
        )
        
        if data is None or len(data) == 0:
            return None, "Pas de données"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Colonnes manquantes: {missing_cols}"
        
        min_bars = config.get('min_bars', 50)
        if len(data) < min_bars:
            return None, f"Données insuffisantes: {len(data)} semaines (min: {min_bars})"
        
        data = data.reset_index()
        
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna(subset=required_cols)
        
        if len(data) < min_bars:
            return None, f"Données insuffisantes après nettoyage: {len(data)} semaines"
        
        return data, None
        
    except Exception as e:
        return None, str(e)
        

def yahoo_to_stooq(ticker):
    for suffix, stooq_suf in STOOQ_SUFFIX.items():
        if ticker.endswith(suffix):
            return ticker[:-len(suffix)] + stooq_suf
    return ticker.lower()        

def download_stock_data_stooq(ticker, config=CONFIG):
    """
    Télécharge les données via STOOQ (daily → weekly)
    Ticker en entrée = format Yahoo (MAJUSCULE)
    """
    try:
        stooq_ticker = yahoo_to_stooq(ticker)

        url = f"https://stooq.pl/q/d/l/?s={stooq_ticker}&i=d"
        r = requests.get(url, timeout=10)

        if r.status_code != 200 or not r.text.strip():
            return None, "Pas de données"

        data = pd.read_csv(StringIO(r.text))
        if data.empty:
            return None, "Pas de données"

        data.rename(columns=COLUMN_MAP, inplace=True)
        required = ['Date', 'Open', 'High', 'Low', 'Close']
        if any(c not in data.columns for c in required):
            return None, "Colonnes manquantes"
        
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date']).set_index('Date').sort_index()

        for c in ['Open', 'High', 'Low', 'Close']:
            data[c] = pd.to_numeric(data[c], errors='coerce')

        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Filtrage période
        years = int(config['period'].replace('y', ''))
        start = pd.Timestamp.today() - pd.DateOffset(years=years)
        data = data[data.index >= start]

        # Weekly
        weekly = data.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum' if 'Volume' in data.columns else 'first'
        }).dropna(subset=['Open', 'High', 'Low', 'Close'])

        min_bars = config.get('min_bars', 50)
        if len(weekly) < min_bars:
            return None, f"Données insuffisantes: {len(weekly)} semaines"

        return weekly.reset_index(), None

    except Exception as e:
        return None, str(e)

# ============================================================================
# DÉTECTION DES POINTS PIVOTS
# ============================================================================
def find_pivot_points(data, order=3):
    """Trouve les points pivots (hauts et bas locaux)"""
    highs = data['High'].values.astype(float)
    lows = data['Low'].values.astype(float)
    
    high_indices = argrelextrema(highs, np.greater_equal, order=order)[0]
    low_indices = argrelextrema(lows, np.less_equal, order=order)[0]
    
    return high_indices, low_indices

def refine_pivot_points(data, indices, pivot_type='low', window=2):
    """Raffine les points pivots en éliminant les doublons très proches"""
    if len(indices) == 0:
        return indices
    
    refined = [indices[0]]
    price_col = 'Low' if pivot_type == 'low' else 'High'
    
    for i in range(1, len(indices)):
        if indices[i] - refined[-1] > window:
            refined.append(indices[i])
        else:
            if pivot_type == 'low':
                if float(data[price_col].iloc[indices[i]]) < float(data[price_col].iloc[refined[-1]]):
                    refined[-1] = indices[i]
            else:
                if float(data[price_col].iloc[indices[i]]) > float(data[price_col].iloc[refined[-1]]):
                    refined[-1] = indices[i]
    
    return np.array(refined)

# ============================================================================
# ALTERNANCE STRICTE AVEC SÉLECTION PAR RAPPORT AU CANAL
# ============================================================================
def alternate_pivots_simple(data, low_indices, high_indices):
    """Alternance simple (sans canal) - utilisée pour le canal préliminaire"""
    if len(low_indices) == 0 or len(high_indices) == 0:
        return np.array([]), np.array([])
    
    pivots = []
    
    for idx in low_indices:
        pivots.append({
            'index': int(idx),
            'type': 'low',
            'price': float(data['Low'].iloc[idx])
        })
    
    for idx in high_indices:
        pivots.append({
            'index': int(idx),
            'type': 'high',
            'price': float(data['High'].iloc[idx])
        })
    
    pivots.sort(key=lambda x: x['index'])
    
    filtered = []
    i = 0
    
    while i < len(pivots):
        current_type = pivots[i]['type']
        group = [pivots[i]]
        j = i + 1
        
        while j < len(pivots) and pivots[j]['type'] == current_type:
            group.append(pivots[j])
            j += 1
        
        if current_type == 'low':
            best = min(group, key=lambda x: x['price'])
        else:
            best = max(group, key=lambda x: x['price'])
        
        filtered.append(best)
        i = j
    
    final_lows = np.array([p['index'] for p in filtered if p['type'] == 'low'])
    final_highs = np.array([p['index'] for p in filtered if p['type'] == 'high'])
    
    return final_lows, final_highs

def alternate_pivots_with_channel(data, low_indices, high_indices, slope, intercept, channel_height):
    """
    Filtre les pivots avec alternance stricte EN FONCTION DU CANAL
    
    - Creux consécutifs: garde celui qui ENFONCE LE PLUS le support
    - Sommets consécutifs: garde celui qui DÉPASSE LE PLUS la résistance
    """
    if len(low_indices) == 0 or len(high_indices) == 0:
        return np.array([]), np.array([]), []
    
    def support_at(idx):
        return slope * idx + intercept
    
    def resistance_at(idx):
        return slope * idx + intercept + channel_height
    
    pivots = []
    
    for idx in low_indices:
        idx = int(idx)
        price = float(data['Low'].iloc[idx])
        support = support_at(idx)
        distance_to_support = price - support
        pivots.append({
            'index': idx,
            'type': 'low',
            'price': price,
            'line_value': support,
            'distance': distance_to_support
        })
    
    for idx in high_indices:
        idx = int(idx)
        price = float(data['High'].iloc[idx])
        resistance = resistance_at(idx)
        distance_to_resistance = price - resistance
        pivots.append({
            'index': idx,
            'type': 'high',
            'price': price,
            'line_value': resistance,
            'distance': distance_to_resistance
        })
    
    pivots.sort(key=lambda x: x['index'])
    
    filtered = []
    i = 0
    
    while i < len(pivots):
        current_type = pivots[i]['type']
        group = [pivots[i]]
        j = i + 1
        
        while j < len(pivots) and pivots[j]['type'] == current_type:
            group.append(pivots[j])
            j += 1
        
        if current_type == 'low':
            best = min(group, key=lambda x: x['distance'])
        else:
            best = max(group, key=lambda x: x['distance'])
        
        filtered.append(best)
        i = j
    
    final_lows = np.array([p['index'] for p in filtered if p['type'] == 'low'])
    final_highs = np.array([p['index'] for p in filtered if p['type'] == 'high'])
    
    return final_lows, final_highs, filtered

def verify_alternation(low_indices, high_indices):
    """Vérifie que l'alternance est bien respectée"""
    if len(low_indices) == 0 or len(high_indices) == 0:
        return False
    
    all_pivots = [(idx, 'L') for idx in low_indices] + [(idx, 'H') for idx in high_indices]
    all_pivots.sort(key=lambda x: x[0])
    
    for i in range(1, len(all_pivots)):
        if all_pivots[i][1] == all_pivots[i-1][1]:
            return False
    
    return True

# ============================================================================
# DÉTECTION DU CANAL HAUSSIER
# ============================================================================
def calculate_channel_fit(data, low_indices, high_indices, slope, intercept, tolerance):
    """Calcule la qualité d'ajustement d'un canal"""
    n = len(data)
    
    support_line = slope * np.arange(n) + intercept
    
    heights = []
    for hi in high_indices:
        expected_support = support_line[hi]
        actual_high = float(data['High'].iloc[hi])
        height = actual_high - expected_support
        if height > 0:
            heights.append(height)
    
    if len(heights) == 0:
        return None
    
    channel_height = np.median(heights)
    
    if channel_height <= 0:
        return None
    
    tolerance_abs = channel_height * tolerance
    resistance_line = support_line + channel_height
    
    valid_lows = []
    for li in low_indices:
        li = int(li)
        expected_support = support_line[li]
        actual_low = float(data['Low'].iloc[li])
        distance = actual_low - expected_support
        
        if -tolerance_abs <= distance <= tolerance_abs:
            valid_lows.append({
                'index': li,
                'price': actual_low,
                'expected': expected_support,
                'distance': distance,
                'distance_pct': (distance / channel_height) * 100
            })
    
    valid_highs = []
    for hi in high_indices:
        hi = int(hi)
        expected_resistance = resistance_line[hi]
        actual_high = float(data['High'].iloc[hi])
        distance = actual_high - expected_resistance
        
        if -tolerance_abs <= distance <= tolerance_abs:
            valid_highs.append({
                'index': hi,
                'price': actual_high,
                'expected': expected_resistance,
                'distance': distance,
                'distance_pct': (distance / channel_height) * 100
            })
    
    return {
        'support_line': support_line,
        'resistance_line': resistance_line,
        'channel_height': channel_height,
        'valid_lows': valid_lows,
        'valid_highs': valid_highs,
        'lows_count': len(valid_lows),
        'highs_count': len(valid_highs)
    }

def detect_ascending_channel(data, config=CONFIG, debug=False):
    """Détecte un canal haussier avec sélection des pivots par rapport au canal"""
    min_lows = config['min_lows']
    min_highs = config['min_highs']
    tolerance = config['tolerance']
    pivot_order = config['pivot_order']
    min_slope = config['min_slope']
    min_r_squared = config['min_r_squared']
    
    high_indices, low_indices = find_pivot_points(data, order=pivot_order)
    
    if debug:
        print(f"   Pivots bruts: {len(low_indices)} creux, {len(high_indices)} sommets")
    
    low_indices = refine_pivot_points(data, low_indices, 'low', window=pivot_order)
    high_indices = refine_pivot_points(data, high_indices, 'high', window=pivot_order)
    
    if debug:
        print(f"   Pivots raffinés: {len(low_indices)} creux, {len(high_indices)} sommets")
    
    if len(low_indices) < 2 or len(high_indices) < 2:
        if debug:
            print(f"   ❌ Pas assez de pivots")
        return None
    
    low_indices_simple, high_indices_simple = alternate_pivots_simple(data, low_indices, high_indices)
    
    if debug:
        print(f"   Pivots alternance simple: {len(low_indices_simple)} creux, {len(high_indices_simple)} sommets")
    
    if len(low_indices_simple) < 2:
        if debug:
            print(f"   ❌ Pas assez de creux pour régression")
        return None
    
    low_x = low_indices_simple.astype(float)
    low_y = data['Low'].iloc[low_indices_simple].values.astype(float)
    
    slope, intercept, r_value, _, _ = linregress(low_x, low_y)
    
    if slope < min_slope:
        if debug:
            print(f"   ❌ Pente trop faible: {slope:.6f} < {min_slope}")
        return None
    
    n = len(data)
    support_line = slope * np.arange(n) + intercept
    
    heights = []
    for hi in high_indices_simple:
        expected_support = support_line[hi]
        actual_high = float(data['High'].iloc[hi])
        height = actual_high - expected_support
        if height > 0:
            heights.append(height)
    
    if len(heights) == 0:
        if debug:
            print(f"   ❌ Impossible de calculer la hauteur du canal")
        return None
    
    channel_height = np.median(heights)
    
    if debug:
        print(f"   Canal préliminaire: pente={slope:.6f}, hauteur={channel_height:.2f}")
    
    low_indices_final, high_indices_final, filtered_pivots = alternate_pivots_with_channel(
        data, low_indices, high_indices, slope, intercept, channel_height
    )
    
    if debug:
        print(f"   Pivots après filtrage canal: {len(low_indices_final)} creux, {len(high_indices_final)} sommets")
        is_alt = verify_alternation(low_indices_final, high_indices_final)
        print(f"   Alternance vérifiée: {'✅' if is_alt else '❌'}")
        
        print(f"   Détail des pivots sélectionnés:")
        for p in filtered_pivots:
            symbol = "▼" if p['type'] == 'low' else "▲"
            line_name = "support" if p['type'] == 'low' else "résistance"
            print(f"     {symbol} semaine={p['index']:3d} prix=${p['price']:.2f} {line_name}=${p['line_value']:.2f} "
                  f"écart={p['distance']:+.2f}")
    
    if len(low_indices_final) < 2 or len(high_indices_final) < 2:
        if debug:
            print(f"   ❌ Pas assez de pivots après filtrage")
        return None
    
    low_x_final = low_indices_final.astype(float)
    low_y_final = data['Low'].iloc[low_indices_final].values.astype(float)
    
    slope_final, intercept_final, r_value_final, _, _ = linregress(low_x_final, low_y_final)
    r_squared_final = r_value_final ** 2
    
    if debug:
        print(f"   Canal final: pente={slope_final:.6f}, R²={r_squared_final:.3f}")
    
    if slope_final < min_slope or r_squared_final < min_r_squared:
        if debug:
            print(f"   ❌ Canal final ne respecte pas les critères")
        return None
    
    channel = calculate_channel_fit(data, low_indices_final, high_indices_final, 
                                    slope_final, intercept_final, tolerance)
    
    if channel is None:
        if debug:
            print(f"   ❌ Impossible de calculer l'ajustement du canal")
        return None
    
    if debug:
        print(f"   Pivots valides dans tolérance: {channel['lows_count']} creux, {channel['highs_count']} sommets")
    
    if channel['lows_count'] >= min_lows and channel['highs_count'] >= min_highs:
        return {
            **channel,
            'slope': slope_final,
            'intercept': intercept_final,
            'r_squared': r_squared_final,
            'low_indices': low_indices_final,
            'high_indices': high_indices_final,
            'all_pivots': filtered_pivots
        }
    
    if debug:
        print(f"   ❌ Pas assez de pivots valides: {channel['lows_count']}L/{channel['highs_count']}H "
              f"(requis: {min_lows}L/{min_highs}H)")
    
    return None

# ============================================================================
# VÉRIFICATION POST-ATH / RECOVERY
# ============================================================================
def check_market_position(data, channel, config=CONFIG):
    """Vérifie si l'action est en position Post-ATH, Recovery ou Canal Actif"""
    if channel is None:
        return False, None
    
    current_price = float(data['Close'].iloc[-1])
    ath = float(data['High'].max())
    
    start_idx = int(len(data) * 0.2)
    min_price_recent = float(data['Low'].iloc[start_idx:].min())
    
    if current_price >= ath * config['ath_threshold']:
        return True, "Post-ATH"
    
    if min_price_recent <= ath * config['recovery_drop']:
        if current_price >= ath * config['recovery_threshold']:
            return True, "Recovery"
    
    last_support = channel['support_line'][-1]
    last_resistance = channel['resistance_line'][-1]
    channel_height = last_resistance - last_support
    margin = channel_height * 0.1
    
    if (last_support - margin) <= current_price <= (last_resistance + margin):
        recent_threshold = len(data) * 0.6
        recent_lows = [l for l in channel['valid_lows'] if l['index'] > recent_threshold]
        recent_highs = [h for h in channel['valid_highs'] if h['index'] > recent_threshold]
        
        if len(recent_lows) >= 1 or len(recent_highs) >= 1:
            return True, "Canal Actif"
    
    return False, None

# ============================================================================
# SCANNER PRINCIPAL
# ============================================================================
def scan_channels(market="US", config=CONFIG, max_stocks=None, api_key=FMP_API_KEY):
    """
    Scanner de canaux haussiers
    
    Args:
        market: "US" pour US Stocks > $1B, "EUR" pour marchés européens
        config: Configuration
        max_stocks: Limite le nombre d'actions (None = toutes)
        api_key: Clé API pour Financial Modeling Prep
    """
    print("=" * 70)
    print(f"SCANNER DE CANAUX HAUSSIERS - {market} (WEEKLY)")
    print("=" * 70)
    
    print(f"\n📊 Récupération de la liste...")
    tickers, index_name, currency = get_tickers(market, api_key)
    
    print(f"\n📈 Indice: {index_name}")
    print(f"💱 Devise: {currency}")
    
    print(f"⏱️  Timeframe: WEEKLY (hebdomadaire)")
    print(f"📅  Période: {config['period']}")
    print(f"Paramètres:")
    print(f"  - Creux minimum: {config['min_lows']}")
    print(f"  - Sommets minimum: {config['min_highs']}")
    print(f"  - Tolérance: ±{config['tolerance']*100:.0f}% de la hauteur du canal")
    print(f"  - R² minimum: {config['min_r_squared']}")
    print(f"  - Pivot order: {config['pivot_order']} semaines")
    print(f"  - Sélection: creux→enfonce support, sommet→dépasse résistance")
    print(f"  - Alternance stricte: L ↔ H ↔ L ↔ H...")
    print("=" * 70)
       
    if max_stocks:
        tickers = tickers[:max_stocks]
    
    print(f"   {len(tickers)} actions à analyser")
    
    results = []
    stats = {
        'download_ok': 0,
        'download_fail': 0,
        'no_channel': 0,
        'channel_found': 0,
        'position_valid': 0,
        'position_invalid': 0
    }
    
    print("\n🔍 Analyse en cours (WEEKLY)...\n")
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"   [{i+1}/{len(tickers)}] Téléchargés: {stats['download_ok']} | "
                  f"Canaux: {stats['channel_found']} | Valides: {stats['position_valid']}")
        
        try:
            data, error = download_stock_data(ticker, config)
            
            if data is None:
                stats['download_fail'] += 1
                continue
            
            stats['download_ok'] += 1
            
            channel = detect_ascending_channel(data, config, debug=False)
            
            if channel is None:
                stats['no_channel'] += 1
                continue
            
            stats['channel_found'] += 1
            
            is_valid, position_type = check_market_position(data, channel, config)
            
            if is_valid:
                stats['position_valid'] += 1
                
                current_price = float(data['Close'].iloc[-1])
                ath = float(data['High'].max())
                pct_from_ath = (current_price / ath - 1) * 100
                
                last_support = channel['support_line'][-1]
                last_resistance = channel['resistance_line'][-1]
                channel_position = ((current_price - last_support) / 
                                   (last_resistance - last_support) * 100)
                
                # Pente annualisée (52 semaines par an)
                weekly_slope_pct = channel['slope'] / current_price * 100
                annual_slope_pct = weekly_slope_pct * 52
                
                # Durée du canal en semaines
                first_pivot = min(
                    min([l['index'] for l in channel['valid_lows']]) if channel['valid_lows'] else 999,
                    min([h['index'] for h in channel['valid_highs']]) if channel['valid_highs'] else 999
                )
                canal_duration_weeks = len(data) - first_pivot
                
                result = {
                    'Ticker': ticker,
                    'Position': position_type,
                    'Prix': round(current_price, 2),
                    'Devise': currency,
                    'ATH': round(ath, 2),
                    '% ATH': round(pct_from_ath, 1),
                    'Creux': channel['lows_count'],
                    'Sommets': channel['highs_count'],
                    'R²': round(channel['r_squared'], 3),
                    'Pente %/an': round(annual_slope_pct, 1),
                    'Pos. Canal %': round(channel_position, 1),
                    'Durée (sem)': canal_duration_weeks,
                }
                
                results.append(result)
                print(f"   ✅ {ticker}: {position_type} | {channel['lows_count']}L/{channel['highs_count']}H | "
                      f"R²={channel['r_squared']:.2f} | {pct_from_ath:+.1f}% ATH | {canal_duration_weeks} sem")
            else:
                stats['position_invalid'] += 1
        
        except Exception as e:
            stats['download_fail'] += 1
            continue
    
    print(f"\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"   Téléchargements réussis: {stats['download_ok']}/{len(tickers)}")
    print(f"   Canaux haussiers détectés: {stats['channel_found']}")
    print(f"   Positions valides: {stats['position_valid']}")
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('R²', ascending=False)
        return df_results, index_name
    
    return pd.DataFrame(), index_name

# ============================================================================
# VISUALISATION
# ============================================================================
def get_currency_for_ticker(ticker):
    """Détermine la devise selon le suffixe du ticker"""
    if ticker.endswith('.PA'):
        return '€'
    elif ticker.endswith('.DE'):
        return '€'
    elif ticker.endswith('.L'):
        return '£'
    elif ticker.endswith('.MC'):
        return '€'
    elif ticker.endswith('.MI'):
        return '€'
    elif ticker.endswith('.SW'):
        return 'CHF'
    elif ticker.endswith('.AS'):
        return '€'
    elif ticker.endswith('.BR'):
        return '€'
    elif ticker.endswith('.ST'):
        return 'SEK'
    elif ticker.endswith('.CO'):
        return 'DKK'
    elif ticker.endswith('.HE'):
        return '€'
    elif ticker.endswith('.OL'):
        return 'NOK'
    elif ticker.endswith('.LS'):
        return '€'
    elif ticker.endswith('.VI'):
        return '€'
    elif ticker.endswith('.IR'):
        return '€'
    elif ticker.endswith('.WA'):
        return 'PLN'
    else:
        return '$'

def plot_channel(ticker, config=CONFIG, save_path=None):
    """Visualise le canal haussier WEEKLY"""
    print(f"\n📈 Génération du graphique WEEKLY pour {ticker}...")
    
    data, error = download_stock_data(ticker, config)
    
    if data is None:
        print(f"   ❌ Erreur: {error}")
        return None
    
    channel = detect_ascending_channel(data, config, debug=True)
    
    if channel is None:
        print(f"   ❌ Aucun canal détecté pour {ticker}")
        plot_stock_with_pivots(ticker, data, config)
        return None
    
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    colors = {
        'up': '#26a69a',
        'down': '#ef5350',
        'support': '#4caf50',
        'resistance': '#f44336',
        'channel_fill': '#2196f3',
    }
    
    # Chandeliers
    for i in range(len(data)):
        o = float(data['Open'].iloc[i])
        c = float(data['Close'].iloc[i])
        h = float(data['High'].iloc[i])
        l = float(data['Low'].iloc[i])
        
        color = colors['up'] if c >= o else colors['down']
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.plot([i, i], [o, c], color=color, linewidth=4)  # Plus épais pour weekly
    
    # Canal
    x = np.arange(len(data))
    ax.plot(x, channel['support_line'], color=colors['support'], linewidth=2, label='Support')
    ax.plot(x, channel['resistance_line'], color=colors['resistance'], linewidth=2, label='Résistance')
    ax.fill_between(x, channel['support_line'], channel['resistance_line'], alpha=0.15, color=colors['channel_fill'])
    
    # Zones de tolérance
    tol = channel['channel_height'] * config['tolerance']
    ax.fill_between(x, channel['support_line'] - tol, channel['support_line'] + tol, 
                   alpha=0.1, color=colors['support'], label=f'Zone support ±{config["tolerance"]*100:.0f}%')
    ax.fill_between(x, channel['resistance_line'] - tol, channel['resistance_line'] + tol, 
                   alpha=0.1, color=colors['resistance'], label=f'Zone résistance ±{config["tolerance"]*100:.0f}%')
    
    # Points pivots valides
    for idx, low in enumerate(channel['valid_lows']):
        ax.scatter(low['index'], low['price'], color=colors['support'], s=200, zorder=5, 
                  marker='^', edgecolors='white', linewidths=2)
        label = f"L{idx+1}\n{low['distance_pct']:+.0f}%"
        ax.annotate(label, (low['index'], low['price']), 
                   textcoords="offset points", xytext=(0, -40), ha='center', 
                   color='#4caf50', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e1e', edgecolor='#4caf50', alpha=0.8))
    
    for idx, high in enumerate(channel['valid_highs']):
        ax.scatter(high['index'], high['price'], color=colors['resistance'], s=200, zorder=5,
                  marker='v', edgecolors='white', linewidths=2)
        label = f"H{idx+1}\n{high['distance_pct']:+.0f}%"
        ax.annotate(label, (high['index'], high['price']), 
                   textcoords="offset points", xytext=(0, 30), ha='center', 
                   color='#f44336', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e1e', edgecolor='#f44336', alpha=0.8))
    
    # Lignes reliant les pivots
    all_valid = [(l['index'], l['price'], 'L') for l in channel['valid_lows']] + \
                [(h['index'], h['price'], 'H') for h in channel['valid_highs']]
    all_valid.sort(key=lambda x: x[0])
    
    if len(all_valid) > 1:
        for i in range(len(all_valid) - 1):
            x1, y1, t1 = all_valid[i]
            x2, y2, t2 = all_valid[i + 1]
            ax.plot([x1, x2], [y1, y2], '--', color='white', alpha=0.4, linewidth=1.5)
    
    is_valid, position_type = check_market_position(data, channel, config)
    current_price = float(data['Close'].iloc[-1])
    ath = float(data['High'].max())
    pct_from_ath = (current_price / ath - 1) * 100
    
    # Ligne du prix actuel
    ax.axhline(y=current_price, color='yellow', linestyle=':', alpha=0.7, linewidth=1)
    ax.annotate(f'Prix: ${current_price:.2f}', xy=(len(data)-1, current_price),
               xytext=(5, 0), textcoords='offset points', color='yellow', fontsize=10)
    
    # Dates sur l'axe X
    if 'Date' in data.columns:
        # Afficher quelques dates
        n_ticks = 10
        tick_positions = np.linspace(0, len(data)-1, n_ticks, dtype=int)
        tick_labels = [data['Date'].iloc[i].strftime('%Y-%m') if hasattr(data['Date'].iloc[i], 'strftime') 
                      else str(data['Date'].iloc[i])[:7] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    title = f"{ticker} - Canal Haussier WEEKLY"
    if position_type:
        title += f" [{position_type}]"
    title += f"\n"
    title += f"Creux (enfoncent support): {channel['lows_count']} | "
    title += f"Sommets (dépassent résistance): {channel['highs_count']} | R²: {channel['r_squared']:.3f}"
    title += f"\nPrix: ${current_price:.2f} | ATH: ${ath:.2f} ({pct_from_ath:+.1f}%) | "
    title += f"Alternance: {'✅ Parfaite' if verify_alternation(channel['low_indices'], channel['high_indices']) else '❌'}"
    
    ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=20)
    ax.legend(loc='upper left', facecolor='#2d2d2d', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_xlabel('Semaines', color='white')
    ax.set_ylabel('Prix ($)', color='white')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
        print(f"   💾 Sauvegardé: {save_path}")
    
    plt.show()
    return channel

def plot_stock_with_pivots(ticker, data, config):
    """Affiche le graphique WEEKLY avec les pivots alternés même sans canal valide"""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    for i in range(len(data)):
        o = float(data['Open'].iloc[i])
        c = float(data['Close'].iloc[i])
        h = float(data['High'].iloc[i])
        l = float(data['Low'].iloc[i])
        
        color = '#26a69a' if c >= o else '#ef5350'
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.plot([i, i], [o, c], color=color, linewidth=4)
    
    high_idx, low_idx = find_pivot_points(data, order=config['pivot_order'])
    low_idx = refine_pivot_points(data, low_idx, 'low')
    high_idx = refine_pivot_points(data, high_idx, 'high')
    low_idx, high_idx = alternate_pivots_simple(data, low_idx, high_idx)
    
    for li in low_idx:
        ax.scatter(li, float(data['Low'].iloc[li]), color='#4caf50', s=120, marker='^', 
                  edgecolors='white', linewidths=1.5)
    for hi in high_idx:
        ax.scatter(hi, float(data['High'].iloc[hi]), color='#f44336', s=120, marker='v',
                  edgecolors='white', linewidths=1.5)
    
    all_pivots = [(int(li), float(data['Low'].iloc[li])) for li in low_idx] + \
                 [(int(hi), float(data['High'].iloc[hi])) for hi in high_idx]
    all_pivots.sort(key=lambda x: x[0])
    
    if len(all_pivots) > 1:
        for i in range(len(all_pivots) - 1):
            x1, y1 = all_pivots[i]
            x2, y2 = all_pivots[i + 1]
            ax.plot([x1, x2], [y1, y2], '--', color='white', alpha=0.4, linewidth=1)
    
    ax.set_title(f"{ticker} - Pivots Alternés WEEKLY (pas de canal haussier valide)", 
                fontsize=14, color='white')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# DIAGNOSTIC
# ============================================================================
def diagnose_stock(ticker, config=CONFIG):
    """Diagnostic détaillé en WEEKLY"""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC DÉTAILLÉ (WEEKLY): {ticker}")
    print(f"{'='*70}")
    
    data, error = download_stock_data(ticker, config)
    
    if data is None:
        print(f"   ❌ Erreur: {error}")
        return
    
    print(f"\n1️⃣ DONNÉES: {len(data)} semaines")
    print(f"   Prix actuel: ${float(data['Close'].iloc[-1]):.2f}")
    print(f"   ATH: ${float(data['High'].max()):.2f}")
    print(f"   Plus bas: ${float(data['Low'].min()):.2f}")
    if 'Date' in data.columns:
        print(f"   Période: {data['Date'].iloc[0]} → {data['Date'].iloc[-1]}")
    
    print(f"\n2️⃣ PIVOTS BRUTS")
    high_idx, low_idx = find_pivot_points(data, order=config['pivot_order'])
    print(f"   Creux: {len(low_idx)} | Sommets: {len(high_idx)}")
    
    print(f"\n3️⃣ PIVOTS RAFFINÉS")
    low_idx = refine_pivot_points(data, low_idx, 'low')
    high_idx = refine_pivot_points(data, high_idx, 'high')
    print(f"   Creux: {len(low_idx)} | Sommets: {len(high_idx)}")
    
    print(f"\n4️⃣ DÉTECTION DU CANAL (avec sélection par rapport au canal)")
    channel = detect_ascending_channel(data, config, debug=True)
    
    if channel:
        print(f"\n5️⃣ SÉQUENCE DES PIVOTS VALIDES")
        all_pivots = [(l['index'], 'L', l['price'], l['distance_pct']) for l in channel['valid_lows']] + \
                     [(h['index'], 'H', h['price'], h['distance_pct']) for h in channel['valid_highs']]
        all_pivots.sort(key=lambda x: x[0])
        
        for idx, ptype, price, dist in all_pivots:
            arrow = "▼" if ptype == 'L' else "▲"
            color_word = "support" if ptype == 'L' else "résist."
            print(f"   {arrow} {ptype} sem={idx:3d} ${price:8.2f} ({dist:+6.1f}% vs {color_word})")
    
    print(f"\n6️⃣ VISUALISATION")
    plot_channel(ticker, config)

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================
def main(market="US"):
    """
    Fonction principale
    
    Args:
        market: "US" pour S&P 500, "EUR" pour CAC 40
    """
    print("\n" + "=" * 70)
    print(f"SCANNER DE CANAUX HAUSSIERS - {market.upper()}")
    print("=" * 70)

    results, index_name = scan_channels(market=market, config=CONFIG, max_stocks=None)

    print("\n" + "=" * 70)
    print(f"RÉSULTATS FINAUX - {index_name}")
    print("=" * 70)

    if len(results) > 0:
        print(f"\n🎯 {len(results)} actions détectées:\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 100)
        print(results.to_string(index=False))

        filename = f"canaux_{market.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(filename, index=False)
        print(f"\n💾 Résultats Sauvegardé: {filename}")

        # Filtrer les résultats où Pos.Canal % < 20%
        best_results = results[results['Pos. Canal %'] < 20]

        # ===================================================================
        # LOAD PREVIOUS REPORT TO IDENTIFY NEW ITEMS ONLY
        # ===================================================================
        # Find all previous report files for this market, sorted by name (date)
        previous_files = sorted(glob.glob(f"canaux_{market.upper()}_*.csv"))

        # Remove the file we just saved from the list
        if filename in previous_files:
            previous_files.remove(filename)

        previous_tickers_below_20 = set()

        if previous_files:
            latest_previous_file = previous_files[-1]  # most recent before current
            print(f"\n📂 Rapport précédent trouvé: {latest_previous_file}")
            try:
                previous_results = pd.read_csv(latest_previous_file)
                # Apply the same < 20% filter on the previous report
                previous_best = previous_results[previous_results['Pos. Canal %'] < 20]
                previous_tickers_below_20 = set(previous_best['Ticker'].tolist())
                print(f"   ➡️  {len(previous_tickers_below_20)} tickers étaient déjà < 20% "
                      f"dans le rapport précédent")
            except Exception as e:
                print(f"   ⚠️ Erreur lecture rapport précédent: {e}")
                print(f"   ➡️  Tous les résultats seront affichés")
        else:
            print("\n📂 Aucun rapport précédent trouvé — tous les résultats seront affichés")

        # Keep only NEW tickers (not present in previous report's < 20% list)
        new_best_results = best_results[
            ~best_results['Ticker'].isin(previous_tickers_below_20)
        ]

        print(f"\n📊 Résumé du filtrage:")
        print(f"   • Total actions < 20%          : {len(best_results)}")
        print(f"   • Déjà présentes (précédent)   : "
              f"{len(best_results) - len(new_best_results)}")
        print(f"   • 🆕 Nouvelles entrées          : {len(new_best_results)}")

        # ===== TOP 20 — NEW ITEMS ONLY =====
        print("\n" + "=" * 70)
        print("TOP 20 - 🆕 NOUVEAUX CANAUX HAUSSIERS (WEEKLY) dans le bas du canal")
        print("=" * 70)

        if len(new_best_results) > 0:
            print(f"\n{new_best_results.head(20).to_string(index=False)}")

            for i, ticker in enumerate(new_best_results['Ticker'].head(20)):
                print(f"\n{'─' * 70}")
                print(f"#{i + 1} - {ticker} 🆕")
                print(f"{'─' * 70}")
                plot_channel(ticker, CONFIG, save_path=f"{ticker}_canal_weekly.png")
        else:
            print("\n✅ Aucune nouvelle action détectée par rapport au scan précédent.")

    else:
        print("\n⚠️ Aucun canal trouvé avec les critères actuels.")
        print("\n💡 Pour diagnostiquer une action spécifique:")
    print("   diagnose_stock('NVDA', CONFIG)")

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Récupérer le marché depuis les arguments
    if len(sys.argv) > 1:
        market = sys.argv[1].upper()
    else:
        market = "US"  # Par défaut
    
    if market not in ["US", "EUR", "FR", "DE", "MDAX", "UK", "ES", "IT", "CH", "NL", "BE", "SE", "DK", "FI", "NO", "PT", "AT", "IE"]:
        print("❌ Marché non reconnu. Utilisez 'US' ou 'EUR'")
        print("   Exemple: python scanner.py US")
        print("   Exemple: python scanner.py EUR")
        sys.exit(1)
    
    main(market=market)
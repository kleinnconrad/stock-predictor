# config/universe.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# 1. THE 360-DEGREE INVESTMENT UNIVERSE
# ==========================================

# Macro Basics
MACRO_INDICATORS = [
    '^TNX',    # US 10-Year Treasury Yield
    '^IRX',    # 13-Week Treasury Bill 
    '^VIX',    # Volatility Index 
    'DX-Y.NYB',# US Dollar Index
]

# Industrial & Precious Metals
COMMODITIES = [
    'CL=F',    # Crude Oil 
    'GC=F',    # Gold 
    'HG=F',    # Copper ("Dr. Copper")
]

# Agricultural Commodities
AGRI_COMMODITIES = [
    'ZC=F',    # Corn (Mais)
    'ZW=F',    # Wheat (Weizen)
    'LE=F',    # Live Cattle (Lebendrind)
]

# Credit Risk & Bonds
CREDIT_RISK = [
    'HYG',     # High Yield Corporate Bonds (Junk Bonds)
    'TLT',     # 20+ Year Treasury Bonds (Safe Haven)
    'LQD',     # Investment Grade Corporate Bonds
]

# Broad Sectors & International Indices
SECTORS_AND_INDICES = [
    'XLF', 'XLK', 'XLE', 'XBI', 'EEM', '^GDAXI', '^N225'
]

# Sector Rotation
MORE_SECTORS = [
    'XLU', 'XLP', 'XLY', 'XLV'
]

# Real Estate & Crypto
REAL_ESTATE = ['VNQ']
CRYPTO = ['BTC-USD']

# Systemically relevant stocks
TICKERS_US = ['AAPL', 'MSFT', 'NVDA', 'BRK-B', 'JPM']
TICKERS_DE = ['SAP.DE', 'SIE.DE', 'BAS.DE']
TICKERS_UK = ['SHEL.L', 'AZN.L', 'RIO.L']
TICKERS_JP = ['7203.T', '9984.T', '8035.T']

# Sovereign Yields & Factor ETFs & FX
SOVEREIGN_YIELDS = ['IGOV', 'BWX', 'BNDX']
FACTOR_ETFS = ['IWM', 'IYT', 'RSP', 'SMH']
MORE_COMMODITIES_AND_FX = ['LBS=F', 'EURUSD=X', 'JPY=X']

# ==========================================
# 2. FRED INDICATORS
# ==========================================

FRED_INDICATORS = ['CPIAUCSL', 'PAYEMS', 'UNRATE', 'T10Y2Y', 'WALCL']
FRED_INDICATORS_EU = ['CP00MI15EA20M086NEST', 'LRHUTTTTEZM156S', 'ECBASSETS', 'PRINTO01EZQ661S']
FRED_INDICATORS_JP = ['JPNCPIALLMINMEI', 'LRHUTTTTJPM156S', 'JPNASSETS', 'JPNPROINDMISMEI']
FRED_INDICATORS_UK = ['GBRCPIALLMINMEI', 'LRHUTTTTGBM156S', 'GBRPROINDMISMEI']
FRED_INDICATORS_UNCERTAINTY = ['USEPUINDXD', 'GEPUCURRENT']
FRED_LIQUIDITY_AND_CREDIT = ['M2SL', 'NFCI']
FRED_LEADING_MACRO = ['PERMIT', 'ICSA', 'UMCSENT', 'DGORDER']

# Aggregated Master Lists for the ingestion engine
ALL_FRED_INDICATORS = (
    FRED_INDICATORS + FRED_INDICATORS_EU + FRED_INDICATORS_JP + 
    FRED_INDICATORS_UK + FRED_INDICATORS_UNCERTAINTY + 
    FRED_LIQUIDITY_AND_CREDIT + FRED_LEADING_MACRO
)

ALL_YF_TICKERS = list(set(
    ['SPY'] + 
    MACRO_INDICATORS + COMMODITIES + AGRI_COMMODITIES + 
    CREDIT_RISK + SECTORS_AND_INDICES + MORE_SECTORS + 
    REAL_ESTATE + CRYPTO + 
    TICKERS_US + TICKERS_DE + TICKERS_UK + TICKERS_JP +
    SOVEREIGN_YIELDS + FACTOR_ETFS + MORE_COMMODITIES_AND_FX
))

# ==========================================
# 3. FUNDAMENTAL UNIVERSE (For Step 2)
# ==========================================
INCOME_STATEMENT = [
    'Total Revenue', 'Operating Revenue', 'Gross Profit', 'Operating Income',
    'EBIT', 'EBITDA', 'Net Income', 'Net Income Common Stockholders',
    'Basic EPS', 'Diluted EPS'
]

BALANCE_SHEET = [
    'Total Assets', 'Current Assets', 'Total Liabilities Net Minority Interest',
    'Current Liabilities', 'Total Debt', 'Net Debt', 'Cash And Cash Equivalents',
    'Stockholders Equity', 'Working Capital', 'Retained Earnings'
]

CASH_FLOW = [
    'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow',
    'Free Cash Flow', 'Capital Expenditure', 'Repayment Of Debt', 'Issuance Of Debt'
]

FUNDAMENTAL_UNIVERSE = INCOME_STATEMENT + BALANCE_SHEET + CASH_FLOW

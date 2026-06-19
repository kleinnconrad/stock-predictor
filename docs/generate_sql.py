import sys
import os

sys.path.append(r'c:\Users\klein\Desktop\stock-predictor')

from config.universe import ALL_FRED_INDICATORS, ALL_YF_TICKERS

RATE_KEYWORDS = ['TNX', 'IRX', 'VIX', 'UNRATE', 'T10Y2Y', 'EPU', 'ratio_', 'HUTTTT', 'NFCI', 'BAML', 'UMCSENT']
MACRO_KEYWORDS = ['CPIAUCSL', 'M2SL', 'PAYEMS', 'UNRATE', 'ASSETS', 'PROIND', 'PERMIT']
windows = [21, 63, 126, 252] 

def escape_col(c):
    if not c.replace('_', '').isalnum():
        return f'"{c}"'
    return c

def format_col(c):
    return c.replace('^', '').replace('=', '').replace('-', '_').replace('.', '_')

ALL_COLS = list(ALL_FRED_INDICATORS) + list(ALL_YF_TICKERS) + [
    'ratio_copper_gold', 'ratio_credit_spread', 'ratio_consumer_risk', 
    'ratio_risk_on_off', 'ratio_tech_dominance', 'ratio_intl_vs_us_bonds'
]

seen = set()
ALL_COLS_DEDUP = []
for c in ALL_COLS:
    if c not in seen:
        seen.add(c)
        ALL_COLS_DEDUP.append(c)

sql_lines = []
sql_lines.append("-- ==============================================================================")
sql_lines.append("-- GLOBAL MACRO & PRICING FEATURE ENGINEERING LOGIC (FULL EXHAUSTIVE UNIVERSE)")
sql_lines.append("-- ==============================================================================")
sql_lines.append("")
sql_lines.append("WITH base_data AS (")
sql_lines.append("    SELECT * FROM daily_market_data")
sql_lines.append("),")
sql_lines.append("")
sql_lines.append("macro_features AS (")
sql_lines.append("    SELECT ")
sql_lines.append("        date,")
sql_lines.append("        Close,")
sql_lines.append("")
sql_lines.append("        -- =======================")
sql_lines.append("        -- A. INTERACTION RATIOS")
sql_lines.append("        -- =======================")
sql_lines.append("        \"HG=F\" / NULLIF(\"GC=F\", 0) AS ratio_copper_gold,")
sql_lines.append("        HYG / NULLIF(LQD, 0)       AS ratio_credit_spread,")
sql_lines.append("        XLY / NULLIF(XLP, 0)       AS ratio_consumer_risk,")
sql_lines.append("        SPY / NULLIF(TLT, 0)       AS ratio_risk_on_off,")
sql_lines.append("        XLK / NULLIF(SPY, 0)       AS ratio_tech_dominance,")
sql_lines.append("        IGOV / NULLIF(TLT, 0)      AS ratio_intl_vs_us_bonds,")
sql_lines.append("")

for i, col in enumerate(ALL_COLS_DEDUP):
    escaped = escape_col(col)
    alias_base = format_col(col)
    
    # Check if this is the very last column to omit the trailing comma
    # Wait, we can just omit the trailing comma logic during the loop and fix the last line later.
    
    if col in ALL_FRED_INDICATORS:
        sql_lines.append(f"        -- FRED Indicator: {col}")
        sql_lines.append(f"        {escaped} AS {alias_base}_Level,")
        sql_lines.append(f"        ({escaped} - AVG({escaped}) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV({escaped}) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS {alias_base}_ZScore,")
        continue
        
    is_rate_or_spread = any(kw in col for kw in RATE_KEYWORDS)
    sql_lines.append(f"        -- Market Indicator: {col}")
    if is_rate_or_spread:
        sql_lines.append(f"        {escaped} AS {alias_base}_Level,")
        
    for w in windows:
        if is_rate_or_spread:
            sql_lines.append(f"        {escaped} - LAG({escaped}, {w}) OVER (ORDER BY date) AS {alias_base}_{w}D_diff,")
        else:
            sql_lines.append(f"        ({escaped} / NULLIF(LAG({escaped}, {w}) OVER (ORDER BY date), 0)) - 1 AS {alias_base}_{w}D_ret,")
            
    if is_rate_or_spread:
        sql_lines.append(f"        {escaped} - AVG({escaped}) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS {alias_base}_Dist_SMA200,")
    else:
        sql_lines.append(f"        ({escaped} / NULLIF(AVG({escaped}) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS {alias_base}_Dist_SMA200,")
        
    if any(kw in col for kw in MACRO_KEYWORDS):
        if is_rate_or_spread:
            sql_lines.append(f"        ({escaped} - LAG({escaped}, 252) OVER (ORDER BY date)) - (LAG({escaped}, 63) OVER (ORDER BY date) - LAG({escaped}, 315) OVER (ORDER BY date)) AS {alias_base}_YoY_Accel_3M,")
        else:
            sql_lines.append(f"        ({escaped} / NULLIF(LAG({escaped}, 252) OVER (ORDER BY date), 0) - 1) - (LAG({escaped}, 63) OVER (ORDER BY date) / NULLIF(LAG({escaped}, 315) OVER (ORDER BY date), 0) - 1) AS {alias_base}_YoY_Accel_3M,")
            
    if 'VIX' in col or 'credit_spread' in col or 'EPU' in col:
        sql_lines.append(f"        ({escaped} - AVG({escaped}) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV({escaped}) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS {alias_base}_Roll_ZScore_2Y,")
        
    sql_lines.append("")

# Remove trailing comma from the very last feature column
last_line = sql_lines[-2]
if last_line.endswith(','):
    sql_lines[-2] = last_line[:-1]
elif sql_lines[-1].endswith(','):
    sql_lines[-1] = sql_lines[-1][:-1]
else:
    for i in range(len(sql_lines)-1, -1, -1):
        if sql_lines[i].endswith(','):
            sql_lines[i] = sql_lines[i][:-1]
            break

sql_lines.append("    FROM base_data")
sql_lines.append("),")
sql_lines.append("")
sql_lines.append("-- 2. TARGET ASSET FEATURE ENGINEERING (from features.py)")
sql_lines.append("target_features AS (")
sql_lines.append("    SELECT ")
sql_lines.append("        *,")
sql_lines.append("        ")
sql_lines.append("        -- Momentum / Returns for the target asset")
sql_lines.append("        (Close / NULLIF(LAG(Close, 21) OVER (ORDER BY date), 0)) - 1  AS Ret_21D,")
sql_lines.append("        (Close / NULLIF(LAG(Close, 63) OVER (ORDER BY date), 0)) - 1  AS Ret_63D,")
sql_lines.append("        (Close / NULLIF(LAG(Close, 126) OVER (ORDER BY date), 0)) - 1 AS Ret_126D,")
sql_lines.append("        (Close / NULLIF(LAG(Close, 252) OVER (ORDER BY date), 0)) - 1 AS Ret_252D,")
sql_lines.append("")
sql_lines.append("        -- Moving Average Distances")
sql_lines.append("        (Close / NULLIF(AVG(Close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 0)) - 1 AS Dist_SMA_50,")
sql_lines.append("        (Close / NULLIF(AVG(Close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS Dist_SMA_200,")
sql_lines.append("")
sql_lines.append("        -- Volatility (Rolling standard deviation of daily returns over 21 days)")
sql_lines.append("        STDDEV( (Close / NULLIF(LAG(Close, 1) OVER (ORDER BY date), 0)) - 1 ) ")
sql_lines.append("            OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS Vol_21D,")
sql_lines.append("")
sql_lines.append("        -- Target Variable Calculation (Future Return over horizon_days, e.g., 126)")
sql_lines.append("        (LEAD(Close, 126) OVER (ORDER BY date) / NULLIF(Close, 0)) - 1 AS Future_Return")
sql_lines.append("    FROM macro_features")
sql_lines.append(")")
sql_lines.append("")
sql_lines.append("-- 3. FINAL OUTPUT WITH BINARY TARGET")
sql_lines.append("SELECT ")
sql_lines.append("    *,")
sql_lines.append("    CASE ")
sql_lines.append("        WHEN Future_Return >= 0.15 THEN 1.0 ")
sql_lines.append("        WHEN Future_Return IS NULL THEN NULL ")
sql_lines.append("        ELSE 0.0 ")
sql_lines.append("    END AS Target")
sql_lines.append("FROM target_features;")
sql_lines.append("")

out_path = r'c:\Users\klein\Desktop\stock-predictor\docs\feature_engineering.sql'
with open(out_path, 'w') as f:
    f.write('\n'.join(sql_lines))

print(f"Successfully generated {out_path} with exhaustive features!")

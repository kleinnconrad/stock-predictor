-- ==============================================================================
-- GLOBAL MACRO & PRICING FEATURE ENGINEERING LOGIC (FULL EXHAUSTIVE UNIVERSE)
-- ==============================================================================

WITH base_data AS (
    SELECT * FROM daily_market_data
),

macro_features AS (
    SELECT 
        date,
        Close,

        -- =======================
        -- A. INTERACTION RATIOS
        -- =======================
        "HG=F" / NULLIF("GC=F", 0) AS ratio_copper_gold,
        HYG / NULLIF(LQD, 0)       AS ratio_credit_spread,
        XLY / NULLIF(XLP, 0)       AS ratio_consumer_risk,
        SPY / NULLIF(TLT, 0)       AS ratio_risk_on_off,
        XLK / NULLIF(SPY, 0)       AS ratio_tech_dominance,
        IGOV / NULLIF(TLT, 0)      AS ratio_intl_vs_us_bonds,

        -- FRED Indicator: CPIAUCSL
        CPIAUCSL AS CPIAUCSL_Level,
        (CPIAUCSL - AVG(CPIAUCSL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(CPIAUCSL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS CPIAUCSL_ZScore,
        -- FRED Indicator: PAYEMS
        PAYEMS AS PAYEMS_Level,
        (PAYEMS - AVG(PAYEMS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(PAYEMS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS PAYEMS_ZScore,
        -- FRED Indicator: UNRATE
        UNRATE AS UNRATE_Level,
        (UNRATE - AVG(UNRATE) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(UNRATE) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS UNRATE_ZScore,
        -- FRED Indicator: T10Y2Y
        T10Y2Y AS T10Y2Y_Level,
        (T10Y2Y - AVG(T10Y2Y) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(T10Y2Y) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS T10Y2Y_ZScore,
        -- FRED Indicator: WALCL
        WALCL AS WALCL_Level,
        (WALCL - AVG(WALCL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(WALCL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS WALCL_ZScore,
        -- FRED Indicator: CP00MI15EA20M086NEST
        CP00MI15EA20M086NEST AS CP00MI15EA20M086NEST_Level,
        (CP00MI15EA20M086NEST - AVG(CP00MI15EA20M086NEST) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(CP00MI15EA20M086NEST) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS CP00MI15EA20M086NEST_ZScore,
        -- FRED Indicator: LRHUTTTTEZM156S
        LRHUTTTTEZM156S AS LRHUTTTTEZM156S_Level,
        (LRHUTTTTEZM156S - AVG(LRHUTTTTEZM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(LRHUTTTTEZM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS LRHUTTTTEZM156S_ZScore,
        -- FRED Indicator: ECBASSETS
        ECBASSETS AS ECBASSETS_Level,
        (ECBASSETS - AVG(ECBASSETS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(ECBASSETS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS ECBASSETS_ZScore,
        -- FRED Indicator: PRINTO01EZQ661S
        PRINTO01EZQ661S AS PRINTO01EZQ661S_Level,
        (PRINTO01EZQ661S - AVG(PRINTO01EZQ661S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(PRINTO01EZQ661S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS PRINTO01EZQ661S_ZScore,
        -- FRED Indicator: JPNCPIALLMINMEI
        JPNCPIALLMINMEI AS JPNCPIALLMINMEI_Level,
        (JPNCPIALLMINMEI - AVG(JPNCPIALLMINMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(JPNCPIALLMINMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS JPNCPIALLMINMEI_ZScore,
        -- FRED Indicator: LRHUTTTTJPM156S
        LRHUTTTTJPM156S AS LRHUTTTTJPM156S_Level,
        (LRHUTTTTJPM156S - AVG(LRHUTTTTJPM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(LRHUTTTTJPM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS LRHUTTTTJPM156S_ZScore,
        -- FRED Indicator: JPNASSETS
        JPNASSETS AS JPNASSETS_Level,
        (JPNASSETS - AVG(JPNASSETS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(JPNASSETS) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS JPNASSETS_ZScore,
        -- FRED Indicator: JPNPROINDMISMEI
        JPNPROINDMISMEI AS JPNPROINDMISMEI_Level,
        (JPNPROINDMISMEI - AVG(JPNPROINDMISMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(JPNPROINDMISMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS JPNPROINDMISMEI_ZScore,
        -- FRED Indicator: GBRCPIALLMINMEI
        GBRCPIALLMINMEI AS GBRCPIALLMINMEI_Level,
        (GBRCPIALLMINMEI - AVG(GBRCPIALLMINMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(GBRCPIALLMINMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS GBRCPIALLMINMEI_ZScore,
        -- FRED Indicator: LRHUTTTTGBM156S
        LRHUTTTTGBM156S AS LRHUTTTTGBM156S_Level,
        (LRHUTTTTGBM156S - AVG(LRHUTTTTGBM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(LRHUTTTTGBM156S) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS LRHUTTTTGBM156S_ZScore,
        -- FRED Indicator: GBRPROINDMISMEI
        GBRPROINDMISMEI AS GBRPROINDMISMEI_Level,
        (GBRPROINDMISMEI - AVG(GBRPROINDMISMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(GBRPROINDMISMEI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS GBRPROINDMISMEI_ZScore,
        -- FRED Indicator: USEPUINDXD
        USEPUINDXD AS USEPUINDXD_Level,
        (USEPUINDXD - AVG(USEPUINDXD) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(USEPUINDXD) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS USEPUINDXD_ZScore,
        -- FRED Indicator: GEPUCURRENT
        GEPUCURRENT AS GEPUCURRENT_Level,
        (GEPUCURRENT - AVG(GEPUCURRENT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(GEPUCURRENT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS GEPUCURRENT_ZScore,
        -- FRED Indicator: M2SL
        M2SL AS M2SL_Level,
        (M2SL - AVG(M2SL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(M2SL) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS M2SL_ZScore,
        -- FRED Indicator: NFCI
        NFCI AS NFCI_Level,
        (NFCI - AVG(NFCI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(NFCI) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS NFCI_ZScore,
        -- FRED Indicator: PERMIT
        PERMIT AS PERMIT_Level,
        (PERMIT - AVG(PERMIT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(PERMIT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS PERMIT_ZScore,
        -- FRED Indicator: ICSA
        ICSA AS ICSA_Level,
        (ICSA - AVG(ICSA) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(ICSA) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS ICSA_ZScore,
        -- FRED Indicator: UMCSENT
        UMCSENT AS UMCSENT_Level,
        (UMCSENT - AVG(UMCSENT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(UMCSENT) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS UMCSENT_ZScore,
        -- FRED Indicator: DGORDER
        DGORDER AS DGORDER_Level,
        (DGORDER - AVG(DGORDER) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(DGORDER) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS DGORDER_ZScore,
        -- Market Indicator: ^IRX
        "^IRX" AS IRX_Level,
        "^IRX" - LAG("^IRX", 21) OVER (ORDER BY date) AS IRX_21D_diff,
        "^IRX" - LAG("^IRX", 63) OVER (ORDER BY date) AS IRX_63D_diff,
        "^IRX" - LAG("^IRX", 126) OVER (ORDER BY date) AS IRX_126D_diff,
        "^IRX" - LAG("^IRX", 252) OVER (ORDER BY date) AS IRX_252D_diff,
        "^IRX" - AVG("^IRX") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS IRX_Dist_SMA200,

        -- Market Indicator: MSFT
        (MSFT / NULLIF(LAG(MSFT, 21) OVER (ORDER BY date), 0)) - 1 AS MSFT_21D_ret,
        (MSFT / NULLIF(LAG(MSFT, 63) OVER (ORDER BY date), 0)) - 1 AS MSFT_63D_ret,
        (MSFT / NULLIF(LAG(MSFT, 126) OVER (ORDER BY date), 0)) - 1 AS MSFT_126D_ret,
        (MSFT / NULLIF(LAG(MSFT, 252) OVER (ORDER BY date), 0)) - 1 AS MSFT_252D_ret,
        (MSFT / NULLIF(AVG(MSFT) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS MSFT_Dist_SMA200,

        -- Market Indicator: AAPL
        (AAPL / NULLIF(LAG(AAPL, 21) OVER (ORDER BY date), 0)) - 1 AS AAPL_21D_ret,
        (AAPL / NULLIF(LAG(AAPL, 63) OVER (ORDER BY date), 0)) - 1 AS AAPL_63D_ret,
        (AAPL / NULLIF(LAG(AAPL, 126) OVER (ORDER BY date), 0)) - 1 AS AAPL_126D_ret,
        (AAPL / NULLIF(LAG(AAPL, 252) OVER (ORDER BY date), 0)) - 1 AS AAPL_252D_ret,
        (AAPL / NULLIF(AVG(AAPL) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS AAPL_Dist_SMA200,

        -- Market Indicator: XLU
        (XLU / NULLIF(LAG(XLU, 21) OVER (ORDER BY date), 0)) - 1 AS XLU_21D_ret,
        (XLU / NULLIF(LAG(XLU, 63) OVER (ORDER BY date), 0)) - 1 AS XLU_63D_ret,
        (XLU / NULLIF(LAG(XLU, 126) OVER (ORDER BY date), 0)) - 1 AS XLU_126D_ret,
        (XLU / NULLIF(LAG(XLU, 252) OVER (ORDER BY date), 0)) - 1 AS XLU_252D_ret,
        (XLU / NULLIF(AVG(XLU) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLU_Dist_SMA200,

        -- Market Indicator: ^VIX
        "^VIX" AS VIX_Level,
        "^VIX" - LAG("^VIX", 21) OVER (ORDER BY date) AS VIX_21D_diff,
        "^VIX" - LAG("^VIX", 63) OVER (ORDER BY date) AS VIX_63D_diff,
        "^VIX" - LAG("^VIX", 126) OVER (ORDER BY date) AS VIX_126D_diff,
        "^VIX" - LAG("^VIX", 252) OVER (ORDER BY date) AS VIX_252D_diff,
        "^VIX" - AVG("^VIX") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS VIX_Dist_SMA200,
        ("^VIX" - AVG("^VIX") OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV("^VIX") OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS VIX_Roll_ZScore_2Y,

        -- Market Indicator: JPM
        (JPM / NULLIF(LAG(JPM, 21) OVER (ORDER BY date), 0)) - 1 AS JPM_21D_ret,
        (JPM / NULLIF(LAG(JPM, 63) OVER (ORDER BY date), 0)) - 1 AS JPM_63D_ret,
        (JPM / NULLIF(LAG(JPM, 126) OVER (ORDER BY date), 0)) - 1 AS JPM_126D_ret,
        (JPM / NULLIF(LAG(JPM, 252) OVER (ORDER BY date), 0)) - 1 AS JPM_252D_ret,
        (JPM / NULLIF(AVG(JPM) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS JPM_Dist_SMA200,

        -- Market Indicator: BWX
        (BWX / NULLIF(LAG(BWX, 21) OVER (ORDER BY date), 0)) - 1 AS BWX_21D_ret,
        (BWX / NULLIF(LAG(BWX, 63) OVER (ORDER BY date), 0)) - 1 AS BWX_63D_ret,
        (BWX / NULLIF(LAG(BWX, 126) OVER (ORDER BY date), 0)) - 1 AS BWX_126D_ret,
        (BWX / NULLIF(LAG(BWX, 252) OVER (ORDER BY date), 0)) - 1 AS BWX_252D_ret,
        (BWX / NULLIF(AVG(BWX) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS BWX_Dist_SMA200,

        -- Market Indicator: RSP
        (RSP / NULLIF(LAG(RSP, 21) OVER (ORDER BY date), 0)) - 1 AS RSP_21D_ret,
        (RSP / NULLIF(LAG(RSP, 63) OVER (ORDER BY date), 0)) - 1 AS RSP_63D_ret,
        (RSP / NULLIF(LAG(RSP, 126) OVER (ORDER BY date), 0)) - 1 AS RSP_126D_ret,
        (RSP / NULLIF(LAG(RSP, 252) OVER (ORDER BY date), 0)) - 1 AS RSP_252D_ret,
        (RSP / NULLIF(AVG(RSP) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS RSP_Dist_SMA200,

        -- Market Indicator: HYG
        (HYG / NULLIF(LAG(HYG, 21) OVER (ORDER BY date), 0)) - 1 AS HYG_21D_ret,
        (HYG / NULLIF(LAG(HYG, 63) OVER (ORDER BY date), 0)) - 1 AS HYG_63D_ret,
        (HYG / NULLIF(LAG(HYG, 126) OVER (ORDER BY date), 0)) - 1 AS HYG_126D_ret,
        (HYG / NULLIF(LAG(HYG, 252) OVER (ORDER BY date), 0)) - 1 AS HYG_252D_ret,
        (HYG / NULLIF(AVG(HYG) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS HYG_Dist_SMA200,

        -- Market Indicator: XBI
        (XBI / NULLIF(LAG(XBI, 21) OVER (ORDER BY date), 0)) - 1 AS XBI_21D_ret,
        (XBI / NULLIF(LAG(XBI, 63) OVER (ORDER BY date), 0)) - 1 AS XBI_63D_ret,
        (XBI / NULLIF(LAG(XBI, 126) OVER (ORDER BY date), 0)) - 1 AS XBI_126D_ret,
        (XBI / NULLIF(LAG(XBI, 252) OVER (ORDER BY date), 0)) - 1 AS XBI_252D_ret,
        (XBI / NULLIF(AVG(XBI) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XBI_Dist_SMA200,

        -- Market Indicator: XLP
        (XLP / NULLIF(LAG(XLP, 21) OVER (ORDER BY date), 0)) - 1 AS XLP_21D_ret,
        (XLP / NULLIF(LAG(XLP, 63) OVER (ORDER BY date), 0)) - 1 AS XLP_63D_ret,
        (XLP / NULLIF(LAG(XLP, 126) OVER (ORDER BY date), 0)) - 1 AS XLP_126D_ret,
        (XLP / NULLIF(LAG(XLP, 252) OVER (ORDER BY date), 0)) - 1 AS XLP_252D_ret,
        (XLP / NULLIF(AVG(XLP) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLP_Dist_SMA200,

        -- Market Indicator: XLV
        (XLV / NULLIF(LAG(XLV, 21) OVER (ORDER BY date), 0)) - 1 AS XLV_21D_ret,
        (XLV / NULLIF(LAG(XLV, 63) OVER (ORDER BY date), 0)) - 1 AS XLV_63D_ret,
        (XLV / NULLIF(LAG(XLV, 126) OVER (ORDER BY date), 0)) - 1 AS XLV_126D_ret,
        (XLV / NULLIF(LAG(XLV, 252) OVER (ORDER BY date), 0)) - 1 AS XLV_252D_ret,
        (XLV / NULLIF(AVG(XLV) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLV_Dist_SMA200,

        -- Market Indicator: DX-Y.NYB
        ("DX-Y.NYB" / NULLIF(LAG("DX-Y.NYB", 21) OVER (ORDER BY date), 0)) - 1 AS DX_Y_NYB_21D_ret,
        ("DX-Y.NYB" / NULLIF(LAG("DX-Y.NYB", 63) OVER (ORDER BY date), 0)) - 1 AS DX_Y_NYB_63D_ret,
        ("DX-Y.NYB" / NULLIF(LAG("DX-Y.NYB", 126) OVER (ORDER BY date), 0)) - 1 AS DX_Y_NYB_126D_ret,
        ("DX-Y.NYB" / NULLIF(LAG("DX-Y.NYB", 252) OVER (ORDER BY date), 0)) - 1 AS DX_Y_NYB_252D_ret,
        ("DX-Y.NYB" / NULLIF(AVG("DX-Y.NYB") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS DX_Y_NYB_Dist_SMA200,

        -- Market Indicator: OJ=F
        ("OJ=F" / NULLIF(LAG("OJ=F", 21) OVER (ORDER BY date), 0)) - 1 AS OJF_21D_ret,
        ("OJ=F" / NULLIF(LAG("OJ=F", 63) OVER (ORDER BY date), 0)) - 1 AS OJF_63D_ret,
        ("OJ=F" / NULLIF(LAG("OJ=F", 126) OVER (ORDER BY date), 0)) - 1 AS OJF_126D_ret,
        ("OJ=F" / NULLIF(LAG("OJ=F", 252) OVER (ORDER BY date), 0)) - 1 AS OJF_252D_ret,
        ("OJ=F" / NULLIF(AVG("OJ=F") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS OJF_Dist_SMA200,

        -- Market Indicator: IGOV
        (IGOV / NULLIF(LAG(IGOV, 21) OVER (ORDER BY date), 0)) - 1 AS IGOV_21D_ret,
        (IGOV / NULLIF(LAG(IGOV, 63) OVER (ORDER BY date), 0)) - 1 AS IGOV_63D_ret,
        (IGOV / NULLIF(LAG(IGOV, 126) OVER (ORDER BY date), 0)) - 1 AS IGOV_126D_ret,
        (IGOV / NULLIF(LAG(IGOV, 252) OVER (ORDER BY date), 0)) - 1 AS IGOV_252D_ret,
        (IGOV / NULLIF(AVG(IGOV) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS IGOV_Dist_SMA200,

        -- Market Indicator: BRK-B
        ("BRK-B" / NULLIF(LAG("BRK-B", 21) OVER (ORDER BY date), 0)) - 1 AS BRK_B_21D_ret,
        ("BRK-B" / NULLIF(LAG("BRK-B", 63) OVER (ORDER BY date), 0)) - 1 AS BRK_B_63D_ret,
        ("BRK-B" / NULLIF(LAG("BRK-B", 126) OVER (ORDER BY date), 0)) - 1 AS BRK_B_126D_ret,
        ("BRK-B" / NULLIF(LAG("BRK-B", 252) OVER (ORDER BY date), 0)) - 1 AS BRK_B_252D_ret,
        ("BRK-B" / NULLIF(AVG("BRK-B") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS BRK_B_Dist_SMA200,

        -- Market Indicator: AZN.L
        ("AZN.L" / NULLIF(LAG("AZN.L", 21) OVER (ORDER BY date), 0)) - 1 AS AZN_L_21D_ret,
        ("AZN.L" / NULLIF(LAG("AZN.L", 63) OVER (ORDER BY date), 0)) - 1 AS AZN_L_63D_ret,
        ("AZN.L" / NULLIF(LAG("AZN.L", 126) OVER (ORDER BY date), 0)) - 1 AS AZN_L_126D_ret,
        ("AZN.L" / NULLIF(LAG("AZN.L", 252) OVER (ORDER BY date), 0)) - 1 AS AZN_L_252D_ret,
        ("AZN.L" / NULLIF(AVG("AZN.L") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS AZN_L_Dist_SMA200,

        -- Market Indicator: ^GDAXI
        ("^GDAXI" / NULLIF(LAG("^GDAXI", 21) OVER (ORDER BY date), 0)) - 1 AS GDAXI_21D_ret,
        ("^GDAXI" / NULLIF(LAG("^GDAXI", 63) OVER (ORDER BY date), 0)) - 1 AS GDAXI_63D_ret,
        ("^GDAXI" / NULLIF(LAG("^GDAXI", 126) OVER (ORDER BY date), 0)) - 1 AS GDAXI_126D_ret,
        ("^GDAXI" / NULLIF(LAG("^GDAXI", 252) OVER (ORDER BY date), 0)) - 1 AS GDAXI_252D_ret,
        ("^GDAXI" / NULLIF(AVG("^GDAXI") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS GDAXI_Dist_SMA200,

        -- Market Indicator: ^TNX
        "^TNX" AS TNX_Level,
        "^TNX" - LAG("^TNX", 21) OVER (ORDER BY date) AS TNX_21D_diff,
        "^TNX" - LAG("^TNX", 63) OVER (ORDER BY date) AS TNX_63D_diff,
        "^TNX" - LAG("^TNX", 126) OVER (ORDER BY date) AS TNX_126D_diff,
        "^TNX" - LAG("^TNX", 252) OVER (ORDER BY date) AS TNX_252D_diff,
        "^TNX" - AVG("^TNX") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS TNX_Dist_SMA200,

        -- Market Indicator: BTC-USD
        ("BTC-USD" / NULLIF(LAG("BTC-USD", 21) OVER (ORDER BY date), 0)) - 1 AS BTC_USD_21D_ret,
        ("BTC-USD" / NULLIF(LAG("BTC-USD", 63) OVER (ORDER BY date), 0)) - 1 AS BTC_USD_63D_ret,
        ("BTC-USD" / NULLIF(LAG("BTC-USD", 126) OVER (ORDER BY date), 0)) - 1 AS BTC_USD_126D_ret,
        ("BTC-USD" / NULLIF(LAG("BTC-USD", 252) OVER (ORDER BY date), 0)) - 1 AS BTC_USD_252D_ret,
        ("BTC-USD" / NULLIF(AVG("BTC-USD") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS BTC_USD_Dist_SMA200,

        -- Market Indicator: TLT
        (TLT / NULLIF(LAG(TLT, 21) OVER (ORDER BY date), 0)) - 1 AS TLT_21D_ret,
        (TLT / NULLIF(LAG(TLT, 63) OVER (ORDER BY date), 0)) - 1 AS TLT_63D_ret,
        (TLT / NULLIF(LAG(TLT, 126) OVER (ORDER BY date), 0)) - 1 AS TLT_126D_ret,
        (TLT / NULLIF(LAG(TLT, 252) OVER (ORDER BY date), 0)) - 1 AS TLT_252D_ret,
        (TLT / NULLIF(AVG(TLT) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS TLT_Dist_SMA200,

        -- Market Indicator: IWM
        (IWM / NULLIF(LAG(IWM, 21) OVER (ORDER BY date), 0)) - 1 AS IWM_21D_ret,
        (IWM / NULLIF(LAG(IWM, 63) OVER (ORDER BY date), 0)) - 1 AS IWM_63D_ret,
        (IWM / NULLIF(LAG(IWM, 126) OVER (ORDER BY date), 0)) - 1 AS IWM_126D_ret,
        (IWM / NULLIF(LAG(IWM, 252) OVER (ORDER BY date), 0)) - 1 AS IWM_252D_ret,
        (IWM / NULLIF(AVG(IWM) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS IWM_Dist_SMA200,

        -- Market Indicator: SIE.DE
        ("SIE.DE" / NULLIF(LAG("SIE.DE", 21) OVER (ORDER BY date), 0)) - 1 AS SIE_DE_21D_ret,
        ("SIE.DE" / NULLIF(LAG("SIE.DE", 63) OVER (ORDER BY date), 0)) - 1 AS SIE_DE_63D_ret,
        ("SIE.DE" / NULLIF(LAG("SIE.DE", 126) OVER (ORDER BY date), 0)) - 1 AS SIE_DE_126D_ret,
        ("SIE.DE" / NULLIF(LAG("SIE.DE", 252) OVER (ORDER BY date), 0)) - 1 AS SIE_DE_252D_ret,
        ("SIE.DE" / NULLIF(AVG("SIE.DE") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS SIE_DE_Dist_SMA200,

        -- Market Indicator: SAP.DE
        ("SAP.DE" / NULLIF(LAG("SAP.DE", 21) OVER (ORDER BY date), 0)) - 1 AS SAP_DE_21D_ret,
        ("SAP.DE" / NULLIF(LAG("SAP.DE", 63) OVER (ORDER BY date), 0)) - 1 AS SAP_DE_63D_ret,
        ("SAP.DE" / NULLIF(LAG("SAP.DE", 126) OVER (ORDER BY date), 0)) - 1 AS SAP_DE_126D_ret,
        ("SAP.DE" / NULLIF(LAG("SAP.DE", 252) OVER (ORDER BY date), 0)) - 1 AS SAP_DE_252D_ret,
        ("SAP.DE" / NULLIF(AVG("SAP.DE") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS SAP_DE_Dist_SMA200,

        -- Market Indicator: LE=F
        ("LE=F" / NULLIF(LAG("LE=F", 21) OVER (ORDER BY date), 0)) - 1 AS LEF_21D_ret,
        ("LE=F" / NULLIF(LAG("LE=F", 63) OVER (ORDER BY date), 0)) - 1 AS LEF_63D_ret,
        ("LE=F" / NULLIF(LAG("LE=F", 126) OVER (ORDER BY date), 0)) - 1 AS LEF_126D_ret,
        ("LE=F" / NULLIF(LAG("LE=F", 252) OVER (ORDER BY date), 0)) - 1 AS LEF_252D_ret,
        ("LE=F" / NULLIF(AVG("LE=F") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS LEF_Dist_SMA200,

        -- Market Indicator: NVDA
        (NVDA / NULLIF(LAG(NVDA, 21) OVER (ORDER BY date), 0)) - 1 AS NVDA_21D_ret,
        (NVDA / NULLIF(LAG(NVDA, 63) OVER (ORDER BY date), 0)) - 1 AS NVDA_63D_ret,
        (NVDA / NULLIF(LAG(NVDA, 126) OVER (ORDER BY date), 0)) - 1 AS NVDA_126D_ret,
        (NVDA / NULLIF(LAG(NVDA, 252) OVER (ORDER BY date), 0)) - 1 AS NVDA_252D_ret,
        (NVDA / NULLIF(AVG(NVDA) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS NVDA_Dist_SMA200,

        -- Market Indicator: ^N225
        ("^N225" / NULLIF(LAG("^N225", 21) OVER (ORDER BY date), 0)) - 1 AS N225_21D_ret,
        ("^N225" / NULLIF(LAG("^N225", 63) OVER (ORDER BY date), 0)) - 1 AS N225_63D_ret,
        ("^N225" / NULLIF(LAG("^N225", 126) OVER (ORDER BY date), 0)) - 1 AS N225_126D_ret,
        ("^N225" / NULLIF(LAG("^N225", 252) OVER (ORDER BY date), 0)) - 1 AS N225_252D_ret,
        ("^N225" / NULLIF(AVG("^N225") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS N225_Dist_SMA200,

        -- Market Indicator: EURUSD=X
        ("EURUSD=X" / NULLIF(LAG("EURUSD=X", 21) OVER (ORDER BY date), 0)) - 1 AS EURUSDX_21D_ret,
        ("EURUSD=X" / NULLIF(LAG("EURUSD=X", 63) OVER (ORDER BY date), 0)) - 1 AS EURUSDX_63D_ret,
        ("EURUSD=X" / NULLIF(LAG("EURUSD=X", 126) OVER (ORDER BY date), 0)) - 1 AS EURUSDX_126D_ret,
        ("EURUSD=X" / NULLIF(LAG("EURUSD=X", 252) OVER (ORDER BY date), 0)) - 1 AS EURUSDX_252D_ret,
        ("EURUSD=X" / NULLIF(AVG("EURUSD=X") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS EURUSDX_Dist_SMA200,

        -- Market Indicator: EEM
        (EEM / NULLIF(LAG(EEM, 21) OVER (ORDER BY date), 0)) - 1 AS EEM_21D_ret,
        (EEM / NULLIF(LAG(EEM, 63) OVER (ORDER BY date), 0)) - 1 AS EEM_63D_ret,
        (EEM / NULLIF(LAG(EEM, 126) OVER (ORDER BY date), 0)) - 1 AS EEM_126D_ret,
        (EEM / NULLIF(LAG(EEM, 252) OVER (ORDER BY date), 0)) - 1 AS EEM_252D_ret,
        (EEM / NULLIF(AVG(EEM) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS EEM_Dist_SMA200,

        -- Market Indicator: VNQ
        (VNQ / NULLIF(LAG(VNQ, 21) OVER (ORDER BY date), 0)) - 1 AS VNQ_21D_ret,
        (VNQ / NULLIF(LAG(VNQ, 63) OVER (ORDER BY date), 0)) - 1 AS VNQ_63D_ret,
        (VNQ / NULLIF(LAG(VNQ, 126) OVER (ORDER BY date), 0)) - 1 AS VNQ_126D_ret,
        (VNQ / NULLIF(LAG(VNQ, 252) OVER (ORDER BY date), 0)) - 1 AS VNQ_252D_ret,
        (VNQ / NULLIF(AVG(VNQ) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS VNQ_Dist_SMA200,

        -- Market Indicator: 7203.T
        ("7203.T" / NULLIF(LAG("7203.T", 21) OVER (ORDER BY date), 0)) - 1 AS 7203_T_21D_ret,
        ("7203.T" / NULLIF(LAG("7203.T", 63) OVER (ORDER BY date), 0)) - 1 AS 7203_T_63D_ret,
        ("7203.T" / NULLIF(LAG("7203.T", 126) OVER (ORDER BY date), 0)) - 1 AS 7203_T_126D_ret,
        ("7203.T" / NULLIF(LAG("7203.T", 252) OVER (ORDER BY date), 0)) - 1 AS 7203_T_252D_ret,
        ("7203.T" / NULLIF(AVG("7203.T") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS 7203_T_Dist_SMA200,

        -- Market Indicator: XLY
        (XLY / NULLIF(LAG(XLY, 21) OVER (ORDER BY date), 0)) - 1 AS XLY_21D_ret,
        (XLY / NULLIF(LAG(XLY, 63) OVER (ORDER BY date), 0)) - 1 AS XLY_63D_ret,
        (XLY / NULLIF(LAG(XLY, 126) OVER (ORDER BY date), 0)) - 1 AS XLY_126D_ret,
        (XLY / NULLIF(LAG(XLY, 252) OVER (ORDER BY date), 0)) - 1 AS XLY_252D_ret,
        (XLY / NULLIF(AVG(XLY) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLY_Dist_SMA200,

        -- Market Indicator: LBR=F
        ("LBR=F" / NULLIF(LAG("LBR=F", 21) OVER (ORDER BY date), 0)) - 1 AS LBRF_21D_ret,
        ("LBR=F" / NULLIF(LAG("LBR=F", 63) OVER (ORDER BY date), 0)) - 1 AS LBRF_63D_ret,
        ("LBR=F" / NULLIF(LAG("LBR=F", 126) OVER (ORDER BY date), 0)) - 1 AS LBRF_126D_ret,
        ("LBR=F" / NULLIF(LAG("LBR=F", 252) OVER (ORDER BY date), 0)) - 1 AS LBRF_252D_ret,
        ("LBR=F" / NULLIF(AVG("LBR=F") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS LBRF_Dist_SMA200,

        -- Market Indicator: SHEL.L
        ("SHEL.L" / NULLIF(LAG("SHEL.L", 21) OVER (ORDER BY date), 0)) - 1 AS SHEL_L_21D_ret,
        ("SHEL.L" / NULLIF(LAG("SHEL.L", 63) OVER (ORDER BY date), 0)) - 1 AS SHEL_L_63D_ret,
        ("SHEL.L" / NULLIF(LAG("SHEL.L", 126) OVER (ORDER BY date), 0)) - 1 AS SHEL_L_126D_ret,
        ("SHEL.L" / NULLIF(LAG("SHEL.L", 252) OVER (ORDER BY date), 0)) - 1 AS SHEL_L_252D_ret,
        ("SHEL.L" / NULLIF(AVG("SHEL.L") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS SHEL_L_Dist_SMA200,

        -- Market Indicator: ZC=F
        ("ZC=F" / NULLIF(LAG("ZC=F", 21) OVER (ORDER BY date), 0)) - 1 AS ZCF_21D_ret,
        ("ZC=F" / NULLIF(LAG("ZC=F", 63) OVER (ORDER BY date), 0)) - 1 AS ZCF_63D_ret,
        ("ZC=F" / NULLIF(LAG("ZC=F", 126) OVER (ORDER BY date), 0)) - 1 AS ZCF_126D_ret,
        ("ZC=F" / NULLIF(LAG("ZC=F", 252) OVER (ORDER BY date), 0)) - 1 AS ZCF_252D_ret,
        ("ZC=F" / NULLIF(AVG("ZC=F") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS ZCF_Dist_SMA200,

        -- Market Indicator: BAS.DE
        ("BAS.DE" / NULLIF(LAG("BAS.DE", 21) OVER (ORDER BY date), 0)) - 1 AS BAS_DE_21D_ret,
        ("BAS.DE" / NULLIF(LAG("BAS.DE", 63) OVER (ORDER BY date), 0)) - 1 AS BAS_DE_63D_ret,
        ("BAS.DE" / NULLIF(LAG("BAS.DE", 126) OVER (ORDER BY date), 0)) - 1 AS BAS_DE_126D_ret,
        ("BAS.DE" / NULLIF(LAG("BAS.DE", 252) OVER (ORDER BY date), 0)) - 1 AS BAS_DE_252D_ret,
        ("BAS.DE" / NULLIF(AVG("BAS.DE") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS BAS_DE_Dist_SMA200,

        -- Market Indicator: ZW=F
        ("ZW=F" / NULLIF(LAG("ZW=F", 21) OVER (ORDER BY date), 0)) - 1 AS ZWF_21D_ret,
        ("ZW=F" / NULLIF(LAG("ZW=F", 63) OVER (ORDER BY date), 0)) - 1 AS ZWF_63D_ret,
        ("ZW=F" / NULLIF(LAG("ZW=F", 126) OVER (ORDER BY date), 0)) - 1 AS ZWF_126D_ret,
        ("ZW=F" / NULLIF(LAG("ZW=F", 252) OVER (ORDER BY date), 0)) - 1 AS ZWF_252D_ret,
        ("ZW=F" / NULLIF(AVG("ZW=F") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS ZWF_Dist_SMA200,

        -- Market Indicator: LQD
        (LQD / NULLIF(LAG(LQD, 21) OVER (ORDER BY date), 0)) - 1 AS LQD_21D_ret,
        (LQD / NULLIF(LAG(LQD, 63) OVER (ORDER BY date), 0)) - 1 AS LQD_63D_ret,
        (LQD / NULLIF(LAG(LQD, 126) OVER (ORDER BY date), 0)) - 1 AS LQD_126D_ret,
        (LQD / NULLIF(LAG(LQD, 252) OVER (ORDER BY date), 0)) - 1 AS LQD_252D_ret,
        (LQD / NULLIF(AVG(LQD) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS LQD_Dist_SMA200,

        -- Market Indicator: RIO.L
        ("RIO.L" / NULLIF(LAG("RIO.L", 21) OVER (ORDER BY date), 0)) - 1 AS RIO_L_21D_ret,
        ("RIO.L" / NULLIF(LAG("RIO.L", 63) OVER (ORDER BY date), 0)) - 1 AS RIO_L_63D_ret,
        ("RIO.L" / NULLIF(LAG("RIO.L", 126) OVER (ORDER BY date), 0)) - 1 AS RIO_L_126D_ret,
        ("RIO.L" / NULLIF(LAG("RIO.L", 252) OVER (ORDER BY date), 0)) - 1 AS RIO_L_252D_ret,
        ("RIO.L" / NULLIF(AVG("RIO.L") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS RIO_L_Dist_SMA200,

        -- Market Indicator: BNDX
        (BNDX / NULLIF(LAG(BNDX, 21) OVER (ORDER BY date), 0)) - 1 AS BNDX_21D_ret,
        (BNDX / NULLIF(LAG(BNDX, 63) OVER (ORDER BY date), 0)) - 1 AS BNDX_63D_ret,
        (BNDX / NULLIF(LAG(BNDX, 126) OVER (ORDER BY date), 0)) - 1 AS BNDX_126D_ret,
        (BNDX / NULLIF(LAG(BNDX, 252) OVER (ORDER BY date), 0)) - 1 AS BNDX_252D_ret,
        (BNDX / NULLIF(AVG(BNDX) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS BNDX_Dist_SMA200,

        -- Market Indicator: IYT
        (IYT / NULLIF(LAG(IYT, 21) OVER (ORDER BY date), 0)) - 1 AS IYT_21D_ret,
        (IYT / NULLIF(LAG(IYT, 63) OVER (ORDER BY date), 0)) - 1 AS IYT_63D_ret,
        (IYT / NULLIF(LAG(IYT, 126) OVER (ORDER BY date), 0)) - 1 AS IYT_126D_ret,
        (IYT / NULLIF(LAG(IYT, 252) OVER (ORDER BY date), 0)) - 1 AS IYT_252D_ret,
        (IYT / NULLIF(AVG(IYT) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS IYT_Dist_SMA200,

        -- Market Indicator: XLE
        (XLE / NULLIF(LAG(XLE, 21) OVER (ORDER BY date), 0)) - 1 AS XLE_21D_ret,
        (XLE / NULLIF(LAG(XLE, 63) OVER (ORDER BY date), 0)) - 1 AS XLE_63D_ret,
        (XLE / NULLIF(LAG(XLE, 126) OVER (ORDER BY date), 0)) - 1 AS XLE_126D_ret,
        (XLE / NULLIF(LAG(XLE, 252) OVER (ORDER BY date), 0)) - 1 AS XLE_252D_ret,
        (XLE / NULLIF(AVG(XLE) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLE_Dist_SMA200,

        -- Market Indicator: SPY
        (SPY / NULLIF(LAG(SPY, 21) OVER (ORDER BY date), 0)) - 1 AS SPY_21D_ret,
        (SPY / NULLIF(LAG(SPY, 63) OVER (ORDER BY date), 0)) - 1 AS SPY_63D_ret,
        (SPY / NULLIF(LAG(SPY, 126) OVER (ORDER BY date), 0)) - 1 AS SPY_126D_ret,
        (SPY / NULLIF(LAG(SPY, 252) OVER (ORDER BY date), 0)) - 1 AS SPY_252D_ret,
        (SPY / NULLIF(AVG(SPY) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS SPY_Dist_SMA200,

        -- Market Indicator: XLF
        (XLF / NULLIF(LAG(XLF, 21) OVER (ORDER BY date), 0)) - 1 AS XLF_21D_ret,
        (XLF / NULLIF(LAG(XLF, 63) OVER (ORDER BY date), 0)) - 1 AS XLF_63D_ret,
        (XLF / NULLIF(LAG(XLF, 126) OVER (ORDER BY date), 0)) - 1 AS XLF_126D_ret,
        (XLF / NULLIF(LAG(XLF, 252) OVER (ORDER BY date), 0)) - 1 AS XLF_252D_ret,
        (XLF / NULLIF(AVG(XLF) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLF_Dist_SMA200,

        -- Market Indicator: SMH
        (SMH / NULLIF(LAG(SMH, 21) OVER (ORDER BY date), 0)) - 1 AS SMH_21D_ret,
        (SMH / NULLIF(LAG(SMH, 63) OVER (ORDER BY date), 0)) - 1 AS SMH_63D_ret,
        (SMH / NULLIF(LAG(SMH, 126) OVER (ORDER BY date), 0)) - 1 AS SMH_126D_ret,
        (SMH / NULLIF(LAG(SMH, 252) OVER (ORDER BY date), 0)) - 1 AS SMH_252D_ret,
        (SMH / NULLIF(AVG(SMH) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS SMH_Dist_SMA200,

        -- Market Indicator: 8035.T
        ("8035.T" / NULLIF(LAG("8035.T", 21) OVER (ORDER BY date), 0)) - 1 AS 8035_T_21D_ret,
        ("8035.T" / NULLIF(LAG("8035.T", 63) OVER (ORDER BY date), 0)) - 1 AS 8035_T_63D_ret,
        ("8035.T" / NULLIF(LAG("8035.T", 126) OVER (ORDER BY date), 0)) - 1 AS 8035_T_126D_ret,
        ("8035.T" / NULLIF(LAG("8035.T", 252) OVER (ORDER BY date), 0)) - 1 AS 8035_T_252D_ret,
        ("8035.T" / NULLIF(AVG("8035.T") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS 8035_T_Dist_SMA200,

        -- Market Indicator: XLK
        (XLK / NULLIF(LAG(XLK, 21) OVER (ORDER BY date), 0)) - 1 AS XLK_21D_ret,
        (XLK / NULLIF(LAG(XLK, 63) OVER (ORDER BY date), 0)) - 1 AS XLK_63D_ret,
        (XLK / NULLIF(LAG(XLK, 126) OVER (ORDER BY date), 0)) - 1 AS XLK_126D_ret,
        (XLK / NULLIF(LAG(XLK, 252) OVER (ORDER BY date), 0)) - 1 AS XLK_252D_ret,
        (XLK / NULLIF(AVG(XLK) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS XLK_Dist_SMA200,

        -- Market Indicator: JPY=X
        ("JPY=X" / NULLIF(LAG("JPY=X", 21) OVER (ORDER BY date), 0)) - 1 AS JPYX_21D_ret,
        ("JPY=X" / NULLIF(LAG("JPY=X", 63) OVER (ORDER BY date), 0)) - 1 AS JPYX_63D_ret,
        ("JPY=X" / NULLIF(LAG("JPY=X", 126) OVER (ORDER BY date), 0)) - 1 AS JPYX_126D_ret,
        ("JPY=X" / NULLIF(LAG("JPY=X", 252) OVER (ORDER BY date), 0)) - 1 AS JPYX_252D_ret,
        ("JPY=X" / NULLIF(AVG("JPY=X") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS JPYX_Dist_SMA200,

        -- Market Indicator: 9984.T
        ("9984.T" / NULLIF(LAG("9984.T", 21) OVER (ORDER BY date), 0)) - 1 AS 9984_T_21D_ret,
        ("9984.T" / NULLIF(LAG("9984.T", 63) OVER (ORDER BY date), 0)) - 1 AS 9984_T_63D_ret,
        ("9984.T" / NULLIF(LAG("9984.T", 126) OVER (ORDER BY date), 0)) - 1 AS 9984_T_126D_ret,
        ("9984.T" / NULLIF(LAG("9984.T", 252) OVER (ORDER BY date), 0)) - 1 AS 9984_T_252D_ret,
        ("9984.T" / NULLIF(AVG("9984.T") OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS 9984_T_Dist_SMA200,

        -- Market Indicator: ratio_copper_gold
        ratio_copper_gold AS ratio_copper_gold_Level,
        ratio_copper_gold - LAG(ratio_copper_gold, 21) OVER (ORDER BY date) AS ratio_copper_gold_21D_diff,
        ratio_copper_gold - LAG(ratio_copper_gold, 63) OVER (ORDER BY date) AS ratio_copper_gold_63D_diff,
        ratio_copper_gold - LAG(ratio_copper_gold, 126) OVER (ORDER BY date) AS ratio_copper_gold_126D_diff,
        ratio_copper_gold - LAG(ratio_copper_gold, 252) OVER (ORDER BY date) AS ratio_copper_gold_252D_diff,
        ratio_copper_gold - AVG(ratio_copper_gold) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_copper_gold_Dist_SMA200,

        -- Market Indicator: ratio_credit_spread
        ratio_credit_spread AS ratio_credit_spread_Level,
        ratio_credit_spread - LAG(ratio_credit_spread, 21) OVER (ORDER BY date) AS ratio_credit_spread_21D_diff,
        ratio_credit_spread - LAG(ratio_credit_spread, 63) OVER (ORDER BY date) AS ratio_credit_spread_63D_diff,
        ratio_credit_spread - LAG(ratio_credit_spread, 126) OVER (ORDER BY date) AS ratio_credit_spread_126D_diff,
        ratio_credit_spread - LAG(ratio_credit_spread, 252) OVER (ORDER BY date) AS ratio_credit_spread_252D_diff,
        ratio_credit_spread - AVG(ratio_credit_spread) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_credit_spread_Dist_SMA200,
        (ratio_credit_spread - AVG(ratio_credit_spread) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV(ratio_credit_spread) OVER (ORDER BY date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW), 0) AS ratio_credit_spread_Roll_ZScore_2Y,

        -- Market Indicator: ratio_consumer_risk
        ratio_consumer_risk AS ratio_consumer_risk_Level,
        ratio_consumer_risk - LAG(ratio_consumer_risk, 21) OVER (ORDER BY date) AS ratio_consumer_risk_21D_diff,
        ratio_consumer_risk - LAG(ratio_consumer_risk, 63) OVER (ORDER BY date) AS ratio_consumer_risk_63D_diff,
        ratio_consumer_risk - LAG(ratio_consumer_risk, 126) OVER (ORDER BY date) AS ratio_consumer_risk_126D_diff,
        ratio_consumer_risk - LAG(ratio_consumer_risk, 252) OVER (ORDER BY date) AS ratio_consumer_risk_252D_diff,
        ratio_consumer_risk - AVG(ratio_consumer_risk) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_consumer_risk_Dist_SMA200,

        -- Market Indicator: ratio_risk_on_off
        ratio_risk_on_off AS ratio_risk_on_off_Level,
        ratio_risk_on_off - LAG(ratio_risk_on_off, 21) OVER (ORDER BY date) AS ratio_risk_on_off_21D_diff,
        ratio_risk_on_off - LAG(ratio_risk_on_off, 63) OVER (ORDER BY date) AS ratio_risk_on_off_63D_diff,
        ratio_risk_on_off - LAG(ratio_risk_on_off, 126) OVER (ORDER BY date) AS ratio_risk_on_off_126D_diff,
        ratio_risk_on_off - LAG(ratio_risk_on_off, 252) OVER (ORDER BY date) AS ratio_risk_on_off_252D_diff,
        ratio_risk_on_off - AVG(ratio_risk_on_off) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_risk_on_off_Dist_SMA200,

        -- Market Indicator: ratio_tech_dominance
        ratio_tech_dominance AS ratio_tech_dominance_Level,
        ratio_tech_dominance - LAG(ratio_tech_dominance, 21) OVER (ORDER BY date) AS ratio_tech_dominance_21D_diff,
        ratio_tech_dominance - LAG(ratio_tech_dominance, 63) OVER (ORDER BY date) AS ratio_tech_dominance_63D_diff,
        ratio_tech_dominance - LAG(ratio_tech_dominance, 126) OVER (ORDER BY date) AS ratio_tech_dominance_126D_diff,
        ratio_tech_dominance - LAG(ratio_tech_dominance, 252) OVER (ORDER BY date) AS ratio_tech_dominance_252D_diff,
        ratio_tech_dominance - AVG(ratio_tech_dominance) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_tech_dominance_Dist_SMA200,

        -- Market Indicator: ratio_intl_vs_us_bonds
        ratio_intl_vs_us_bonds AS ratio_intl_vs_us_bonds_Level,
        ratio_intl_vs_us_bonds - LAG(ratio_intl_vs_us_bonds, 21) OVER (ORDER BY date) AS ratio_intl_vs_us_bonds_21D_diff,
        ratio_intl_vs_us_bonds - LAG(ratio_intl_vs_us_bonds, 63) OVER (ORDER BY date) AS ratio_intl_vs_us_bonds_63D_diff,
        ratio_intl_vs_us_bonds - LAG(ratio_intl_vs_us_bonds, 126) OVER (ORDER BY date) AS ratio_intl_vs_us_bonds_126D_diff,
        ratio_intl_vs_us_bonds - LAG(ratio_intl_vs_us_bonds, 252) OVER (ORDER BY date) AS ratio_intl_vs_us_bonds_252D_diff,
        ratio_intl_vs_us_bonds - AVG(ratio_intl_vs_us_bonds) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ratio_intl_vs_us_bonds_Dist_SMA200

    FROM base_data
),

-- 2. TARGET ASSET FEATURE ENGINEERING (from features.py)
target_features AS (
    SELECT 
        *,
        
        -- Momentum / Returns for the target asset
        (Close / NULLIF(LAG(Close, 21) OVER (ORDER BY date), 0)) - 1  AS Ret_21D,
        (Close / NULLIF(LAG(Close, 63) OVER (ORDER BY date), 0)) - 1  AS Ret_63D,
        (Close / NULLIF(LAG(Close, 126) OVER (ORDER BY date), 0)) - 1 AS Ret_126D,
        (Close / NULLIF(LAG(Close, 252) OVER (ORDER BY date), 0)) - 1 AS Ret_252D,

        -- Moving Average Distances
        (Close / NULLIF(AVG(Close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 0)) - 1 AS Dist_SMA_50,
        (Close / NULLIF(AVG(Close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW), 0)) - 1 AS Dist_SMA_200,

        -- Volatility (Rolling standard deviation of daily returns over 21 days)
        STDDEV( (Close / NULLIF(LAG(Close, 1) OVER (ORDER BY date), 0)) - 1 ) 
            OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS Vol_21D,

        -- Target Variable Calculation (Future Return over horizon_days, e.g., 126)
        (LEAD(Close, 126) OVER (ORDER BY date) / NULLIF(Close, 0)) - 1 AS Future_Return
    FROM macro_features
)

-- 3. FINAL OUTPUT WITH BINARY TARGET
SELECT 
    *,
    CASE 
        WHEN Future_Return >= 0.15 THEN 1.0 
        WHEN Future_Return IS NULL THEN NULL 
        ELSE 0.0 
    END AS Target
FROM target_features;

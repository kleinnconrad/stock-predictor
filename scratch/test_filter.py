import pandas as pd
df = pd.read_csv('data/raw/t7-xetr-allTradableInstruments.csv', sep=';', skiprows=2, low_memory=False)
print(f'Total: {len(df)}')
de_shares = df[(df['Instrument Type'] == 'CS') & (df['Product Assignment Group Description'] == 'DEUTSCHLAND')]
print(f'DE Shares: {len(de_shares)}')
liquid_de = de_shares[de_shares['Regulatory Liquid Instrument'] == 'Y']
print(f'Liquid DE Shares: {len(liquid_de)}')


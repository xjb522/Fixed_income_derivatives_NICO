import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tidyfinance as td

DD = td.download_data(
    domain="stock_prices",
    symbols="^IXIC",
    start_date="2015-01-01",
    end_date="2025-01-01"
)

dd1 = pd.pivot(DD, index="date", columns="symbol", values="close")

print(dd1.isna().count())

dd1.dropna()

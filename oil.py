"""
Регрессия Лассо котировки акций Роснефть к Brent
date	price
01.01.2022	77,78
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ----------------------------
# 1. Загрузка данных
# ----------------------------
stocks = pd.read_csv("stocks.csv", sep="\t")
stocks.rename(columns=lambda x: x.strip('<>'), inplace=True)
stocks['DATE'] = pd.to_datetime(stocks['DATE'], format='%Y%m%d')
stocks['year_month'] = stocks['DATE'].dt.to_period('M').dt.to_timestamp()
stocks_monthly = stocks.groupby('year_month')['CLOSE'].mean().reset_index()

# Brent
brent = pd.read_csv("oil.csv", sep="\t")  # CSV: date price
brent['price'] = brent['price'].str.replace(',', '.').astype(float)
brent['date'] = pd.to_datetime(brent['date'], format='%d.%m.%Y')
brent['year_month'] = brent['date'].dt.to_period('M').dt.to_timestamp()
brent_monthly = brent.groupby('year_month')['price'].mean().reset_index()

# ----------------------------
# 2. Объединяем данные
# ----------------------------
data = pd.merge(stocks_monthly, brent_monthly, on='year_month', how='inner')
data.rename(columns={'CLOSE':'stock_close', 'price':'brent_price'}, inplace=True)

# ----------------------------
# 3. Модель Lasso
# ----------------------------
X = data[['brent_price']].values
y = data['stock_close'].values

model = Lasso(alpha=0.1)
model.fit(X, y)
data['predicted'] = model.predict(X)

# Метрики
r2 = r2_score(y, data['predicted'])
mape = mean_absolute_percentage_error(y, data['predicted']) * 100
print(f"R^2: {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
print(f"Коэффициенты: {model.coef_}")
print(f"Свободный член: {model.intercept_:.2f}")

# ----------------------------
# 4. Прогноз Brent на 2025
# ----------------------------
brent_sorted = brent_monthly.sort_values('year_month')
brent_sorted['month_index'] = np.arange(len(brent_sorted))
brent_model = LinearRegression()
brent_model.fit(brent_sorted[['month_index']], brent_sorted['brent_price'] if 'brent_price' in brent_sorted else brent_sorted['price'])

future_months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_index = np.arange(len(brent_sorted), len(brent_sorted) + len(future_months))
future_X = future_index.reshape(-1,1)
predicted_brent = brent_model.predict(future_X)

# Прогноз цены акции с Lasso
future_X_lasso = predicted_brent.reshape(-1,1)
predicted_stock_price = model.predict(future_X_lasso)

forecast_2025 = pd.DataFrame({
    'month': future_months,
    'predicted_brent': predicted_brent,
    'predicted_stock_price': predicted_stock_price
})

print("\nПрогноз на 2025 год:")
print(forecast_2025)

# ----------------------------
# 5. График котировок + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['stock_close'], marker='o', color='blue', label='Фактическая цена')
plt.plot(forecast_2025['month'], forecast_2025['predicted_stock_price'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.plot(data['year_month'], data['predicted'], color='red', linestyle='-', label='Lasso (fit)')
plt.xlabel('Месяц')
plt.ylabel('Цена акции (руб)')
plt.title('Котировки акции ROSN и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 6. График Brent + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['brent_price'], marker='o', color='orange', label='Фактическая цена Brent')
plt.plot(forecast_2025['month'], forecast_2025['predicted_brent'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.xlabel('Месяц')
plt.ylabel('Цена Brent (USD)')
plt.title('Котировки Brent и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

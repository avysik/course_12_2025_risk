"""
Регрессия Лассо котировки акций Роснефть к курсу USD/RUB
date	price
10.11.2020	76,95
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ----------------------------
# 1. Загрузка данных
# ----------------------------
stocks = pd.read_csv("stocks.csv", sep="\t")
stocks.rename(columns=lambda x: x.strip('<>'), inplace=True)
stocks['DATE'] = pd.to_datetime(stocks['DATE'], format='%Y%m%d')
stocks['year_month'] = stocks['DATE'].dt.to_period('M').dt.to_timestamp()
stocks_monthly = stocks.groupby('year_month')['CLOSE'].mean().reset_index()

# USD/RUB
usd = pd.read_csv("dollar.csv", sep="\t")  # CSV: date price
usd['price'] = usd['price'].str.replace(',', '.').astype(float)
usd['date'] = pd.to_datetime(usd['date'], format='%d.%m.%Y')
usd['year_month'] = usd['date'].dt.to_period('M').dt.to_timestamp()
usd_monthly = usd.groupby('year_month')['price'].mean().reset_index()

# ----------------------------
# 2. Объединяем данные
# ----------------------------
data = pd.merge(stocks_monthly, usd_monthly, on='year_month', how='inner')
data.rename(columns={'CLOSE':'stock_close', 'price':'usd_rate'}, inplace=True)

# ----------------------------
# 3. Модель Lasso
# ----------------------------
X = data[['usd_rate']].values
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
# 4. Прогноз USD/RUB на 2025
# ----------------------------
usd_sorted = usd_monthly.sort_values('year_month').copy()
usd_sorted['month_index'] = np.arange(len(usd_sorted))
usd_model = LinearRegression()
usd_model.fit(usd_sorted[['month_index']], usd_sorted['price'])

future_months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_index = np.arange(len(usd_sorted), len(usd_sorted) + len(future_months)).reshape(-1,1)
predicted_usd = usd_model.predict(future_index)

# Прогноз цены акции с Lasso
future_X_lasso = predicted_usd.reshape(-1,1)
predicted_stock_price = model.predict(future_X_lasso)

forecast_2025 = pd.DataFrame({
    'month': future_months,
    'predicted_usd': predicted_usd,
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
# 6. График курса USD/RUB + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['usd_rate'], marker='o', color='orange', label='Фактический курс USD/RUB')
plt.plot(forecast_2025['month'], forecast_2025['predicted_usd'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.xlabel('Месяц')
plt.ylabel('Курс USD/RUB')
plt.title('Курс USD/RUB и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

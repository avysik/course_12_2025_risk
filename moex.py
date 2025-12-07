"""
Регрессия Лассо котировки акций Роснефть к индексу MOEX
date	price
30.12.2024	2 883,04
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

# MOEX
moex = pd.read_csv("moex.csv", sep="\t")  # CSV: date price
# Убираем пробелы и заменяем запятую на точку
moex['price'] = moex['price'].str.replace(' ', '').str.replace(',', '.').astype(float)
moex['date'] = pd.to_datetime(moex['date'], format='%d.%m.%Y')
moex['year_month'] = moex['date'].dt.to_period('M').dt.to_timestamp()
moex_monthly = moex.groupby('year_month')['price'].mean().reset_index()

# ----------------------------
# 2. Объединяем данные
# ----------------------------
data = pd.merge(stocks_monthly, moex_monthly, on='year_month', how='inner')
data.rename(columns={'CLOSE':'stock_close', 'price':'moex_index'}, inplace=True)

# ----------------------------
# 3. Модель Lasso
# ----------------------------
X = data[['moex_index']].values
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
# 4. Прогноз MOEX на 2025
# ----------------------------
moex_sorted = moex_monthly.sort_values('year_month').copy()
moex_sorted['month_index'] = np.arange(len(moex_sorted))
moex_model = LinearRegression()
moex_model.fit(moex_sorted[['month_index']], moex_sorted['moex_index'] if 'moex_index' in moex_sorted else moex_sorted['price'])

future_months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_index = np.arange(len(moex_sorted), len(moex_sorted) + len(future_months)).reshape(-1,1)
predicted_moex = moex_model.predict(future_index)

# Прогноз цены акции с Lasso
future_X_lasso = predicted_moex.reshape(-1,1)
predicted_stock_price = model.predict(future_X_lasso)

forecast_2025 = pd.DataFrame({
    'month': future_months,
    'predicted_moex': predicted_moex,
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
# 6. График индекса MOEX + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['moex_index'], marker='o', color='orange', label='Фактический индекс MOEX')
plt.plot(forecast_2025['month'], forecast_2025['predicted_moex'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.xlabel('Месяц')
plt.ylabel('Индекс MOEX')
plt.title('Индекс MOEX и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

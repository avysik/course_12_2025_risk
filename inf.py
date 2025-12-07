"""
Регрессия Лассо котировки акций Роснефть к инфляции
data	inf
12.2024	9,52
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

# ----------------------------
# Загрузка инфляции
# ----------------------------
inf_data = pd.read_csv("inf.csv", sep="\t")  # табуляция
inf_data['inf'] = inf_data['inf'].astype(str).str.replace(',', '.').astype(float)
inf_data['data'] = inf_data['data'].astype(str).str.strip()
inf_data = inf_data[inf_data['data'].str.len() == 7].copy()  # оставляем только корректные строки
inf_data['year_month'] = pd.to_datetime(inf_data['data'], format='%m.%Y')
inf_monthly = inf_data.groupby('year_month')['inf'].mean().reset_index()

# ----------------------------
# 2. Объединяем данные
# ----------------------------
data = pd.merge(stocks_monthly, inf_monthly, on='year_month', how='inner')
data.rename(columns={'CLOSE':'stock_close', 'inf':'inflation'}, inplace=True)

# ----------------------------
# 3. Модель Lasso
# ----------------------------
X = data[['inflation']].values  # используем инфляцию
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
# 4. Прогноз инфляции на 2025
# ----------------------------
inf_monthly_sorted = inf_monthly.sort_values('year_month')
inf_monthly_sorted['month_index'] = np.arange(len(inf_monthly_sorted))
inf_model = LinearRegression()
inf_model.fit(inf_monthly_sorted[['month_index']], inf_monthly_sorted['inf'])

future_months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_index = np.arange(len(inf_monthly_sorted), len(inf_monthly_sorted) + len(future_months))
future_X = pd.DataFrame(future_index, columns=['month_index'])
predicted_inflation = inf_model.predict(future_X)

# Прогноз цены акции с Lasso
future_X_lasso = predicted_inflation.reshape(-1,1)
predicted_stock_price = model.predict(future_X_lasso)

forecast_2025 = pd.DataFrame({
    'month': future_months,
    'predicted_inflation': predicted_inflation,
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
# 6. График инфляции + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['inflation'], marker='o', color='orange', label='Фактическая инфляция')
plt.plot(forecast_2025['month'], forecast_2025['predicted_inflation'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.xlabel('Месяц')
plt.ylabel('Инфляция (%)')
plt.title('Инфляция и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

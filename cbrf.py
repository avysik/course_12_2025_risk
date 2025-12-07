"""
Регрессия Лассо котировки акций Роснефть к ключевой ставке цбрф
date	price
30.12.2024	21,00
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

cb = pd.read_csv("cbrf.csv", sep="\t")
cb['price'] = cb['price'].str.replace(',', '.').astype(float)
cb['date'] = pd.to_datetime(cb['date'], format='%d.%m.%Y')
cb['year_month'] = cb['date'].dt.to_period('M').dt.to_timestamp()
cb_monthly = cb.groupby('year_month')['price'].mean().reset_index()

# ----------------------------
# 2. Объединяем данные
# ----------------------------
data = pd.merge(stocks_monthly, cb_monthly, on='year_month', how='inner')
data.rename(columns={'CLOSE':'stock_close', 'price':'cb_rate'}, inplace=True)

# ----------------------------
# 3. Модель Lasso
# ----------------------------
X = data[['cb_rate']].values  # оставляем только ставку ЦБ
y = data['stock_close'].values

model = Lasso(alpha=0.1)  # alpha можно подбирать
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
# 4. Прогноз ставки ЦБ на 2025
# ----------------------------
cb_monthly_sorted = cb_monthly.sort_values('year_month')
cb_monthly_sorted['month_index'] = np.arange(len(cb_monthly_sorted))
cb_model = LinearRegression()
cb_model.fit(cb_monthly_sorted[['month_index']], cb_monthly_sorted['price'])

future_months = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future_index = np.arange(len(cb_monthly_sorted), len(cb_monthly_sorted) + len(future_months))
future_X = pd.DataFrame(future_index, columns=['month_index'])
predicted_cb_rate = cb_model.predict(future_X)

# Прогноз цены акции с Lasso
future_X = predicted_cb_rate.reshape(-1,1)
predicted_stock_price = model.predict(future_X)

forecast_2025 = pd.DataFrame({
    'month': future_months,
    'predicted_cb_rate': predicted_cb_rate,
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
# 6. График ставки ЦБ + прогноз
# ----------------------------
plt.figure(figsize=(14,7))
plt.plot(data['year_month'], data['cb_rate'], marker='o', color='orange', label='Фактическая ставка')
plt.plot(forecast_2025['month'], forecast_2025['predicted_cb_rate'], marker='o', linestyle='--', color='green', label='Прогноз 2025')
plt.xlabel('Месяц')
plt.ylabel('Ставка ЦБ (%)')
plt.title('Ставка ЦБ и прогноз на 2025 год')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

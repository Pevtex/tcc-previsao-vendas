# Bibliotecas
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import statsmodels.api as sm

# ----------------------------
# 1. Carregamento dos Dados
# ----------------------------

path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))

# ----------------------------
# 2. Pré-processamento
# ----------------------------

merged = order_items.merge(products, on="product_id").merge(orders, on="order_id")
merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
merged['data_pedido'] = merged['order_purchase_timestamp'].dt.date

# Agregar vendas por dia
vendas_diarias = merged.groupby('data_pedido').size().reset_index(name='qtd_vendas')
vendas_diarias['data_pedido'] = pd.to_datetime(vendas_diarias['data_pedido'])

# Criar variáveis temporais
vendas_diarias['tendencia'] = range(len(vendas_diarias))
vendas_diarias['dia_semana'] = vendas_diarias['data_pedido'].dt.dayofweek
vendas_diarias['semana_ano'] = vendas_diarias['data_pedido'].dt.isocalendar().week
vendas_diarias['lag1'] = vendas_diarias['qtd_vendas'].shift(1)
vendas_diarias['lag7'] = vendas_diarias['qtd_vendas'].shift(7)
vendas_diarias['media_movel_7'] = vendas_diarias['qtd_vendas'].rolling(window=7).mean()

# Remover primeiros dias com NaN
vendas_diarias.dropna(inplace=True)

# Visualização das vendas diárias
plt.figure(figsize=(14, 6))
sns.lineplot(data=vendas_diarias, x='data_pedido', y='qtd_vendas')
plt.title("Vendas Diárias Agregadas (Total)")
plt.xlabel("Data")
plt.ylabel("Quantidade de Vendas")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 3. Feriados
# ----------------------------

feriados_fixos = {"01-01": "Ano Novo", "04-21": "Tiradentes", "05-01": "Dia do Trabalho",
                  "09-07": "Independência", "10-12": "Nossa Senhora", "11-02": "Finados",
                  "11-15": "Proclamação", "12-25": "Natal"}

feriados_moveis = {"2017-02-28": "Carnaval", "2017-04-14": "Sexta-feira Santa",
                   "2017-04-16": "Páscoa", "2017-11-24": "Black Friday",
                   "2018-02-13": "Carnaval", "2018-03-30": "Sexta-feira Santa",
                   "2018-04-01": "Páscoa", "2018-11-23": "Black Friday"}

def identificar_feriado(data):
    data_str = data.strftime("%Y-%m-%d")
    mmdd = data.strftime("%m-%d")
    if data_str in feriados_moveis:
        return feriados_moveis[data_str]
    elif mmdd in feriados_fixos:
        return feriados_fixos[mmdd]
    else:
        return "Nenhum"

vendas_diarias['tipo_feriado'] = vendas_diarias['data_pedido'].apply(identificar_feriado)
vendas_dummies = pd.get_dummies(vendas_diarias, columns=['tipo_feriado'], drop_first=True)

# ----------------------------
# 4. Separar Variáveis
# ----------------------------

variaveis_explicativas = ['tendencia', 'dia_semana', 'semana_ano', 'lag1', 'lag7', 'media_movel_7'] + \
                         [col for col in vendas_dummies.columns if col.startswith("tipo_feriado_")]

X = vendas_dummies[variaveis_explicativas]
y = vendas_dummies['qtd_vendas']

# ----------------------------
# 5. Separação Treino e Teste
# ----------------------------

corte = int(len(vendas_dummies) * 0.8)
X_train, X_test = X.iloc[:corte], X.iloc[corte:]
y_train, y_test = y.iloc[:corte], y.iloc[corte:]

# ----------------------------
# 6. Modelos de Previsão
# ----------------------------

# --- Regressão Linear
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# --- Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# --- XGBoost
xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=100).fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# --- Rede Neural (com early stopping)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
nn_model.fit(X_train_scaled, y_train, epochs=300, batch_size=16, verbose=0, callbacks=[early_stop])
y_pred_nn = nn_model.predict(X_test_scaled).flatten()

# --- ARIMA
serie_treino = y_train.values
modelo_arima = sm.tsa.ARIMA(serie_treino, order=(7,1,1))
resultado_arima = modelo_arima.fit()
y_pred_arima = resultado_arima.forecast(steps=len(y_test))

# ----------------------------
# 7. Avaliação
# ----------------------------

def avaliar(y_true, y_pred, nome):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{nome}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape*100:.2f}%")

avaliar(y_test, y_pred_lr, "Regressão Linear")
avaliar(y_test, y_pred_rf, "Random Forest")
avaliar(y_test, y_pred_xgb, "XGBoost")
avaliar(y_test, y_pred_nn, "Rede Neural")
avaliar(y_test, y_pred_arima, "ARIMA")

# ----------------------------
# 8. Gráficos
# ----------------------------

df_resultado = pd.DataFrame({
    'Data': vendas_dummies['data_pedido'].iloc[corte:].values,
    'Real': y_test.values,
    'Regressão Linear': y_pred_lr,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb,
    'Rede Neural': y_pred_nn,
    'ARIMA': y_pred_arima
})

print (df_resultado.head())

#Gárifo comparativo geral
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['Regressão Linear'], label="Regressão Linear", linestyle='--')
plt.plot(df_resultado['Data'], df_resultado['Random Forest'], label="Random Forest", linestyle='--')
plt.plot(df_resultado['Data'], df_resultado['XGBoost'], label="XGBoost", linestyle='--')
plt.plot(df_resultado['Data'], df_resultado['Rede Neural'], label="Rede Neural", linestyle='--')
plt.plot(df_resultado['Data'], df_resultado['ARIMA'], label="ARIMA", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.savefig("comparacao_modelos_final.png")
plt.show()

#Gárifo regressão linear
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['Regressão Linear'], label="Regressão Linear", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.savefig("comparacao_modelos_final.png")
plt.show()


#Gárifo Random Forest
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['Random Forest'], label="Random Forest", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.show()

#Gárifo XGBoost
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['XGBoost'], label="XGBoost", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.show()

#Gárifo Rede neural
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['Rede Neural'], label="Rede Neural", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.show()

#Gárifo ARIMA
plt.figure(figsize=(15,6))
plt.plot(df_resultado['Data'], df_resultado['Real'], label="Vendas Reais", linewidth=3)
plt.plot(df_resultado['Data'], df_resultado['ARIMA'], label="ARIMA", linestyle='--')
plt.legend()
plt.title("Comparação de Modelos - Previsão de Vendas Diárias")
plt.xlabel("Data")
plt.ylabel("Vendas")
plt.grid(True)
plt.tight_layout()
plt.show()



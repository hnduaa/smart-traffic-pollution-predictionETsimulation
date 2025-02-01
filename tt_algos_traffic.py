import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Charger les données
data = pd.read_csv('traffic.csv')

# Préparation des données
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour
data['Day'] = data['DateTime'].dt.day
data['Month'] = data['DateTime'].dt.month
data['Weekday'] = data['DateTime'].dt.weekday

# Supprimer les colonnes inutiles
data.drop(['DateTime', 'ID'], axis=1, inplace=True)

# Gestion des valeurs manquantes
data.dropna(inplace=True)

# Séparer les caractéristiques (X) et la cible (y)
X = data[['Junction', 'Hour', 'Day', 'Month', 'Weekday']]
y = data['Vehicles']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fonction pour afficher les métriques de performance
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2: {r2:.2f}")
    print("-" * 40)

# 1. Régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lin, "Régression linéaire")

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, "Random Forest")

# 3. XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# 4. KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
evaluate_model(y_test, y_pred_knn, "KNN")

# 5. Réseau de neurones artificiels (ANN)
ann = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Corrige l'erreur d'avertissement
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=1)
y_pred_ann = ann.predict(X_test_scaled).flatten()
evaluate_model(y_test, y_pred_ann, "ANN")

# 6. Clustering avec K-Means
clustering_data = data[['Junction', 'Vehicles']].copy()
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(clustering_data)
print("K-Means Clustering - Done")

# Identifier les "Top Priority Junctions"
junction_priority = data.groupby('Junction')['Vehicles'].mean().sort_values(ascending=False)
top_priority_junctions = junction_priority.head(3)
print("Top Priority Junctions for Traffic Light Installation:")
print(top_priority_junctions)

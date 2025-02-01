import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
data = pd.read_csv('pollution.csv', encoding='latin1')  # Exemple avec latin1
data.head()
# Nettoyage des données
data.replace('NA', np.nan, inplace=True)
data.dropna(inplace=True)  # Supprime les lignes avec des valeurs manquantes

# Sélection des colonnes pertinentes
columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5', 'type']
data = data[columns]

# Encodage des variables catégoriques
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Séparer les caractéristiques (X) et la cible (y)
X = data.drop('pm2_5', axis=1)  # Prédire PM2.5
y = data['pm2_5']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fonction d'évaluation
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

# 2. Random Forest avec optimisation
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_squared_error', verbose=0)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, "Random Forest")

# 3. XGBoost avec optimisation
xgb = XGBRegressor(random_state=42)
param_grid_xgb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_squared_error', verbose=0)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# 4. KNN avec optimisation
knn = KNeighborsRegressor()
param_grid_knn = {'n_neighbors': [3, 5, 7, 10]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='neg_mean_squared_error', verbose=0)
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)
evaluate_model(y_test, y_pred_knn, "KNN")

# 5. Réseau de neurones artificiels (ANN)
ann = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=1)
y_pred_ann = ann.predict(X_test_scaled).flatten()
evaluate_model(y_test, y_pred_ann, "ANN")

# 6. Clustering avec K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
print("K-Means Clustering - Done")

# Visualisation des clusters (facultative)
print("Clusters formés :", data['Cluster'].value_counts())


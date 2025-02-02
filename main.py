import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse et Prédiction", layout="wide")
st.title("🌍 Analyse et Prédiction - Trafic et Pollution de l'Air")

# Menu de navigation dans la barre latérale
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Choisissez le type d'analyse :", ["🚦 Analyse du Trafic", "🌍 Analyse de la Pollution de l'Air", "🎯 Prédictions"])

# Fonctions pour l'analyse du trafic
def traffic_analysis():
    st.header("🚦 Analyse du Trafic")
    # Code de l'analyse du trafic
    # Sidebar for user interaction
st.sidebar.header("🔍 Data & Settings")

# Load Dataset
data = pd.read_csv('traffic.csv')

# Validate DateTime Column
if 'DateTime' in data.columns:
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Hour'] = data['DateTime'].dt.hour
    data['Day'] = data['DateTime'].dt.day
    data['Month'] = data['DateTime'].dt.month
    data['Weekday'] = data['DateTime'].dt.weekday
else:
    st.sidebar.error("⚠️ 'DateTime' column not found. Please check the dataset.")

# Drop unnecessary columns
if 'ID' in data.columns:
    data.drop(columns=['DateTime', 'ID'], inplace=True)

# Check for missing values
if data.isnull().sum().sum() > 0:
    st.sidebar.warning("⚠️ There are missing values in the dataset.")
    st.write("Missing Values:", data.isnull().sum())

# **Traffic Density Analysis**
st.subheader("📊 Traffic Density Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
hourly_traffic = data.groupby('Hour')['Vehicles'].mean()
hourly_traffic.plot(kind='bar', ax=ax, color='royalblue', alpha=0.7)
ax.set_title('Average Vehicles by Hour')
ax.set_xlabel('Hour')
ax.set_ylabel('Average Vehicles')
st.pyplot(fig)

# **Traffic Clustering using K-Means**
st.subheader("🚗 Traffic Clustering")
features = data[['Junction', 'Vehicles']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='Junction', y='Vehicles', hue='Cluster', data=data, palette='viridis', ax=ax)
ax.set_title('Traffic Clusters')
st.pyplot(fig)

# **Machine Learning Models for Prediction**
st.subheader("🧠 Traffic Prediction Using Machine Learning")

X = data[['Junction', 'Hour', 'Day', 'Month', 'Weekday']]
y = data['Vehicles']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Train KNN Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# **Train ANN Model (Optional)**
st.sidebar.subheader("🔬 Train ANN Model")
if st.sidebar.button("Train ANN Model"):
    ann_model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse')
    ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    ann_pred = ann_model.predict(X_test)
else:
    ann_pred = np.zeros_like(y_test)

# **Model Performance Comparison Table**
metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost", "ANN", "KNN"],
    "R2 Score": [
        r2_score(y_test, y_pred_lin),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
        r2_score(y_test, ann_pred),
        r2_score(y_test, y_pred_knn)
    ],
    "Mean Squared Error": [
        mean_squared_error(y_test, y_pred_lin),
        mean_squared_error(y_test, y_pred_rf),
        mean_squared_error(y_test, y_pred_xgb),
        mean_squared_error(y_test, ann_pred),
        mean_squared_error(y_test, y_pred_knn)
    ]
})

st.subheader("📊 Model Performance Comparison")
st.dataframe(metrics_df)

# **Dynamic Traffic Light Durations**
st.subheader("⏳ Dynamic Traffic Light Timing")

def calculate_light_durations(predicted_vehicles, total_predicted_traffic, min_green_time=20, max_green_time=120):
    green_time = max((predicted_vehicles / total_predicted_traffic) * max_green_time, min_green_time)
    yellow_time = 10  # fixed yellow light duration
    red_time = max_green_time - green_time - yellow_time
    red_time = max(red_time, 0)
    return green_time, red_time, yellow_time

total_predicted_traffic = sum(y_pred_rf)

high_traffic_junctions = data.groupby('Junction')['Vehicles'].mean().sort_values(ascending=False)
for junction in high_traffic_junctions.index[:5]:
    predicted_vehicles = data[data['Junction'] == junction]['Vehicles'].mean()
    green, red, yellow = calculate_light_durations(predicted_vehicles, total_predicted_traffic)
    st.write(f"🚦 Junction {junction} - Vehicles: {predicted_vehicles:.2f}")
    st.write(f"🟢 Green: {green:.2f} sec | 🔴 Red: {red:.2f} sec | 🟡 Yellow: {yellow} sec")

# **Conclusion & Insights**
st.subheader("📌 Key Insights & Recommendations")
st.markdown("- **Junction 1 has the highest predicted traffic**, followed by Junctions 3 and 2.")
st.markdown("- **Random Forest performed best** for traffic predictions.")
st.markdown("- **Dynamic traffic light durations** should be optimized to reduce congestion.")
st.markdown("- **Consider using ANN or XGBoost** for more accurate predictions.")

st.success("✅ Analysis Complete! Use the sidebar to explore more.")


# Fonctions pour l'analyse de la pollution de l'air
def pollution_analysis():
    st.header("🌍 Analyse de la Pollution de l'Air")

    # Chargement des données
    @st.cache_data
    def load_data():
        data = pd.read_csv('pollution_data_with_etat.csv', encoding='latin1', low_memory=False)
        data = data.replace(['NA', 'na', 'Na', 'nA'], np.nan)
        columns = ['so2', 'no2', 'pm2_5']
        data = data[columns]
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna()
        return data

    data = load_data()

    # Analyse des données
    st.subheader("📊 Analyse des Données")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre d'échantillons", f"{len(data):,}")
    with col2:
        st.metric("SO₂ moyen", f"{data['so2'].mean():.2f} µg/m³")
    with col3:
        st.metric("NO₂ moyen", f"{data['no2'].mean():.2f} µg/m³")

    # Visualisations
    st.write("### Visualisations")
    viz_type = st.selectbox(
        'Choisissez une visualisation:',
        ['Distribution des polluants', 'Matrice de corrélation']
    )

    if viz_type == 'Distribution des polluants':
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, col in enumerate(['so2', 'no2', 'pm2_5']):
            sns.histplot(data=data, x=col, ax=axes[idx], bins=30)
            axes[idx].set_title(f'Distribution de {col}')
        st.pyplot(fig)

    elif viz_type == 'Matrice de corrélation':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Entraînement des modèles
    st.subheader("🤖 Entraînement des Modèles")
    X = data.drop('pm2_5', axis=1)
    y = data['pm2_5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selected_models = st.multiselect(
        'Choisissez les modèles à entraîner:',
        ['Linear Regression', 'Random Forest', 'XGBoost', 'KNN', 'ANN'],
        default=['Linear Regression', 'Random Forest']
    )

    if st.button('Entraîner les modèles'):
        results = {}
        for model_name in selected_models:
            if model_name == 'Linear Regression':
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train_scaled, y_train)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(random_state=42)
                model.fit(X_train_scaled, y_train)
            elif model_name == 'KNN':
                model = KNeighborsRegressor()
                model.fit(X_train_scaled, y_train)
            elif model_name == 'ANN':
                model = Sequential([
                    Input(shape=(X_train_scaled.shape[1],)),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(1, activation='linear')
                ])
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model.predict(X_test_scaled).flatten()
            else:
                continue

            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

        st.subheader("📊 Résultats des Modèles")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

def show_predictions(data):
    """Page de prédictions"""
    st.header('🎯 Prédictions')
    
    try:
        # Charger le modèle et le scaler
        model = joblib.load('best_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Interface de saisie
        st.write("### Entrez les valeurs des polluants")
        col1, col2 = st.columns(2)
        with col1:
            so2_value = st.number_input('SO₂ (µg/m³)', 
                                       min_value=0.0,
                                       max_value=data['so2'].max(),
                                       value=data['so2'].mean())
        with col2:
            no2_value = st.number_input('NO₂ (µg/m³)',
                                       min_value=0.0,
                                       max_value=data['no2'].max(),
                                       value=data['no2'].mean())

        if st.button('Prédire PM2.5'):
            # Préparer les données pour la prédiction
            input_data = pd.DataFrame([[so2_value, no2_value]], columns=['so2', 'no2'])
            input_scaled = scaler.transform(input_data)
            
            # Faire la prédiction
            if hasattr(model, 'predict'):
                prediction = model.predict(input_scaled)[0]
            else:  # Pour le modèle ANN
                prediction = model.predict(input_scaled).flatten()[0]
            
            # Afficher les résultats
            st.success(f"Prédiction de PM2.5: {prediction:.2f} µg/m³")
            
            # Ajouter un contexte à la prédiction
            avg_pm25 = data['pm2_5'].mean()
            if prediction > avg_pm25:
                st.warning(f"Cette valeur est supérieure à la moyenne ({avg_pm25:.2f} µg/m³)")
            else:
                st.info(f"Cette valeur est inférieure à la moyenne ({avg_pm25:.2f} µg/m³)")
            
            # Afficher un graphique de contexte
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x='pm2_5', bins=30, ax=ax)
            plt.axvline(prediction, color='red', linestyle='--', label='Prédiction')
            plt.axvline(avg_pm25, color='green', linestyle='--', label='Moyenne')
            plt.legend()
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error("⚠️ Aucun modèle n'a été trouvé. Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement des Modèles'.")

# Sélection de l'analyse à afficher
if analysis_type == "🚦 Analyse du Trafic":
    traffic_analysis()
elif analysis_type == "🌍 Analyse de la Pollution de l'Air":
    pollution_analysis()
elif analysis_type == "🎯 Prédictions":
    # Chargement des données
    @st.cache_data
    def load_data():
        data = pd.read_csv('pollution_data_with_etat.csv', encoding='latin1', low_memory=False)
        data = data.replace(['NA', 'na', 'Na', 'nA'], np.nan)
        columns = ['so2', 'no2', 'pm2_5']
        data = data[columns]
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna()
        return data

    data = load_data()
    show_predictions(data)
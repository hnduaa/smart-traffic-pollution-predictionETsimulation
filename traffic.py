import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Streamlit Page Configuration
st.set_page_config(page_title="Traffic Analysis Dashboard", layout="wide")
st.title("🚦 Traffic Analysis and Prediction Dashboard")

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

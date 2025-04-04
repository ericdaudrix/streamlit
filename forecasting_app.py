#===================================================#
# Eric Daudrix - Lycée Monnerville Cahors - CMQE IF #
#===================================================#

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skforecast.recursive import ForecasterRecursive
import plotly.graph_objects as go

# Streamlit App Configuration
st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("📈 Forecasting Application")
st.sidebar.header("⚙️ Configuration du modèle prédictif")

# Sidebar inputs for parameters
data_freq = st.sidebar.number_input("⏱️ Fréquence des données (en secondes)", min_value=1, value=60, step=1)
steps = st.sidebar.number_input("📦 Taille des données de test", min_value=1, value=120, step=1)
lags = st.sidebar.number_input("🧠 Nombre de lags", min_value=1, value=120, step=1)
pred_steps = st.sidebar.number_input("🔮 Pas de prédiction", min_value=1, value=120, step=1)

# File upload
uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, sep=';')
        if data.shape[1] < 2:
            st.error("❌ Le fichier CSV doit contenir au moins deux colonnes (date, valeur).")
            st.stop()
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")
        st.stop()

    # Choix des colonnes
    columns = data.columns.tolist()
    date_col = st.sidebar.selectbox("🗓️ Colonne de date", options=columns, index=0)
    target_col = st.sidebar.selectbox("🎯 Colonne cible", options=columns, index=1)

    # Préparation des données
    data.rename(columns={date_col: 'date', target_col: 'y'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.asfreq(f'{data_freq}s')

    with st.expander("🔍 Aperçu des données brutes"):
        st.write(data.head())

    with st.expander("📊 Statistiques de la série"):
        st.write(data.describe())

    # Interpolation des valeurs manquantes
    data = data.interpolate()

    # Split train/test
    train = data[:-steps]
    test = data[-steps:]

    if st.button("🚀 Entraîner le modèle"):
        # Modélisation
        forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=lags)
        forecaster.fit(y=train['y'])

        st.success("✅ Modèle entraîné avec succès !")

        # Prédictions
        predictions = forecaster.predict(steps=pred_steps)

        # Affichage des courbes
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['y'], mode='lines', name='Train', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test.index, y=test['y'], mode='lines', name='Test', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions.values, mode='lines', name='Prédictions', line=dict(color='red')))
        fig.update_layout(title="Prédictions vs Réalité", xaxis_title="Date", yaxis_title="Valeur")
        st.plotly_chart(fig, use_container_width=True)

        # Évaluation
        y_true = test['y'][:pred_steps]
        y_pred = predictions[:len(y_true)]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        st.subheader("📐 Évaluation du modèle")
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")

        # Téléchargement CSV des prédictions
        result_df = pd.DataFrame({'date': predictions.index, 'prediction': predictions.values})
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les prédictions", data=csv, file_name='predictions.csv', mime='text/csv')

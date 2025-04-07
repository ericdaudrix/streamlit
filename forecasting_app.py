#===================================================#
# Eric Daudrix - Lycée Monnerville Cahors - CMQE IF #
#===================================================#

import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skforecast.recursive import ForecasterRecursive
import plotly.graph_objects as go
from datetime import datetime

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
        # Détection du séparateur
        sample = uploaded_file.read(1024).decode('utf-8')
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, sep=dialect.delimiter)

        # Vérification que le parsing a réussi
        if data.shape[1] < 2:
            st.warning(f"⚠️ Séparateur détecté : '{dialect.delimiter}' — tentative de relecture avec ','")
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, sep=',')

        if data.shape[1] < 2:
            st.error("❌ Le fichier CSV doit contenir au moins deux colonnes (date, valeur).")
            st.stop()

    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")
        st.stop()

    # Nom du fichier et horodatage
    #filename = uploaded_file.name.replace(".csv", "")
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Utilisation automatique des deux premières colonnes
    date_col = data.columns[0]
    target_col = data.columns[1]
    data.rename(columns={date_col: 'date', target_col: 'y'}, inplace=True)

    # Préparation des données
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.asfreq(f'{data_freq}s')
    data = data.interpolate()

    # Aperçu
    with st.expander("🔍 Aperçu des données brutes"):
        st.write(data.head())

    with st.expander("📊 Statistiques de la série"):
        st.write(data.describe())

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

        # Affichage graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['y'], mode='lines', name='Train', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test.index, y=test['y'], mode='lines', name='Test', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions.values, mode='lines', name='Prédictions', line=dict(color='red')))
        fig.update_layout(
            title=f"📊 Prédictions vs Réalité — {filename}",
            xaxis_title="Date",
            yaxis_title="Valeur"
        )
        st.plotly_chart(fig, use_container_width=True)

        

        # Téléchargement des prédictions
        result_df = pd.DataFrame({'date': predictions.index, 'prediction': predictions.values})
        csv_out = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les prédictions",
            data=csv_out,
            file_name=f'predictions_{filename}_{timestamp}.csv',
            mime='text/csv'
        )

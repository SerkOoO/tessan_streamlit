import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import json
import io
from snowflake.snowpark import Session
import librosa.display
import os
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION ET CONNEXION SNOWFLAKE
# ==========================================
st.set_page_config(page_title="Tessan - IA Respiratoire", page_icon="🩺", layout="wide")

load_dotenv()

@st.cache_resource
def init_snowpark():
    # /!\ TES IDENTIFIANTS /!\
    connection_parameters = {
        "account": os.getenv("ACCOUNT"),
        "user": os.getenv("USER"),
        "password": os.getenv("PASSWORD"),
        "passcode": "574179", # important à changer
        "role": "M2_BIGDATA_EQUIPE_1_ROLE",
        "warehouse": "HACKATHON_WH",
        "database": "M2_BIGDATA_EQUIPE_1_DB",
        "schema": "AUDIO_DATA"
    }
    return Session.builder.configs(connection_parameters).create()

# Initialisation sécurisée de la session
session = None
try:
    session = init_snowpark()
    st.sidebar.success("✅ Connecté à Snowflake")
except Exception as e:
    st.sidebar.error(f"❌ Erreur Snowflake : {e}")  # <-- Ceci va afficher le vrai problème !

# --- CAPTEURS DE LA CABINE ---
st.sidebar.header("🎛️ Capteurs de la cabine (Direct)")
spo2 = st.sidebar.slider("Taux d'oxygène (SpO2 %)", min_value=85, max_value=100, value=98, step=1)
temperature = st.sidebar.slider("Température (°C)", min_value=36.0, max_value=41.0, value=37.2, step=0.1)
frequence_cardiaque = st.sidebar.slider("Rythme cardiaque (BPM)", min_value=50, max_value=150, value=75, step=1)

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_classes=5):
        super(CNNBiLSTMAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.lstm_input_size = 128 * 16
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes))
    
    def forward(self, x):
        x = self.cnn(x)
        batch, channels, freq, temps = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, temps, channels * freq)
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)

@st.cache_resource
def load_model():
    model = CNNBiLSTMAttention(n_classes=5)
    model.load_state_dict(torch.load('meilleur_modele_bilstm.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
CLASSES = ['Asthme', 'BPCO', 'Bronchite', 'Pneumonie', 'Sain']

# ==========================================
# 3. PRÉTRAITEMENT
# ==========================================
def creer_filtres_mel(sr, n_fft, n_mels, n_freqs):
    f_min, f_max = 0.0, sr / 2.0
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points / (sr / n_fft)).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)
    filters = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m-1], bin_points[m], bin_points[m+1]
        if f_center > f_left:
            filters[m-1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / (f_center - f_left)
        if f_right > f_center:
            filters[m-1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / (f_right - f_center)
        filters[m-1, f_center] = 1.0
    return filters

def preparer_audio_exact(fichier_audio):
    audio_bytes = fichier_audio.read()
    sr_orig, y = wav.read(io.BytesIO(audio_bytes))
    
    if y.ndim == 2: y = y[:, 0]
    y = y.astype(np.float32)

    TARGET_SR = 22050
    if sr_orig != TARGET_SR:
        nb_samples_cible = int(len(y) * TARGET_SR / sr_orig)
        y = signal.resample(y, nb_samples_cible)

    nyquist = TARGET_SR / 2
    b, a = signal.butter(4, [250/nyquist, 2000/nyquist], btype='band')
    y = signal.filtfilt(b, a, y).astype(np.float32)

    TARGET_LENGTH = TARGET_SR * 5
    if np.max(np.abs(y)) > 0: y = y / np.max(np.abs(y))
    if len(y) > TARGET_LENGTH: y = y[:TARGET_LENGTH]
    elif len(y) < TARGET_LENGTH: y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')

    _, _, Sxx = signal.spectrogram(y, fs=TARGET_SR, nperseg=2048, noverlap=2048 - 512, window='hann')
    mel_filters = creer_filtres_mel(TARGET_SR, 2048, 128, Sxx.shape[0])
    mel_db = 10 * np.log10(np.maximum(np.dot(mel_filters, Sxx), 1e-10))

    # Z-Score global ajusté pour la démo
    mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-10)
    return mel_norm, mel_db

# ==========================================
# 4. INTERFACE PRODUIT
# ==========================================

st.title("🩺 Cabine Connectée Tessan - Diagnostic IA Multimodal") 

choix_entree = st.radio("Acquisition du signal :", ["📂 Uploader une auscultation (.wav)", "🎤 Stéthoscope Connecté (Temps réel)"], horizontal=True)

audio_data = None
if choix_entree == "📂 Uploader une auscultation (.wav)":
    audio_data = st.file_uploader("📥 Charger un fichier", type=["wav"])
else:
    st.info("Placez le stéthoscope sur le patient (ou le micro contre votre gorge) et respirez doucement.")
    audio_data = st.audio_input("Enregistrer l'auscultation")

if audio_data is not None:
    mel_norm, mel_db = preparer_audio_exact(audio_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Signature Acoustique")
        st.audio(audio_data, format='audio/wav')
        fig, ax = plt.subplots(figsize=(10, 5))
        librosa.display.specshow(mel_db, sr=22050, x_axis='time', y_axis='mel', ax=ax, cmap='inferno')
        st.pyplot(fig)

    with col2:
        st.subheader("🧠 Pré-diagnostic IA")
        tensor_input = torch.tensor(mel_norm[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        
        with st.spinner('Analyse des biomarqueurs en cours...'):
            with torch.no_grad():
                outputs = model(tensor_input)
                probs = torch.softmax(outputs, dim=1)[0].numpy() * 100
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.barh(CLASSES, probs, color=['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd'])
            ax2.set_xlabel('Probabilité (%)')
            st.pyplot(fig2)
            
            idx = np.argmax(probs)
            confiance = probs[idx]
            
    st.divider()
    st.markdown("### 📋 Synthèse Clinique & Sévérité Multimodale")
    
    severite = "LÉGER"
    couleur = "green"
    
    if CLASSES[idx] in ['Pneumonie', 'BPCO', 'Asthme']:
        if spo2 < 92 or temperature > 38.5:
            severite = "SÉVÈRE (Alerte Rouge)"
            couleur = "red"
        elif spo2 < 95 or temperature > 37.8:
            severite = "MODÉRÉ (Alerte Orange)"
            couleur = "orange"
            
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Diagnostic principal ciblé", value=f"{CLASSES[idx]}", delta=f"Confiance IA : {confiance:.1f}%")
        st.markdown(f"**Niveau de Sévérité estimé :** :{couleur}[{severite}]")
        
        if st.button("💾 Synchroniser avec le dossier Patient (Snowflake)"):
            if session is not None:  # <--- VÉRIFICATION DE SÉCURITÉ ICI
                try:
                    p_json = json.dumps({CLASSES[i]: float(probs[i]) for i in range(5)})
                    session.sql(f"""
                        INSERT INTO PREDICTIONS (PHARMACIE_ID, CLASSE_PREDITE, PROBABILITES, CONFIANCE)
                        SELECT 'CABINE_TEST', '{CLASSES[idx]}', PARSE_JSON('{p_json}'), {float(confiance)}
                    """).collect()
                    st.toast('✅ Données enregistrées dans Snowflake avec succès !')
                except Exception as e:
                    st.error(f"Erreur Snowflake : {e}")
            else:
                st.error("⚠️ Sauvegarde impossible : non connecté à Snowflake.")

    with col_b:
        if severite == "SÉVÈRE (Alerte Rouge)":
            st.error("🚨 DÉTRESSE RESPIRATOIRE SUSPECTÉE.\n- Hypoxie ou fièvre détectée par la cabine.\n- Redirection immédiate vers les urgences (15).")
        elif severite == "MODÉRÉ (Alerte Orange)":
            st.warning("⚠️ ANOMALIE CLINIQUE CONFIRMÉE.\n- Téléconsultation prioritaire avec un médecin généraliste requise.\n- Prescription probable.")
        else:
            if CLASSES[idx] == 'Sain':
                st.success("✅ PARAMÈTRES NORMAUX.\n- Auscultation claire, constantes stables.\n- Fin de consultation classique.")
            else:
                st.info("ℹ️ PATHOLOGIE DÉTECTÉE MAIS CONSTANTES STABLES.\n- Pas de détresse immédiate (SpO2 normale).\n- Recommandation : suivi médical régulier.")

    # ==========================================
    # --- DASHBOARD LONGITUDINAL RÉEL ---
    # ==========================================
    with st.expander("📈 Voir l'historique pulmonaire du patient (Extraction Snowflake en temps réel)"):
        if session is not None:
            try:
                # On requête Snowflake pour extraire TOUTES les probabilités du JSON
                query = """
                    SELECT 
                        TIMESTAMP as "Date", 
                        PROBABILITES:Asthme::FLOAT as "Asthme",
                        PROBABILITES:BPCO::FLOAT as "BPCO",
                        PROBABILITES:Bronchite::FLOAT as "Bronchite",
                        PROBABILITES:Pneumonie::FLOAT as "Pneumonie",
                        PROBABILITES:Sain::FLOAT as "Sain"
                    FROM PREDICTIONS 
                    ORDER BY TIMESTAMP ASC
                """
                df_history = session.sql(query).to_pandas()
                
                if not df_history.empty:
                    # Convertir la colonne en format Date pour Streamlit
                    df_history['Date'] = pd.to_datetime(df_history['Date'])
                    
                    # Définir la Date comme index pour que Streamlit trace les 5 courbes correctement
                    df_history = df_history.set_index('Date')
                    
                    st.line_chart(df_history)
                    st.caption("Cette vue permet au médecin de suivre l'évolution de toutes les pathologies simultanément.")
                else:
                    st.info("Aucun historique patient disponible dans la base de données. Sauvegardez une prédiction pour commencer le suivi.")
            except Exception as e:
                st.error(f"Erreur lors de la récupération de l'historique : {e}")
        else:
            st.info("🖥️ Mode Local Actif : Historique désactivé. Connectez-vous à Snowflake pour voir le suivi.")
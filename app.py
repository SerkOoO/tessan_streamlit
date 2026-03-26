import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import json
import io
from snowflake.snowpark import Session
import librosa.display
import speech_recognition as sr
from gtts import gTTS
import base64
import os
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION ET CONNEXION SNOWFLAKE
# ==========================================
st.set_page_config(page_title="Tessan - IA Respiratoire", page_icon="🩺", layout="wide")

load_dotenv()

@st.cache_resource
def init_snowpark():
    # /!\ IDENTIFIANTS /!\
    connection_parameters = {
        "account": os.getenv("ACCOUNT"),
        "user": os.getenv("USER"),
        "password": os.getenv("PASSWORD"),
        "passcode": os.getenv("PASSCODE"),
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
    st.sidebar.error(f"❌ Erreur Snowflake : {e}") 

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

    # Filtrage passe-bande standard (100 - 2000 Hz)
    nyquist = TARGET_SR / 2
    b, a = signal.butter(4, [100/nyquist, 2000/nyquist], btype='band')
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



def generate_gradcam(model, tensor_input, target_class_idx=None):
    model.eval()
    activations = []

    # On met un crochet pour capturer l'image directement à la sortie du CNN (avant le LSTM)
    def forward_hook(module, input, output):
        activations.append(output)

    # On s'accroche à la dernière couche d'activation ReLU du CNN (index 10)
    handle = model.cnn[10].register_forward_hook(forward_hook)

    # On fait passer le son (juste en marche avant, pas de backward compliqué)
    with torch.no_grad():
        _ = model(tensor_input)

    handle.remove()

    # On récupère l'image vue par le CNN
    acts = activations[0].squeeze(0)  # Shape: [128 filtres, H, W]

    # On fait la moyenne de tous les filtres pour voir les zones qui "réagissent" le plus au son
    heatmap = torch.mean(acts, dim=0)

    # Normalisation propre entre 0 et 1 pour forcer l'apparition du Rouge/Jaune
    heatmap = F.relu(heatmap)
    heatmap_min, heatmap_max = torch.min(heatmap), torch.max(heatmap)
    if heatmap_max - heatmap_min > 1e-8:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = torch.zeros_like(heatmap)

    # On étire la carte pour qu'elle fasse la taille de ton spectrogramme (128, 212)
    original_shape = (tensor_input.shape[2], tensor_input.shape[3])
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap, size=original_shape, mode='bilinear', align_corners=False)

    return heatmap_resized[0, 0].cpu().numpy()

def faire_parler_ia(texte):
    # Génère l'audio en français
    tts = gTTS(text=texte, lang='fr')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    # Convertit en base64 pour forcer la lecture automatique en HTML
    b64 = base64.b64encode(fp.read()).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# ==========================================
# 4. INTERFACE PRODUIT
# ==========================================

st.title("🩺 Cabine Connectée Tessan - Diagnostic IA Multimodal") 

# --- GESTIONNAIRE DE SCÉNARIO ---
if 'etape_consultation' not in st.session_state:
    st.session_state.etape_consultation = 0
if 'fumeur_text' not in st.session_state:
    st.session_state.fumeur_text = ""
if 'symptomes_text' not in st.session_state:
    st.session_state.symptomes_text = ""
if 'audio_ia_a_jouer' not in st.session_state:
    st.session_state.audio_ia_a_jouer = None

# Fonction pour préparer l'audio de l'IA (Text-To-Speech)
def preparer_audio_ia(texte):
    tts = gTTS(text=texte, lang='fr')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    b64 = base64.b64encode(fp.read()).decode()
    return f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """

# Lecture automatique du son généré à l'étape précédente
if st.session_state.audio_ia_a_jouer:
    st.markdown(st.session_state.audio_ia_a_jouer, unsafe_allow_html=True)
    st.session_state.audio_ia_a_jouer = None

# ==============================================================
# LE BOUTON DE DÉMARRAGE
# ==============================================================
if st.session_state.etape_consultation == 0:
    st.markdown("### 🤖 Étape 1 : Assistant Médical Vocal")
    st.info("Bienvenue dans la cabine Tessan. Cliquez ci-dessous pour lancer la consultation.")
    if st.button("▶️ Démarrer la consultation"):
        st.session_state.audio_ia_a_jouer = preparer_audio_ia("Bonjour. Je suis l'assistant médical de la cabine. Pour commencer, êtes-vous fumeur ?")
        st.session_state.etape_consultation = 1
        st.rerun()

# ==============================================================
# QUESTION 1 (FUMEUR)
# ==============================================================
elif st.session_state.etape_consultation == 1:
    st.markdown("### 🤖 Étape 1 : Assistant Médical Vocal")
    st.warning("🗣️ L'IA : 'Bonjour. Je suis l'assistant médical de la cabine. Pour commencer, êtes-vous fumeur ?'")
    
    audio_fumeur = st.audio_input("🎤 Répondez à l'IA (ex: 'Non, je ne fume pas' ou 'Oui, un paquet par jour')")
    
    if audio_fumeur is not None:
        with st.spinner("L'IA transcrit votre voix..."):
            r = sr.Recognizer()
            with sr.AudioFile(audio_fumeur) as source:
                audio_data_sr = r.record(source)
            try:
                texte_transcrit = r.recognize_google(audio_data_sr, language="fr-FR")
                st.session_state.fumeur_text = texte_transcrit
                
                # On prépare la question 2
                st.session_state.audio_ia_a_jouer = preparer_audio_ia("D'accord, c'est noté. Pouvez-vous maintenant me décrire vos symptômes ?")
                st.session_state.etape_consultation = 2
                st.rerun()
            except sr.UnknownValueError:
                st.error("L'IA n'a pas bien compris. Pouvez-vous répéter ?")
            except sr.RequestError:
                st.error("Erreur de connexion au service vocal de Google.")

# ==============================================================
# QUESTION 2 (SYMPTÔMES)
# ==============================================================
elif st.session_state.etape_consultation == 2:
    st.markdown("### 🤖 Étape 1 : Assistant Médical Vocal")
    st.success(f"🚭 **Tabagisme :** '{st.session_state.fumeur_text}'")
    st.warning("🗣️ L'IA : 'D'accord, c'est noté. Pouvez-vous maintenant me décrire vos symptômes ?'")
    
    audio_symptomes = st.audio_input("🎤 Décrivez vos symptômes (ex: 'Je tousse et j'ai de la fièvre')")
    
    if audio_symptomes is not None:
        with st.spinner("L'IA transcrit votre voix..."):
            r = sr.Recognizer()
            with sr.AudioFile(audio_symptomes) as source:
                audio_data_sr = r.record(source)
            try:
                texte_transcrit = r.recognize_google(audio_data_sr, language="fr-FR")
                st.session_state.symptomes_text = texte_transcrit
                
                # On prépare le passage à l'auscultation
                reponse_ia = "Merci pour ces informations. Veuillez maintenant placer le stéthoscope sur votre poitrine, et respirez profondément par la bouche pendant 5 secondes."
                st.session_state.audio_ia_a_jouer = preparer_audio_ia(reponse_ia)
                st.session_state.etape_consultation = 3
                st.rerun()
            except sr.UnknownValueError:
                st.error("L'IA n'a pas bien compris. Pouvez-vous répéter ?")
            except sr.RequestError:
                st.error("Erreur de connexion au service vocal.")

# ==============================================================
# L'AUSCULTATION
# ==============================================================
elif st.session_state.etape_consultation >= 3:
    col_recap1, col_recap2 = st.columns(2)
    with col_recap1:
        st.success(f"🚭 **Tabagisme :** '{st.session_state.fumeur_text}'")
    with col_recap2:
        st.success(f"🤒 **Symptômes :** '{st.session_state.symptomes_text}'")
        
    st.warning("🗣️ L'IA : 'Veuillez maintenant placer le stéthoscope sur votre poitrine...'")
    
    st.divider()
    st.markdown("### 🩺 Étape 2 : Auscultation Connectée")
    
    choix_entree = st.radio("Acquisition du signal :", ["📂 Uploader une auscultation (.wav)", "🎤 Stéthoscope Connecté (Temps réel)"], horizontal=True)

    audio_data = None
    if choix_entree == "📂 Uploader une auscultation (.wav)":
        audio_data = st.file_uploader("📥 Charger un fichier", type=["wav"])
    else:
        st.info("Le stéthoscope est activé. Enregistrez la respiration.")
        audio_data = st.audio_input("Enregistrer l'auscultation")

    # --- LE MOTEUR IA (CNN-BiLSTM + Grad-CAM + Snowflake) ---
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
        
        # -----------------------------------------------------
        # BLOC GRAD-CAM (EXPLICABILITÉ)
        # -----------------------------------------------------
        st.divider()
        with st.expander("🔍 Explicabilité de l'IA (Grad-CAM) - Pourquoi ce diagnostic ?", expanded=True):
            st.markdown(f"**Analyse des biomarqueurs spatiaux pour la classe : `{CLASSES[idx]}`**")
            st.caption("Cette carte de chaleur met en évidence les fréquences et les instants précis (en rouge/jaune) qui ont convaincu le réseau de neurones de son diagnostic.")
            
            cam_heatmap = generate_gradcam(model, tensor_input, idx)
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            img_bg = ax3.imshow(mel_db, aspect='auto', origin='lower', cmap='gray')
            cam_heatmap_masked = np.ma.masked_where(cam_heatmap < 0.3, cam_heatmap)
            img_heatmap = ax3.imshow(cam_heatmap_masked, aspect='auto', origin='lower', cmap='jet', alpha=0.6)
            ax3.set_xlabel('Temps')
            ax3.set_ylabel('Fréquence (Mel)')
            st.pyplot(fig3)

            # --- GÉNÉRATION DU RAPPORT CORTEX MULTIMODAL ---
            max_y, max_x = np.unravel_index(np.argmax(cam_heatmap), cam_heatmap.shape)
            temps_sec = (max_x / cam_heatmap.shape[1]) * 5.0
            
            if max_y < 42:
                bande_freq = "basses fréquences"
            elif max_y < 85:
                bande_freq = "fréquences moyennes"
            else:
                bande_freq = "hautes fréquences"

            if session is not None:
                with st.spinner("Génération du rapport médical par LLM (Snowflake Cortex)..."):
                    try:
                        # PROMPT COMPLET
                        prompt = f"Agis comme un pneumologue. Le patient a déclaré à l'oral être fumeur ('{st.session_state.fumeur_text}') et avoir ces symptômes ('{st.session_state.symptomes_text}'). L'IA stéthoscopique a détecté une probabilité de '{CLASSES[idx]}' (anomalie trouvée à {temps_sec:.1f}s dans les {bande_freq}). Rédige un compte-rendu clinique de 3 phrases maximum en fusionnant ses antécédents, ses symptômes vocaux et l'analyse audio."
                        prompt_sql = prompt.replace("'", "''") 
                        query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_sql}')"
                        
                        rapport_genere = session.sql(query).collect()[0][0]
                        st.success("🤖 **Rapport IA (Snowflake Cortex) :**")
                        st.write(rapport_genere)
                    except Exception as e:
                        st.warning(f"L'analyse mathématique pointe une anomalie à {temps_sec:.1f}s ({bande_freq}).")

        # -----------------------------------------------------
        # SYNTHÈSE CLINIQUE ET SNOWFLAKE DB
        # -----------------------------------------------------
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
                if session is not None: 
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
                st.error("🚨 DÉTRESSE RESPIRATOIRE SUSPECTÉE.\n- Redirection immédiate vers les urgences (15).")
            elif severite == "MODÉRÉ (Alerte Orange)":
                st.warning("⚠️ ANOMALIE CLINIQUE CONFIRMÉE.\n- Téléconsultation prioritaire requise.")
            else:
                st.success("✅ PARAMÈTRES NORMAUX ou CONSTANTES STABLES.")
        
        # --- DASHBOARD ---
        with st.expander("📈 Voir l'historique pulmonaire du patient (Extraction Snowflake en temps réel)"):
            if session is not None:
                try:
                    query = """
                        SELECT TIMESTAMP as "Date", PROBABILITES:Asthme::FLOAT as "Asthme", PROBABILITES:BPCO::FLOAT as "BPCO", PROBABILITES:Bronchite::FLOAT as "Bronchite", PROBABILITES:Pneumonie::FLOAT as "Pneumonie", PROBABILITES:Sain::FLOAT as "Sain"
                        FROM PREDICTIONS ORDER BY TIMESTAMP ASC
                    """
                    df_history = session.sql(query).to_pandas()
                    if not df_history.empty:
                        df_history['Date'] = pd.to_datetime(df_history['Date'])
                        df_history = df_history.set_index('Date')
                        st.line_chart(df_history)
                except Exception as e:
                    pass

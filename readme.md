# 🩺 Tessan x Snowflake - Détection IA de Maladies Respiratoires

Ce projet a été réalisé dans le cadre du **Hackathon Snowflake x Tessan**. Il s'agit d'une application de télémédecine intégrant un modèle d'Intelligence Artificielle capable de détecter des maladies respiratoires (Asthme, BPCO, Bronchite, Pneumonie) à partir de sons d'auscultation pulmonaire.

L'interface simule l'environnement d'une cabine médicale connectée Tessan, combinant l'analyse audio en temps réel avec un croisement multimodal des constantes vitales (SpO2, température) pour fournir un diagnostic et un score de sévérité au médecin.

---

## 🚀 Prérequis

Avant de commencer, assurez-vous d'avoir installé sur votre machine :
* **Python 3.11** (recommandé pour la compatibilité avec Snowflake Snowpark et PyTorch).
* Un compte **Snowflake** fonctionnel avec les droits nécessaires pour lire/écrire dans une base de données.

---

## 🛠️ Installation du projet en local

Pour isoler les dépendances du projet et éviter les conflits avec votre système, nous allons créer un environnement virtuel Python (`.venv`).

### 1. Cloner le dépôt
Ouvrez votre terminal et clonez le projet (ou téléchargez les fichiers) :
```bash
git clone https://github.com/SerkOoO/tessan_streamlit.git
cd <NOM_DU_DOSSIER> 
```

### 2. Créer l'environnement virtuel (.venv)
```bash
python -m venv .venv
```
### 3. Activer l'environnement virtuel

#### Windows
```bash
.venv\Scripts\activate
```
#### macOS et Linux
```bash
source .venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```
ou

```bash
pip install streamlit torch torchvision torchaudio librosa snowflake-snowpark-python scipy numpy matplotlib pandas scikit-learn
```

### 5. Lancement de l'application streamlit
```bash
streamlit run app.py
```
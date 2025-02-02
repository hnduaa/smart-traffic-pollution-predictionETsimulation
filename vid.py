import streamlit as st

# Titre de l'application
st.title("📽️ Lecteur de vidéo local avec Streamlit")

# Instructions pour l'utilisateur
st.write("Téléchargez un fichier vidéo (WebM, MP4, AVI, MOV, MKV) pour l'afficher dans le lecteur.")

# Télécharger un fichier vidéo
uploaded_file = st.file_uploader(
    "Choisissez un fichier vidéo", 
    type=["webm", "mp4", "avi", "mov", "mkv"]  # Formats supportés
)

# Vérifier si un fichier a été téléchargé
if uploaded_file is not None:
    # Lire le fichier vidéo
    video_bytes = uploaded_file.read()
    
    # Afficher la vidéo
    st.video(video_bytes)
    
    # Afficher un message de succès
    st.success("Vidéo affichée avec succès !")
else:
    # Afficher un message si aucun fichier n'a été téléchargé
    st.warning("Veuillez télécharger un fichier vidéo.")
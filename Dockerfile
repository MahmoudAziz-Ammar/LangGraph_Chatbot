FROM python:3.12-slim

# Dossier de travail
WORKDIR /app

# Copie les dépendances
COPY requirements.txt .

# Installation
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le projet
COPY . .

# Crée les dossiers nécessaires
RUN mkdir -p data/docs vectorstore

# Port Streamlit
EXPOSE 8501

# Lancement
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
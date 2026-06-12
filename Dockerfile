# Usamos una versión oficial de Python ligera
FROM python:3.11-slim

# Regla estricta de Hugging Face: Crear un usuario que no sea administrador (seguridad)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Definimos nuestra carpeta de trabajo
WORKDIR /app

# Copiamos e instalamos los requerimientos primero
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de tu código
COPY --chown=user . .

# Hugging Face exige que las aplicaciones escuchen obligatoriamente en el puerto 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
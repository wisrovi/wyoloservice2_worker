
FROM wisrovi/agents:gpu-slim-yolo

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    net-tools \
    git \
    iputils-ping \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    make \
    cifs-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear directorios de montaje
RUN mkdir -p /config_versions \
    && mkdir -p /database \
    && mkdir -p /datasets

RUN chmod 755 /config_versions \
    && chmod 755 /database \
    && chmod 755 /datasets    


# Limpiar caché y archivos temporales
RUN rm -rf /tmp/*
RUN apt-get clean


# Configurar el entorno de trabajo
WORKDIR /requirements

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /lib/wyoloservice
COPY lib .
RUN pip install -e .

WORKDIR /usr/local/bin/
COPY docker/mount-cifs.sh .
RUN chmod +x /usr/local/bin/mount-cifs.sh

WORKDIR /app/
COPY config.yaml .


RUN echo "figlet wyoloservice" >> ~/.zshrc
RUN echo "figlet wyoloservice" >> ~/.bashrc
RUN echo "sh -c /usr/local/bin/mount-cifs.sh" >> ~/.zshrc
RUN echo "sh -c /usr/local/bin/mount-cifs.sh" >> ~/.bashrc
RUN echo "alias train_service='python'" >> ~/.zshrc
RUN echo "alias train_service='python'" >> ~/.bashrc
# Create the train_service script
RUN echo '#!/bin/sh\nexec python "$@"' > /usr/local/bin/train_service && chmod +x /usr/local/bin/train_service




EXPOSE 8000


# CMD ["sh", "-c", "/usr/local/bin/mount-cifs.sh && train_service worker.py"]
CMD ["sh", "-c", "/usr/local/bin/mount-cifs.sh && tail -f /dev/null"]
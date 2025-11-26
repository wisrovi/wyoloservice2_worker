
FROM wisrovi/agents:gpu-slim-yolo

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install net-tools
RUN apt-get -y install git
RUN apt install iputils-ping -y

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN apt-get update && apt-get install -y \
    make \
    cifs-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear directorios de montaje
RUN mkdir -p /config_versions \
    && mkdir -p /database \
    && mkdir -p /datasets

RUN chmod -R 777 /config_versions \
    && chmod -R 777 /database \
    && chmod -R 777 /datasets    


# Limpiar caché y archivos temporales
RUN rm -rf /var/cache/apk/*
RUN rm -rf /tmp/*
RUN apt-get clean

# Copiar el script para montar las unidades CIFS
WORKDIR /usr/local/bin/
COPY docker/mount-cifs.sh .
RUN chmod +x /usr/local/bin/mount-cifs.sh

# Configurar el entorno de trabajo
WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /lib/wyoloservice
COPY lib .
RUN pip install -e .


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
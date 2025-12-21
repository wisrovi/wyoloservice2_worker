
FROM wisrovi/wyoloservice_wyolo_base:v1.0.0


# Configurar el entorno de trabajo
WORKDIR /requirements

# COPY docker/requirements.txt .
# RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt


WORKDIR /usr/local/bin/
COPY docker/mount-cifs.sh .
RUN chmod +x /usr/local/bin/mount-cifs.sh


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
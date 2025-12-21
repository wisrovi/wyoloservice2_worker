
FROM wisrovi/wyoloservice_wyolo_base:v1.0.0

WORKDIR /lib/wyoloservice
COPY lib .
# RUN pip install -e .

WORKDIR /usr/local/bin/
COPY docker/mount-cifs.sh .
RUN chmod +x /usr/local/bin/mount-cifs.sh

WORKDIR /app/
COPY ./config/config.yaml .


RUN echo "figlet wyolo" >> ~/.zshrc
RUN echo "figlet wyolo" >> ~/.bashrc
RUN echo "sh -c /usr/local/bin/mount-cifs.sh" >> ~/.zshrc
RUN echo "sh -c /usr/local/bin/mount-cifs.sh" >> ~/.bashrc
RUN echo "alias train_service='python'" >> ~/.zshrc
RUN echo "alias train_service='python'" >> ~/.bashrc
# Create the train_service script
RUN echo '#!/bin/sh\nexec python "$@"' > /usr/local/bin/train_service && chmod +x /usr/local/bin/train_service






# CMD ["sh", "-c", "/usr/local/bin/mount-cifs.sh && train_service worker.py"]
CMD ["sh", "-c", "/usr/local/bin/mount-cifs.sh && tail -f /dev/null"]
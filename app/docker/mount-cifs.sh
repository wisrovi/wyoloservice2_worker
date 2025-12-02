#!/bin/bash

# Leer variables de entorno (con valores predeterminados)
CONTROL_HOST=${CONTROL_HOST:-localhost}
CIFS_USER=${CIFS_USER:-wisrovi}
CIFS_PASS=${CIFS_PASS:-wyoloservice}

# Imprimir variables de entorno para depuración
echo
echo "Credenciales samba"
echo "control_server_HOST: $CONTROL_HOST"
echo "USER: $CIFS_USER"
echo "PASS: $CIFS_PASS"


# Instalar paquetes necesarios si no están disponibles
if ! command -v mount.cifs &> /dev/null; then
    echo "Instalando cifs-utils..."
    apt-get update && apt-get install -y cifs-utils
fi

echo
echo "Montando /mnt/control_server ..."


# Crear directorio de montaje
mkdir -p /mnt/control_server/datasets
mkdir -p /mnt/control_server/config_versions
mkdir -p /mnt/control_server/api_database

chmod 777 -R /mnt/control_server
chmod 777 -R /mnt/control_server/datasets
chmod 777 -R /mnt/control_server/config_versions
chmod 777 -R /mnt/control_server/api_database

# Probar diferentes métodos de montaje que usa Nautilus
echo "Intentando montaje en control_server"

# Probar diferentes configuraciones de montaje
echo "Probando configuración 1: vers=3.0"
MOUNT_OPTS="username=$CIFS_USER,password=$CIFS_PASS,port=23449,file_mode=0777,dir_mode=0777,iocharset=utf8,uid=$(id -u),gid=$(id -g),vers=3.0,soft"

# Montar cada share con manejo de errores
echo "Montando datasets..."
mount -t cifs //$CONTROL_HOST/datasets /mnt/control_server/datasets -o $MOUNT_OPTS || echo "Error montando datasets"

echo "Montando config_versions..."
mount -t cifs //$CONTROL_HOST/config_versions /mnt/control_server/config_versions -o $MOUNT_OPTS || echo "Error montando config_versions"

echo "Montando api_database..."
mount -t cifs //$CONTROL_HOST/api_database /mnt/control_server/api_database -o $MOUNT_OPTS || echo "Error montando api_database"


echo
echo "Contenido de /mnt/control_server:"
ls -la /mnt/control_server
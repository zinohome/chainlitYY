#!/bin/bash
set -e
set -x
apt-get update && DEBIAN_FRONTEND=noninteractive && \
apt -y dist-upgrade && \
apt install -y --no-install-recommends build-essential libssl-dev libffi-dev python3-dev net-tools libsasl2-dev curl wget procps git libnss3-tools python3-pip && \
apt install -y software-properties-common  && add-apt-repository -y ppa:deadsnakes/ppa && apt install -y python3.10 && \
rm /usr/bin/python && ln -s /usr/bin/python3.10 /usr/bin/python && \
python -m pip install virtualenv && \
cd /opt && \
git clone https://github.com/zinohome/crtool.git && \
cd /opt/crtool && \
git pull && \
chmod 755 mkcert-v1.4.4-linux-amd64 && mv mkcert-v1.4.4-linux-amd64 mkcert && mv mkcert /usr/bin/ && \
mkcert -install && \
mkdir -p /opt/crtool/backend/log && \
mkdir -p /opt/crtool/cert && \
mkcert -cert-file /opt/crtool/cert/cert.pem -key-file /opt/crtool/cert/key.pem ibmtls.com crtool.ibmtls.com localhost 127.0.0.1 ::1 && \
cd /opt/crtool && \
virtualenv venv && \
. venv/bin/activate && \
pip install -r requirements.txt && \
cp /opt/crtool/docker/bd_build/wait-for /usr/bin/wait-for && chmod 755 /usr/bin/wait-for && \
ls -l /opt/crtool/docker/bd_build/ && \
cp /opt/crtool/docker/bd_build/50_start_h.sh /etc/my_init.d/50_start_crtool.sh &&
chmod 755 /etc/my_init.d/50_start_crtool.sh
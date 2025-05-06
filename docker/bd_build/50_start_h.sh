#!/bin/bash
cd /opt/chainlitYY && \
. venv/bin/activate && \
nohup /opt/chainlitYY/venv/bin/chainlit run app.py -h --host 0.0.0.0 --port 8000 >> /tmp/chainlitYY.log 2>&1 &
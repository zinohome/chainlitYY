#!/bin/bash
FIND_FILE="/opt/crtool/apprun/hypercorn.py"
FIND_STR="workers"
if [ `grep -c "$FIND_STR" $FIND_FILE` -ne '0' ];then
    echo "hypercorn config exist"
else
    cp /opt/crtool/apprun/hypercorn_default.py /opt/crtool/apprun/hypercorn.py
fi

cd /opt/crtool/backend && \
nohup /opt/crtool/venv/bin/hypercorn -c /opt/crtool/apprun/hypercorn.py main:app >> /tmp/crtool.log 2>&1 &
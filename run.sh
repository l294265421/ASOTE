#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=anaconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

${python} $@


end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}

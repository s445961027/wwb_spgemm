# !/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <executable> <#nodes> <#processes> <#threads per process>" >&2
    exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-fall/course/hpc/projects/2023-xue/mat_data
RESPATH=/home/2023-fall/course/hpc/projects/2023-xue/gemm_res

EXECUTABLE=$1
REP=64

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

for FILE in ${FILELIST}; do
    FILEPATH=${DATAPATH}/${FILE}
    RES=${RESPATH}/${FILE}
    if test -f ${FILEPATH}; then
        srun -N $2 --exclusive -n $3 -c $4 ${EXECUTABLE} ${REP} ${FILEPATH} ${RES}
    fi
done

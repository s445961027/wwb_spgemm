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

srun -N $2 --exclusive -n $3 -c $4 ${EXECUTABLE} ${REP} ${DATAPATH}/sd2010.csr ${RESPATH}/sd2010.csr

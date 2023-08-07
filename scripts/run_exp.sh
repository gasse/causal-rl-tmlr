#!/bin/bash

usage()
{
  echo "Usage: $0 -e|--experiment EXPERIMENT -x|--expert EXPERT -m|--method METHOD -s|--seed SEED -g|--gpu GPU"
  exit 1
}

unset EXPERIMENT
unset EXPERT
unset SEED
unset METHOD
unset GPU

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--experiment)
      EXPERIMENT="$2"
      shift # past argument
      shift # past value
      ;;
    -x|--expert)
      EXPERT="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--method)
      METHOD="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--seed)
      SEED="$2"
      shift # past argument
      shift # past value
      ;;
    -g|--gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      usage
      ;;
  esac
done

if [ -z "${EXPERIMENT}" ] ; then
    usage
fi

if [ -z "${EXPERT}" ] ; then
    usage
fi

if [ -z "${METHOD}" ] ; then
    usage
fi

if [ -z "${SEED}" ] ; then
    usage
fi

if [ -z "${GPU}" ] ; then
    usage
fi

NINTS_SEQ=$(python - <<-EOF
import json
print(*json.load(open('experiments/${EXPERIMENT}/hyperparams.json', 'r'))['nsamples_int'])
EOF
)

python scripts/01_generate_test_data.py --experiment $EXPERIMENT --seed $SEED
python scripts/02_generate_data_obs.py --experiment $EXPERIMENT --expert $EXPERT --seed $SEED
for NINTS in ${NINTS_SEQ}; do
    python scripts/03_generate_new_data_int.py --experiment $EXPERIMENT --expert $EXPERT --seed $SEED --method $METHOD --nints $NINTS --gpu $GPU
    python scripts/04_train_model.py --experiment $EXPERIMENT --expert $EXPERT --seed $SEED --method $METHOD --nints $NINTS --gpu $GPU
    python scripts/05_train_agent.py --experiment $EXPERIMENT --expert $EXPERT --seed $SEED --method $METHOD --nints $NINTS --gpu $GPU
    python scripts/06_evaluate.py --experiment $EXPERIMENT --expert $EXPERT --seed $SEED --method $METHOD --nints $NINTS --gpu $GPU
done
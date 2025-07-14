#!/bin/bash

set -xeu

N_SAMPLES=${N_SAMPLES:-24576} #24576 #3072 #2457 #6
TP=1
DP=${DP:-8}
WD=${WD:-0}
SORTING=${SORTING:-descending} #ascending #descending #lexicographic #skip

export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1
export VLLM_SCHED_PREFILL_KVC_FREEPCT=31.0

export HARNESS_DISABLE_VLLM_LOGS=1
export VLLM_LOGGING_LEVEL=ERROR

MODEL_PATH=/data/llm/llama2-70b-chat/
DATASET_PATH=/data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
QUANTIZED_WEIGHTS_PATH=quantized/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors/llama.safetensors
QUANTIZATION_PARAM_PATH=/app/kv_cache_scales.json

MLPERF_CONF=/app/mlperf_inference/mlperf.conf
USER_CONF="${USER_CONF:-/lab-mlperf-inference/code/llama2-70b-99.9/mlperf_config_VllmFp8/user.conf}"

SUBMISSION=${SUBMISSION:-0}
if [ "$SUBMISSION" -eq "0" ]; then
    BASE_LOG_DIR="${BASE_LOG_DIR:-${LAB_CLOG}/offline/`date +%m%d-%H%M%S`}"
    LOG_DIR=${BASE_LOG_DIR}/perf
else
    ITER=${ITER:-1}
    TS_NOW=`date +%m%d-%H%M%S`
    TS_RESULTS="${TS_START_BENCHMARKS:-${TS_NOW}}"

    BASE_LOG_DIR="${BASE_LOG_DIR:-${LAB_CLOG}/${TS_RESULTS}/Offline}"
    LOG_DIR=${BASE_LOG_DIR}/performance/run_${ITER}
fi
mkdir -p $LOG_DIR

env | sort >> ${LOG_DIR}/ct-env.txt
cp $USER_CONF ${LOG_DIR}/user.conf

MLPERF_QUANTIZATION_METHOD="${MLPERF_QUANTIZATION_METHOD:-fp8}"
MLPERF_GPU_MEM_UTIL_RATIO="${MLPERF_GPU_MEM_UTIL_RATIO:-0.90}"
MLPERF_PYTHON_BINARY="${MLPERF_PYTHON_BINARY:-/usr/bin/python3}"
MLPERF_KV_CACHE_DTYPE="${MLPERF_KV_CACHE_DTYPE:-fp8}"

${MLPERF_PYTHON_BINARY} /lab-mlperf-inference/code/llama2-70b-99.9/VllmFp8/mainVllmFp8_Offline.py \
    --scenario Offline \
    --output-log-dir ${LOG_DIR} \
    --model-path $MODEL_PATH \
    --mlperf-conf $MLPERF_CONF \
    --user-conf $USER_CONF \
    --total-sample-count $N_SAMPLES \
    --dataset-path $DATASET_PATH \
    --dtype float16 \
    --backend vllm \
    --device cuda:0 \
    --kv-cache-dtype ${MLPERF_KV_CACHE_DTYPE} \
    -tp ${TP} \
    -dp ${DP} \
    --quantization ${MLPERF_QUANTIZATION_METHOD} \
    --quantized-weights-path ${QUANTIZED_WEIGHTS_PATH} \
    --quantization-param-path ${QUANTIZATION_PARAM_PATH} \
    --warmup-duration ${WD} \
    --sorting ${SORTING} \
    --enforce-eager True \
    --gpu-memory-utilization ${MLPERF_GPU_MEM_UTIL_RATIO} \
    2>&1 | tee ${LOG_DIR}/perf.offline.DP${DP}TP${TP}.${N_SAMPLES}.log

#!/bin/bash

: ${3?"usage: ${0} <src> <out> <size>"}
SRC="${1}"
OUT="${2}/word2vec.out"

mkdir -p $(dirname ${SRC})
mkdir -p $(dirname ${OUT})
options="-size ${3} -windows 10 -sample 1e-4 -hs 1 -negative 0 -iter 20 -min-count 1 -cbow 0"
start_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
time ./word2vec/word2vec -train ${SRC} -output ${OUT} ${options}
finish_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
duration=$(($(($(date +%s -d "$finish_time")-$(date +%s -d "$start_time")))))
echo "word2vec execution duration(s): $duration"
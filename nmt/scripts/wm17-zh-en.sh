#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

rm -rf my-zh-en

OUTPUT_DIR="${1:-my-zh-en}"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

#cp /mnt/home/itlgpu/fyk/nmt/temp/training/*.zh-en.* ./$OUTPUT_DIR_DATA
#cp ./test.tgz ${OUTPUT_DIR_DATA}
#cp ./dev.tgz ${OUTPUT_DIR_DATA}

#echo "Downloading Europarl v7. This may take a while..."
#curl -o ${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz \
#  http://www.statmt.org/europarl/v7/de-en.tgz
#
#echo "Downloading Common Crawl corpus. This may take a while..."
#curl -o ${OUTPUT_DIR_DATA}/common-crawl.tgz \
#  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
#

#############################################################################
echo "Downloading dev sets"
curl -o ${OUTPUT_DIR_DATA}/dev.tgz \
  http://data.statmt.org/wmt17/translation-task/dev.tgz

echo "Downloading test sets"
curl -o ${OUTPUT_DIR_DATA}/test.tgz \
  http://data.statmt.org/wmt17/translation-task/test.tgz

echo "Downloading training sets"
curl -o ${OUTPUT_DIR_DATA}/training-parallel-nc-v12.tgz \  
  http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz


## Extract everything
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"
mkdir -p "${OUTPUT_DIR_DATA}/training"
tar -xvzf "${OUTPUT_DIR_DATA}/itraining-parallel-nc-v12.tgz" -C "${OUTPUT_DIR_DATA}/training"


# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/training/news-commentary-v12.zh-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/training/news-commentary-v12.zh-en.zh" \
  > "${OUTPUT_DIR}/train.zh"
wc -l "${OUTPUT_DIR}/train.zh"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

cp ../nmt/scripts/wmt16_de_en/mosesdecoder/ ./my-zh-en/ -r

## Convert SGM files

#Convert newsdev2017 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newsdev2017-zhen-src.zh.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newsdev2017.zh
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newsdev2017-zhen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newsdev2017.en

# Convert newstest2017 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2017-zhen-src.zh.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2017.zh
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2017-zhen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2017.en

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/newsdev20*.zh ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newsdev20*.en ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.zh ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.en ${OUTPUT_DIR}

# Tokenize data
for f in ${OUTPUT_DIR}/*.zh; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l zh -threads 8 < $f > ${f%.*}.tok.zh
done

for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase zh en "${fbase}.clean" 1 80
done

# Generate Subword Units (BPE)
# Clone Subword NMT

if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.zh" "${OUTPUT_DIR}/train.tok.clean.en" | \
    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en zh; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  echo -e "<unk>\n<s>\n</s>" > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.zh" | \
    ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

# Duplicate vocab file with language suffix
cp "${OUTPUT_DIR}/vocab.bpe.32000" "${OUTPUT_DIR}/vocab.bpe.32000.en"
cp "${OUTPUT_DIR}/vocab.bpe.32000" "${OUTPUT_DIR}/vocab.bpe.32000.zh"

echo "All done."

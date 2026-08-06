#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python Dependencies
pip install -r requirements.txt;

conda install mkl
conda install -c conda-forge faiss-cpu

# Install Environment Dependencies via `conda`
# conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command.
# If your server cannot reach Google Drive, download the files manually on
# another machine and drop them into `data/` before rerunning this script —
# the helper below skips any file that already exists.
mkdir -p data;
cd data;

fetch() {
  local fname="$1"
  local gid="$2"
  if [ -f "$fname" ]; then
    echo "[setup] $fname already present, skipping download"
  else
    gdown "https://drive.google.com/uc?id=${gid}" -O "$fname"
  fi
}

if [ "$data" == "small" ]; then
  fetch items_shuffle_1000.json        1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
  fetch items_ins_v2_1000.json         1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
elif [ "$data" == "all" ]; then
  fetch items_shuffle_1000.json        1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
  fetch items_ins_v2_1000.json         1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
  fetch items_shuffle.json             1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
  fetch items_ins_v2.json              1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
fetch items_human_ins.json             14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
cd ..

# Download spaCy large NLP model
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

# Create logging folder + samples of log data
# get_human_trajs () {
#   PYCMD=$(cat <<EOF
# import gdown
# url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
# gdown.download_folder(url, quiet=True, remaining_ok=True)
# EOF
#   )
#   python -c "$PYCMD"
# }
# mkdir -p user_session_logs/
# cd user_session_logs/
# echo "Downloading 50 example human trajectories..."
# get_human_trajs
# echo "Downloading example trajectories complete"
# cd ..
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${SEARCH_CUDA_VISIBLE_DEVICES:-0,5,6,7}"

search_data="${SEARCH_DATA:-$PWD/search_data}"
save_path="${SEARCH_INDEX_DIR:-$search_data}"
index_file="${SEARCH_INDEX_FILE:-$save_path/e5_Flat.index}"
corpus_file="${SEARCH_CORPUS_FILE:-$save_path/wiki-18.jsonl}"
retriever_name="${SEARCH_RETRIEVER_NAME:-e5}"
retriever_path="${SEARCH_RETRIEVER_PATH:-models/e5-base-v2}"
port="${SEARCH_PORT:-8030}"
topk="${SEARCH_TOPK:-3}"

exec python examples/search/retriever/retrieval_server.py \
  --index_path "$index_file" \
  --corpus_path "$corpus_file" \
  --topk "$topk" \
  --retriever_name "$retriever_name" \
  --retriever_model "$retriever_path" \
  --faiss_gpu \
  --port "$port"

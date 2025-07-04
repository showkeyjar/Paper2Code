MODEL_NAME="magistral"
TP_SIZE=2

PAPER_NAME="Carbon2EC"
PDF_PATH="../examples/carbone2ec.pdf" # .pdf
PDF_JSON_PATH="../examples/carbone2ec.json" # .json
PDF_JSON_CLEANED_PATH="../examples/carbone2ec_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/carbone2ec_dscoder"
OUTPUT_REPO_DIR="../outputs/carbone2ec_dscoder_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

python ../codes/1_planning_llm.py \
    --provider ollama \
    --model_name $MODEL_NAME \
    --base_url http://192.168.31.8:11434 \
    --paper_name $PAPER_NAME \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing_llm.py \
    --paper_name $PAPER_NAME \
    --provider ollama \
    --model_name $MODEL_NAME \
    --base_url http://192.168.31.8:11434 \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding_llm.py  \
    --paper_name $PAPER_NAME \
    --provider ollama \
    --model_name $MODEL_NAME \
    --base_url http://192.168.31.8:11434 \
    --tp_size ${TP_SIZE} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \

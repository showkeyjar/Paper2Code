#!/bin/bash

# DeepSeek model configuration
MODEL_NAME="deepseek-coder-33b-instruct"  # Adjust based on available DeepSeek models
API_KEY="your_deepseek_api_key_here"  # Replace with your actual API key
API_ENDPOINT="https://api.deepseek.com/v1/chat/completions"  # Example endpoint, adjust as needed

# Paper configuration
PAPER_NAME="Carbon2EC"
PDF_PATH="../examples/carbone2ec.pdf"  # .pdf
PDF_JSON_PATH="../examples/carbone2ec.json"  # .json
PDF_JSON_CLEANED_PATH="../examples/carbone2ec_cleaned.json"  # _cleaned.json
OUTPUT_DIR="../outputs/carbone2ec_deepseek"
OUTPUT_REPO_DIR="../outputs/carbone2ec_deepseek_repo"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo "Processing paper: $PAPER_NAME"

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH}

echo "------- Running DeepSeek -------"

# Example API call to DeepSeek
# This is a placeholder - adjust the API call according to DeepSeek's API documentation
curl -X POST $API_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful AI assistant that helps with code generation from research papers."
      },
      {
        "role": "user",
        "content": "Process the paper at '"$PDF_JSON_CLEANED_PATH"' and generate the corresponding code."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }' > "$OUTPUT_DIR/deepseek_response.json"

echo "------- Processing Complete -------"
echo "Output saved to: $OUTPUT_DIR"
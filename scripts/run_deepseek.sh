#!/bin/bash
# DeepSeek model configuration
MODEL_NAME="deepseek-reasoner"
API_KEY="your_deepseek_api_key_here"
API_ENDPOINT="https://api.deepseek.com/v1/chat/completions"

# Paper configuration
PAPER_NAME="Carbon2EC"
PDF_PATH="../examples/carbone2ec.pdf"
PDF_JSON_PATH="../examples/carbone2ec.json"
PDF_JSON_CLEANED_PATH="../examples/carbone2ec_cleaned.json"
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

# Create a Python script to handle JSON serialization
PY_SCRIPT=$(cat <<'EOF'
import json
import sys
from pathlib import Path

# Read the paper content
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    paper_content = f.read().strip()

# Create the request payload
payload = {
    "model": sys.argv[2],
    "messages": [
        {
            "role": "system",
            "content": "You are an AI assistant that helps generate code implementations from research papers."
        },
        {
            "role": "user",
            "content": f"Please analyze this research paper and generate the corresponding code implementation. Focus on the core algorithms and methods described. Paper content: {paper_content[:1000]}..."  # Limit content length
        }
    ],
    "temperature": 0.7,
    "max_tokens": 2048
}

# Save the payload to a file
with open(sys.argv[3], 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
EOF
)

# Generate a timestamp for file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REQUEST_FILE="$OUTPUT_DIR/request_${TIMESTAMP}.json"
RESPONSE_FILE="$OUTPUT_DIR/response_${TIMESTAMP}.json"
DEBUG_LOG="$OUTPUT_DIR/curl_debug_${TIMESTAMP}.log"

# Generate the request file using Python for proper JSON handling
echo "$PY_SCRIPT" | python3 - "$PDF_JSON_CLEANED_PATH" "$MODEL_NAME" "$REQUEST_FILE"

echo "Sending request to DeepSeek API..."
echo "Request payload size: $(wc -c < "$REQUEST_FILE") bytes"

# Make the API request
curl -v -X POST "$API_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "@$REQUEST_FILE" \
  -o "$RESPONSE_FILE" \
  2> "$DEBUG_LOG"

# Check the response
HTTP_STATUS=$(grep -oP '^< HTTP/1.1 \K[0-9]+' "$DEBUG_LOG" | tail -1)

if [ "$HTTP_STATUS" = "200" ]; then
    echo "API request successful (HTTP $HTTP_STATUS)"
    
    # Extract the content using jq
    if command -v jq &> /dev/null; then
        jq -r '.choices[0].message.content' "$RESPONSE_FILE" > "$OUTPUT_DIR/generated_code_${TIMESTAMP}.py"
        echo "Generated code saved to: $OUTPUT_DIR/generated_code_${TIMESTAMP}.py"
    else
        echo "jq not installed. Raw response saved to: $RESPONSE_FILE"
    fi
else
    echo "Error: API request failed with HTTP $HTTP_STATUS"
    echo "Check the following files for details:"
    echo "- Request: $REQUEST_FILE"
    echo "- Response: $RESPONSE_FILE"
    echo "- Debug log: $DEBUG_LOG"
    
    # Print the error response
    if [ -f "$RESPONSE_FILE" ]; then
        echo -e "\nError details:"
        cat "$RESPONSE_FILE"
        echo
    fi
fi

echo "------- Processing Complete -------"
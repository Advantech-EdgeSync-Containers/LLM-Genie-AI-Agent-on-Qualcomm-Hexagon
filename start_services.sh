#!/bin/bash
# Script to run Genie LLM server with OpenWebUI integration
set -e

# Load environment variables from .env file if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

cd langchain-agent

# Info log
echo "=========================================="
echo " Starting GENIE LLM server"
echo "------------------------------------------"
echo "Model Name:    $MODEL_NAME"
echo "Host:          $GENIE_LLM_HOST"
echo "Port:          $GENIE_LLM_PORT"
echo "=========================================="

# Run server
nohup uvicorn app:app --host $GENIE_LLM_HOST --port $GENIE_LLM_PORT > ../genie_llm_uvicorn.log 2>&1 &
echo $! > genie_llm_uvicorn.pid
echo "Genie llm server started (PID: $(cat genie_llm_uvicorn.pid))"

#!/bin/bash



# Created by Samir Singh <samir.singh@advantech.com>

# Copyright (c) 2025 Advantech Corporation



# This script is a wrapper that runs the encoded entrypoint script

# The encoding protects the implementation details while allowing execution



# Clear the terminal

clear


LOG_FILE="/workspace/wise-bench.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Append timestamp to start of each run
{
  echo "==========================================================="
  echo ">>> Diagnostic Run Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "==========================================================="
} >> "$LOG_FILE"


# Save original stdout and stderr
exec 3>&1 4>&2

# Redirect stdout & stderr to both console and file (append mode)
exec > >(tee -a "$LOG_FILE") 2>&1

# Simplified script with minimal formatting to avoid ANSI code issues



# Display banner

GREEN='\033[0;32m'

RED='\033[0;31m'

YELLOW='\033[0;33m'

BLUE='\033[0;34m'

CYAN='\033[0;36m'

BOLD='\033[1m'

PURPLE='\033[0;35m'

NC='\033[0m' # No Color



# Display fancy banner

echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"

echo -e "${BLUE}${BOLD}|    ${PURPLE}Advantech_COE Qualcomm® Hardware Diagnostics Tool${BLUE}    |${NC}"

echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"

echo

# Show Advantech COE ASCII logo - with COE integrated

echo -e "${BLUE}"

echo "       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗"

echo "      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝"

echo "      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  "

echo "      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  "

echo "      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗"

echo "      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝"

echo -e "${WHITE}                                  Center of Excellence${NC}"

echo
echo -e "${YELLOW}${BOLD}▶ Starting hardware acceleration tests...${NC}"

echo

# Helper functions

print_header() {

    echo

    echo "+--- $1 ----$(printf '%*s' $((47 - ${#1})) | tr ' ' '-')+"

    echo "|$(printf '%*s' 50 | tr ' ' ' ')|"

    echo "+--------------------------------------------------+"

}



print_success() {

    echo "✓ $1"

}



print_warning() {

    echo "⚠ $1"

}



print_info() {

    echo "ℹ $1"

}



print_table_header() {

    echo "+--------------------------------------------------+"

    echo "| $1$(printf '%*s' $((47 - ${#1})) | tr ' ' ' ')|"

    echo "+--------------------------------------------------+"

}

print_table_row() {
  if [ $# -eq 3 ]; then
    printf "| %-25s | %s | %s |\n" "$1" "$2" "$3"
  elif [ $# -eq 2 ]; then
    printf "| %-25s | %s |\n" "$1" "$2"
  else
    printf "| %-25s | %s |\n" "$1" "$2"
  fi
}


print_table_footer() {

    echo "+--------------------------------------------------+"

}


echo "▶ Setting up hardware acceleration environment..."



# -- Add trap to kill and wait for any background jobs on script exit --
trap 'jobs -p | xargs -r kill 2>/dev/null; wait 2>/dev/null' EXIT INT TERM


# System Information in a fancy tabular format

print_header "SYSTEM INFORMATION"

print_table_header "SYSTEM DETAILS"



# Get system information

KERNEL=$(uname -r)

ARCHITECTURE=$(uname -m)

HOSTNAME=$(hostname)

OS=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "Unknown")

MEMORY_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')

MEMORY_USED=$(free -h | awk '/^Mem:/ {print $3}')

CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2- | sed 's/^[ \t]*//' | head -1 || echo "Unknown")

CPU_CORES=$(nproc --all)

UPTIME=$(uptime -p | sed 's/^up //')



# Print detailed system information in a fancy table

print_table_row "Hostname" "$HOSTNAME"

print_table_row "OS" "$OS"

print_table_row "Kernel" "$KERNEL"

print_table_row "Architecture" "$ARCHITECTURE"

print_table_row "CPU" "$CPU_MODEL ($CPU_CORES cores)"

print_table_row "Memory" "$MEMORY_USED used of $MEMORY_TOTAL"

print_table_row "Uptime" "$UPTIME"

print_table_row "Date" "$(date "+%a %b %d %H:%M:%S %Y")"

print_table_footer

# QNN Information with fancy graphics

print_header "QNN INFORMATION"

# Show fancy QNN ASCII art
echo -e "${YELLOW}"

echo "      ██████╗ ███╗   ██╗███╗   ██╗"

echo "     ██╔═══██╗████╗  ██║████╗  ██║"

echo "     ██║   ██║██╔██╗ ██║██╔██╗ ██║"

echo "     ██║   ██║██║╚██╗██║██║╚██╗██║"

echo "     ╚██████╔╝██║ ╚████║██║ ╚████║"

echo "      ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝"

echo "          ██║"

echo "           ╚╝"

echo -e "${NC}"



echo -ne "▶ Checking QNN backend... "
for i in {1..5}; do
    for c in ⣾ ⣽ ⣻ ⢿ ⡿ ⣟ ⣯ ⣷; do
        echo -ne "\b$c"; sleep 0.1
    done
done
echo -ne "\b✓\n"

# QNN SDK Version Check
# Capture full output (stdout and stderr)
full_out=$(qnn-net-run --version 2>&1)

# Extract the line containing the SDK version
QNN_SDK_LINE=$(echo "$full_out" | grep "QNN SDK v")
# Parse out the version number (after 'v')
QNN_SDK_VER=$(echo "$QNN_SDK_LINE" | awk -F'v' '{print $2}')

print_table_header "QNN RUNTIME DETAILS"
if [[ -n "$QNN_SDK_VER" ]]; then
	QNN_SDK_VER=$(echo "$QNN_SDK_VER" | cut -d'.' -f1-3)
	print_table_row "QNN SDK Version" "$QNN_SDK_VER"
else
    print_table_row "QNN SDK Version" "Not Found"
fi

echo

# Run qnn-platform-validator, capture output
QNN_OUTPUT=$(qnn-platform-validator --backend all --coreVersion --libVersion --testBackend 2>&1)

# Extract Backend blocks info - GPU and DSP
for backend in GPU DSP; do
  # Extract the block between "Backend = $backend" and the next "Backend ="
  block=$(echo "$QNN_OUTPUT" | sed -n "/Backend = $backend/,/Backend =/p" | sed '$d')

  # Parse each field from the block
  backend_hw=$(echo "$block" | grep -i "Backend Hardware" | awk -F':' '{print $2}' | xargs)
  backend_lib=$(echo "$block" | grep -i "Backend Libraries" | awk -F':' '{print $2}' | xargs)
  lib_version=$(echo "$block" | grep -i "Library Version" | awk -F':' '{print $2}' | xargs)
  core_version=$(echo "$block" | grep -i "Core Version" | awk -F':' '{print $2}' | xargs)
  unit_test=$(echo "$block" | grep -i "Unit Test" | awk -F':' '{print $2}' | xargs)

  if [[ "$backend" == "GPU" ]]; then
    QNN_GPU_TEST_RESULT="$unit_test"
  elif [[ "$backend" == "DSP" ]]; then
    QNN_DSP_TEST_RESULT="$unit_test"
  fi

  print_table_row "Backend" "$backend"
  print_table_row "Backend Hardware" "$backend_hw"
  print_table_row "Backend Libraries" "$backend_lib"
  print_table_row "Library Version" "$lib_version"
  print_table_row "Core Version" "$core_version"
  print_table_row "Unit Test" "$unit_test"
  echo
done

print_table_footer

# SNPE Information with ASCII logo and details

print_header "SNPE INFORMATION"

echo -e "${CYAN}"
echo "       ██████╗███╗   ██╗██████╗ ███████╗"
echo "      ██╔════╝████╗  ██║██╔══██╗██╔════╝"
echo "       ████╗  ██╔██╗ ██║██████╔╝█████╗  "
echo "      ╚════██╗██║╚██╗██║██╔═══╝ ██╔══╝  "
echo "      ██████╔╝██║ ╚████║██║     ███████╗"
echo "      ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝"
echo -e "${NC}"


# Animated detection
echo -ne "▶ Checking SNPE runtimes... "
for i in {1..5}; do
    for c in ⣾ ⣽ ⣻ ⢿ ⡿ ⣟ ⣯ ⣷; do
        echo -ne "\b$c"; sleep 0.1
    done
done
echo -ne "\b✓\n"

# SNPE SDK Version Check
# Capture full output (both stdout and stderr)
snpe_full_out=$(snpe-net-run --version 2>&1)

# Extract the version line that starts with "SNPE v"
SNPE_SDK_LINE=$(echo "$snpe_full_out" | grep "^SNPE v")

# Trim off the 'SNPE v' prefix
SNPE_SDK_VER=$(echo "$SNPE_SDK_LINE" | awk -F'v' '{print $2}')

print_table_header "SNPE RUNTIME DETAILS"

if [[ -n "$SNPE_SDK_VER" ]]; then
    SNPE_SDK_VER=$(echo "$SNPE_SDK_VER" | cut -d'.' -f1-3)
    print_table_row "SNPE SDK Version" "$SNPE_SDK_VER "
else
    print_table_row "SNPE SDK Version" "Not Found"
fi

# Run snpe-platform-validator with timeout and capture output
# Capture clean SNPE validator output (nulls removed)
SNPE_OUTPUT=$(timeout 1 snpe-platform-validator --runtime all --coreVersion --libVersion --testRuntime 2>/dev/null | tr -d '\0')

parse_snpe_runtime() {
    local RUNTIME_NAME=$1
    # No warnings, clean output
    UNIT_TEST_LINE=$(echo "$SNPE_OUTPUT" | grep "Unit Test on the runtime $RUNTIME_NAME")
    UNIT_TEST_RESULT=$(echo "$UNIT_TEST_LINE" | awk -F':' '{print $2}' | xargs)

    if [[ -n "$UNIT_TEST_RESULT" ]]; then
        print_table_row "$RUNTIME_NAME Runtime Unit Test" "$UNIT_TEST_RESULT"
    else
        print_table_row "$RUNTIME_NAME Runtime Unit Test" "Not Detected"
    fi
}

# Parse GPU and DSP runtime results
parse_snpe_runtime "GPU"
parse_snpe_runtime "DSP"

print_table_footer



print_header "FINAL SUMMARY TABLE"
print_table_header "Summary Results"

# QNN GPU Backend
if [[ "$QNN_GPU_TEST_RESULT" == "Passed" ]]; then
    print_table_row "QNN GPU Backend" "$QNN_SDK_VER" "Supported"
else
    print_table_row "QNN GPU Backend" "$QNN_SDK_VER" "Not Supported"
fi

# QNN DSP Backend
if [[ "$QNN_DSP_TEST_RESULT" == "Passed" ]]; then
    print_table_row "QNN DSP Backend" "$QNN_SDK_VER" "Supported"
else
    print_table_row "QNN DSP Backend" "$QNN_SDK_VER" "Not Supported"
fi

# SNPE GPU
if echo "$SNPE_OUTPUT" | grep -q "Unit Test on the runtime GPU.*Passed"; then
    print_table_row "SNPE GPU Runtime" "$SNPE_SDK_VER" "Supported"
else
    print_table_row "SNPE GPU Runtime" "$SNPE_SDK_VER" "Not Supported"
fi

# SNPE DSP
if echo "$SNPE_OUTPUT" | grep -q "Unit Test on the runtime DSP.*Passed"; then
    print_table_row "SNPE DSP Runtime" "$SNPE_SDK_VER" "Supported"
else
    print_table_row "SNPE DSP Runtime" "$SNPE_SDK_VER" "Not Supported"
fi

# Genie
Genie_version=$(python3 -c "import ctypes; lib=ctypes.CDLL('libGenie.so'); print(f'{lib.Genie_getApiMajorVersion()}.{lib.Genie_getApiMinorVersion()}.{lib.Genie_getApiPatchVersion()}.')" 2>/dev/null || echo "Not Installed")

if [[ "$Genie_version" != "Not Installed" ]]; then
    print_table_row "Genie Runtime" "$Genie_version" "Supported"
else
    print_table_row "Genie Runtime" "$Genie_version" "Not Supported"
fi

#############################################
print_header "GENIE LLM + LANGCHAIN + FAISS CHECK"

MAX=3
GENIE_STATUS=0
INFERENCE_STATUS=0
LANGCHAIN_STATUS=0
FAISS_STATUS=0

# ---- Check if GENIE is working ----
if command -v genie-t2t-run >/dev/null 2>&1; then
    print_table_row "GENIE Binary" "✓ Available"
    GENIE_STATUS=1
    MAX=$((MAX + 1))

    # ---- Run basic inference ----
    RESPONSE=$( genie-t2t-run -c "$MODEL_CONFIG" -p "Hi" 2>&1)

    if echo "$RESPONSE" | grep -q "Using libGenie"; then
        INFERENCE_STATUS=1
        SHORT_MSG=$(echo "$RESPONSE" | grep -A2 "\[BEGIN\]" | tail -n 1 | cut -c1-60)
        [[ -z "$SHORT_MSG" ]] && SHORT_MSG="Response detected"
        print_table_row "GENIE Test Inference (Hi)" "✓ $SHORT_MSG"
    elif echo "$RESPONSE" | grep -qi "error"; then
        print_table_row "GENIE Test Inference (Hi)" "⚠ Error during run"
    else
        print_table_row "GENIE Test Inference (Hi)" "⚠ No valid response"
    fi
else
    print_table_row "GENIE Binary" "⚠ Not Found"
fi



# ---- Check LangChain installation ----
if python3 -c "import langchain" &>/dev/null; then
    LANGCHAIN_STATUS=1
    print_table_row "LangChain" "✓ Available"
else
    print_table_row "LangChain" "⚠ Not Available"
fi


# ---- Check FAISS installation ----
if python3 -c "import faiss" &>/dev/null; then
    FAISS_STATUS=1
    print_table_row "FAISS" "✓ Available"
else
    print_table_row "FAISS" "⚠ Not Available"
fi



# ---- Calculate overall score ----
TOTAL=$((GENIE_STATUS + INFERENCE_STATUS + LANGCHAIN_STATUS + FAISS_STATUS))
PERCENTAGE=$((TOTAL * 100 / MAX))

print_table_row "Overall Score" "$PERCENTAGE% ($TOTAL/$MAX)"


# ---- Visual progress bar ----
BAR_SIZE=20
FILLED=$((BAR_SIZE * TOTAL / MAX))
EMPTY=$((BAR_SIZE - FILLED))

BAR=""
for ((i=0; i<FILLED; i++)); do
    BAR="${BAR}█"
done
for ((i=0; i<EMPTY; i++)); do
    BAR="${BAR}░"
done

print_table_row "Progress" "$BAR"




print_table_footer



echo -e "${BOLD}>>> Diagnostic Completed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo


# At the end, restore original stdout and stderr
exec >&3 2>&4
exec 3>&- 4>&-

exit 0

#!/usr/bin/env bash
# Run:
#   chmod +x setup.sh
#   ./setup.sh
# After the script finishes, activate environment
#  source .venv/bin/activate

set -e
# Color setup
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info() {
    echo -e "${CYAN}[INFO]${RESET} $*";
}
success() {
    echo -e "${GREEN}[OK]${RESET} $*";
}
warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*";
}
error() {
    echo -e "${RED}[ERROR]${RESET} $*";
}

echo -e "Setup Starting...\n";

# 0) Check python is installed
#PYTHON3
info "Checking python3 is installed...";
command -v python3 >/dev/null 2>&1 || { error "Python3 is not installed. Please install Python3 and try again."; exit 1; }
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')");
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1);
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2);
if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 12) ]]; then
    error "Python 3.12+ required (found $PYTHON_VERSION)."
fi
success "Python3 is installed (version $PYTHON_VERSION).";
#PIP3
command -v pip3 >/dev/null 2>&1 || error "pip3 not found"
success "pip3 is installed."

# 1) Create virtual environment
VENV_DIR=".wine_env"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment directory '$VENV_DIR' already exists. Skipping creation."
    warn "To rebuild from scratch: rm -rf $VENV_DIR && ./setup.sh"
else
    info "Creating virtual environment in './$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    success "Virtual environment created"
fi

source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# 2) Install Packets
info "Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"
#Packets Installation
info "Installing packages from requirements.txt..."
pip install -r requirements.txt -q
success "Packages installed"

success "Project setup completed!"

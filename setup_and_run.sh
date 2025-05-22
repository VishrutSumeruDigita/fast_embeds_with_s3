#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up environment for face embedding generation from S3...${NC}"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${YELLOW}Installing pyenv...${NC}"
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH for current session
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    
    # Add pyenv to shell profile
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    
    echo -e "${GREEN}Pyenv installed successfully!${NC}"
else
    echo -e "${GREEN}Pyenv is already installed.${NC}"
fi

# Make sure pyenv is in PATH
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.10.17 using pyenv
echo -e "${YELLOW}Installing Python 3.10.17...${NC}"
pyenv install -s 3.10.17
pyenv local 3.10.17
echo -e "${GREEN}Python 3.10.17 installed successfully!${NC}"

# Create and activate virtual environment
VENV_DIR="venv"
echo -e "${YELLOW}Creating virtual environment in ${VENV_DIR}...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
    echo -e "${GREEN}Virtual environment created successfully!${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed successfully!${NC}"

# Check if .env file exists and has AWS credentials
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.template .env
    echo -e "${RED}Please edit the .env file and add your AWS credentials before running this script again.${NC}"
    exit 1
fi

# Check if AWS credentials are set in .env
if grep -q "AWS_ACCESS_KEY_ID=$" .env || grep -q "AWS_SECRET_ACCESS_KEY=$" .env; then
    echo -e "${RED}Error: AWS credentials not set in .env file!${NC}"
    echo -e "${RED}Please edit the .env file and add your AWS credentials before running this script again.${NC}"
    exit 1
fi

# Run the face embedding script
echo -e "${YELLOW}Running face embedding generation script...${NC}"
python s3_face_embeddings.py

echo -e "${GREEN}Script execution completed!${NC}"

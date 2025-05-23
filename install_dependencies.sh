#!/bin/bash

# Face Embedding Project Dependencies Installation Script
# This script installs all necessary dependencies for the face embedding project

echo "Starting installation of dependencies for face embedding project..."

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Python 3.10 and pip if not already installed
echo "Installing Python 3.10 and pip..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Install system dependencies for OpenCV and other packages
echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.template" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file with your AWS credentials and other settings."
fi

echo "Installation complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"

#!/usr/bin/env bash

set -e
echo "Configuring Git global username and email..."
git config --global user.name "NguyenHongSon"
git config --global user.email "sonnguyen.15012004@gmail.com"
echo "Git global username and email configured."
echo "--------------------------------------------------"
echo "Install required packages..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "Required packages installed."
echo "--------------------------------------------------"
echo "Setup completed successfully."
#!/bin/bash

# Activate venv and run app
source oled/bin/activate
echo "🚀 Starting OLED Assistant..."

# Run from src directory for correct imports
cd src
streamlit run app.py

#!/bin/bash
# AI-Driven OLED Assistant - Setup Script
# Usage: ./setup.sh
echo "================================================="
echo "🔬 AI-Driven OLED Assistant Setup"
echo "================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install it first."
    exit 1
fi
echo "✅ Python3 found"

# 2. Create Virtual Environment
echo "📦 Creating virtual environment (oled)..."
python3 -m venv oled
source oled/bin/activate
echo "✅ Virtual environment activated"

# 3. Install Dependencies
echo "⬇️ Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# 4. Check & Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed."
    echo "   Please download it from https://ollama.com/download"
    echo "   After installing, run this script again."
    exit 1
else
    echo "✅ Ollama found"
fi

# 5. Pull Mistral Model
echo "🧠 Pulling Mistral-Nemo model (this may take a while)..."
ollama pull mistral-nemo
echo "✅ Model ready"

echo "================================================="
echo "🎉 Setup Complete!"
echo "Run './run_app.sh' to start the assistant."
echo "================================================="

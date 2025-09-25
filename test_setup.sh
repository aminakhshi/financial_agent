#!/bin/bash

echo "Testing Financial Agent Setup..."
echo "================================="

# Test 1: Check if required files exist
echo "1. Checking required files..."
required_files=(
    "Dockerfile"
    "docker-compose.yml" 
    "requirements.txt"
    ".gitignore"
    ".env.example"
    "src/main.py"
    "src/config/settings.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "$file exists"
    else
        echo "$file missing"
    fi
done

# Test 2: Check Python syntax
echo -e "\n2. Checking Python syntax..."
python_files=$(find src -name "*.py")
for file in $python_files; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "$file syntax OK"
    else
        echo "$file has syntax errors"
        python -m py_compile "$file"
    fi
done

# Test 3: Check if environment variables are being loaded
echo -e "\n3. Testing environment variable loading..."
python -c "
import sys
sys.path.append('src')
try:
    from config.settings import API_KEYS, DATABASE_CONFIG
    print('Settings module loads successfully')
    print('API_KEYS configuration found')
    print('DATABASE_CONFIG configuration found')
except ImportError as e:
    print(f'Import error: {e}')
except Exception as e:
    print(f'Configuration error: {e}')
"

echo -e "\nTest completed!"

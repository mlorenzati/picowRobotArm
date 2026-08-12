#!/bin/bash

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# pyinstaller \
#     --clean \
#     --noconfirm \
#     --onefile \
#     --windowed \
#     --name teleop \
#     teleop.py

echo
echo "Executable generated in dist/"
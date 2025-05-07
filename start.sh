#!/usr/bin/env bash
set -euo pipefail

echo "→ Starting frontend…"
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!

echo "→ Starting backend…"
cd ../backend

if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment…"
  python3 -m venv venv
fi

echo "→ Activating virtual environment…"
# shellcheck source=/dev/null
source venv/bin/activate

pip3 install -r requirements.txt

flask --app app run --debug

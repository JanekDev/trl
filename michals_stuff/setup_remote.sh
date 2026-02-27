apt-get update && apt-get install -y tmux

curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env" && git clone https://github.com/JanekDev/trl.git ~/trl && cd ~/trl && uv sync --all-extras
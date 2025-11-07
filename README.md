# Lumen
Основной файл - Lumen_bot.py
Чтобы запустить его локально (на Linux-системе) нужно:
в терминале последовательно запустить:
sudo apt update && sudo apt install python3-venv python3-full -y
python3 -m venv tg_bot_env
source tg_bot_env/bin/activate
pip install --upgrade pip
pip install python-telegram-bot langchain-gigachat langgraph python-dotenv

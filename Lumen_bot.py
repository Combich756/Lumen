
import os
import asyncio
from typing import Sequence

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv()  # загружает .env в os.environ

# === Настройки (введите здесь или через переменные окружения) 
TELEGRAM_BOT_TOKEN = "8250825015:AAE9nIh5RLmbjNFl2yS0m3sBUKhfi3VJXd8" 
GIGACHAT_CREDENTIALS = "MDE5OTc4MTEtM2NjMS03ODNkLTkxYzAtMmM4MzZhN2UxNzM2OmE0YmY3NTdkLTkxNjItNGRjNi04ZDA1LTBiOTM4ZTRjM2JjOA=="    

# === Системный промпт (измените по желанию!) ===
SYSTEM_PROMPT = (
    "Ты — дружелюбный и умный ассистент. Отвечай чётко, по делу и на русском языке. "
    "Если не знаешь ответа — скажи, что не знаешь."
)

# === Инициализация модели ===
model = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-Max",
    verify_ssl_certs=False,
)

# === Состояние чата ===
class ChatState(TypedDict):
    messages: Annotated[Sequence, add_messages]

# === Создание графа с памятью ===
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

def call_model(state: ChatState):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": [response]}

workflow = StateGraph(state_schema=ChatState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# === Обработчики Telegram ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Меня зовут Lumen! Я чат-бот на базе GigaChat. Помогу тебе составить учебное расписание. Напиши мне что-нибудь — и я отвечу!\n\n"
        f"Текущий системный промпт:\n> {SYSTEM_PROMPT}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = update.message.text.strip()
    config = {"configurable": {"thread_id": user_id}}

    try:
        output = app.invoke({"messages": [HumanMessage(content=text)]}, config)
        response = output["messages"][-1].content
        await update.message.reply_text(response)
    except Exception as e:
        print(f"Ошибка: {e}")
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

# === Запуск бота ===
async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот запущен! Напишите ему в Telegram.")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    try:
        # Блокируем выполнение до KeyboardInterrupt
        await asyncio.Event().wait()
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

# Запуск
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
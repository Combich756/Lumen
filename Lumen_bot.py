
import os
import asyncio
from typing import Sequence

from telegram import Update, InputFile 
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import pandas as pd
import json
import re
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()  # загружает .env в os.environ

# === Настройки (введите здесь или через переменные окружения) 
TELEGRAM_BOT_TOKEN = "8250825015:AAE9nIh5RLmbjNFl2yS0m3sBUKhfi3VJXd8" 
GIGACHAT_CREDENTIALS = "MDE5OTc4MTEtM2NjMS03ODNkLTkxYzAtMmM4MzZhN2UxNzM2OmE0YmY3NTdkLTkxNjItNGRjNi04ZDA1LTBiOTM4ZTRjM2JjOA=="    

# === Системный промпт (измените по желанию!) ===
SYSTEM_PROMPT = (
    "Ты — интеллектуальный помощник по обучению. Твоя основная задача — составлять персонализированные учебные планы в виде структурированной таблицы на основе требований пользователя. Учебные планы составляй в виде строго структурированной Markdown-таблицы. "
    "Когда составляешь таблицу, следуй этим правилам БЕЗ ИСКЛЮЧЕНИЙ:\n"
    "1. Таблица ДОЛЖНА начинаться со строки '### Учебный план' и содержать РОВНО 7 колонок в этом порядке:\n"
    "   № | Тема/Модуль | Цель изучения | Рекомендуемые ресурсы | Формат занятий | Продолжительность | Рекомендуемая дата завершения\n"
    "2. В колонке 'Рекомендуемые ресурсы' используй ТОЛЬКО формат ссылок Markdown: [Название](URL). "
    "   Не добавляй текст вне скобок. Если ресурсов несколько — перечисли через запятую: [A](url1), [B](url2).\n"
    "3. В колонке 'Продолжительность' указывай ТОЛЬКО число и единицу: '1 неделя', '2 недели', '3 недели'. Никаких 'Неделя' без цифры.\n"
    "4. В колонке 'Рекомендуемая дата завершения' используй формат: 'неделя 1', 'недели 2-3', 'недели 8-9'.\n"
    "Если не знаешь ответа — скажи, что не знаешь. ДО составления таблицы выясни у пользователя всю недостающую информацию задавая уточняющие вопросы"
)

# Парсинг таблицы
def parse_markdown_table_to_df(text: str) -> pd.DataFrame:
    """
    Извлекает первую Markdown-таблицу из текста.
    Игнорирует строки до таблицы (например, '### Учебный план').
    Поддерживает пробелы вокруг | и внутри ячеек.
    """
    lines = text.strip().split('\n')
    table_lines = []
    table_started = False

    for line in lines:
        stripped = line.strip()
        # Игнорируем всё до первой строки, начинающейся с |
        if not table_started:
            if stripped.startswith('|') and len(stripped) > 3:
                table_started = True
            else:
                continue
        # Продолжаем собирать, пока идут строки таблицы
        if stripped.startswith('|') and stripped.endswith('|'):
            # Пропускаем пустые строки вроде '|   |'
            cells = [cell.strip() for cell in stripped.split('|')[1:-1]]
            if any(cell for cell in cells):  # хотя бы одна непустая ячейка
                table_lines.append(cells)
        elif table_started:
            # Прерываемся при первой не-таблице строке
            break

    if len(table_lines) < 2:
        raise ValueError("Недостаточно строк для таблицы (требуется заголовок + ≥1 строка)")

    headers = table_lines[0]
    # Фильтруем строки-разделители (|---| и подобные)
    data_rows = [
        row for row in table_lines[1:]
        if not all(re.fullmatch(r'-+', cell) for cell in row)
    ]

    if not data_rows:
        raise ValueError("Нет данных в таблице (только заголовок и разделитель)")

    return pd.DataFrame(data_rows, columns=headers)

def extract_links(text: str) -> list:
    """
    Извлекает все [title](url) из строки.
    Обрезает пробелы в title и url.
    Если есть текст вне скобок — сохраняет в отдельном поле 'context'.
    """
    LINK_PATTERN = r'\[([^\]]+)\]\(\s*([^)]*?)\s*\)'
    matches = re.findall(LINK_PATTERN, text)
    links = [{"title": title.strip(), "url": url.strip()} for title, url in matches]

    # Если есть текст вне ссылок — сохраняем как контекст
    text_without_links = re.sub(LINK_PATTERN, '', text).strip()
    if text_without_links and not re.match(r'^[\s,]*$', text_without_links):
        return [{"context": text_without_links, "links": links}] if links else [{"raw": text.strip()}]
    
    return links if links else [{"raw": text.strip()}]

# Конвертация в Excel-файл
def df_to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    try:
        # Создаём копию, чтобы не менять оригинал
        df_out = df.copy()

        # Добавляем колонку с распарсенными ресурсами
        if 'Рекомендуемые ресурсы' in df_out.columns:
            df_out['resources'] = df_out['Рекомендуемые ресурсы'].apply(extract_links)

        # Опционально: нормализуем длительность и дату
        if 'Продолжительность' in df_out.columns:
            def parse_duration(s):
                if not s: return 1
                m = re.search(r'(\d+)', s)
                return int(m.group(1)) if m else 1
            df_out['duration_weeks'] = df_out['Продолжительность'].apply(parse_duration)

        if 'Рекомендуемая дата завершения' in df_out.columns:
            def parse_week_range(s):
                if not s: return {"start": 1, "end": 1}
                m = re.search(r'(\d+)(?:\s*-\s*(\d+))?', s)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else start
                    return {"start": start, "end": end}
                return {"raw": s}
            df_out['completion'] = df_out['Рекомендуемая дата завершения'].apply(parse_week_range)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='План', index=False)
            worksheet = writer.sheets['План']
            for i, col in enumerate(df_out.columns, start=1):
                max_len = max(df_out[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(64 + i)].width = min(max_len, 50)
        output.seek(0)
        return output

    except Exception as e:
        logging.exception("Ошибка в df_to_excel_bytes")
        raise









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


# === Telegram обработчики ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Меня зовут Lumen. Я помогу тебе составить план обучения 📚\n"
        "Напиши, что хочешь изучить и за какое время."
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)  # ← ДОЛЖНО БЫТЬ СТРОКОЙ для thread_id
    # Очистка состояния: сброс чекпоинта для пользователя
    config = {"configurable": {"thread_id": user_id}}
    # LangGraph не имеет прямого метода 'clear', но можно перезаписать состояние
    await app.aupdate_state(config, {"messages": []})  # асинхронное обновление
    await update.message.reply_text("Память очищена!")

import logging  # убедитесь, что импортировано

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = update.message.text.strip()
    config = {"configurable": {"thread_id": user_id}}

    try:
        thinking_msg = await update.message.reply_text("Пожалуйста подождите... ⏳")

        # Вызов LangGraph-агента
        try:
            output = await asyncio.wait_for(
                app.ainvoke({"messages": [HumanMessage(content=text)]}, config),
                timeout=90.0
            )
        except asyncio.TimeoutError:
            await thinking_msg.edit_text("Слишком долго думаю... Попробуйте уточнить запрос.")
            return

        plan_text = output["messages"][-1].content

        # → Конвертация в Excel
        try:
            df = parse_markdown_table_to_df(plan_text)
            excel_bytes = df_to_excel_bytes(df)
            await update.message.reply_document(
                document=InputFile(excel_bytes, filename="plan.xlsx"),
                caption="Ваш учебный план в Excel"
            )
        except Exception as e:
            logging.exception("Ошибка конвертации в Excel")
            await update.message.reply_text(
                "\n\n" + plan_text[:4000]
            )
            return

    except Exception as e:
        # ← ВНЕШНИЙ except: обрабатывает всё остальное (например, ошибки в LangGraph, памяти, сети)
        logging.exception(f"Критическая ошибка при обработке запроса от {user_id}")
        try:
            await thinking_msg.edit_text("Произошла внутренняя ошибка. Попробуйте позже.")
        except:
            await update.message.reply_text("Произошла внутренняя ошибка.")

        # Опционально: также отправить JSON (для логирования/резервной копии)
        # json_str = df.to_json(orient='records', force_ascii=False, indent=2)
        # await update.message.reply_document(
        #     document=BytesIO(json_str.encode()),
        #     filename="plan.json"
        # )

    except Exception as e:
        logging.exception(f"Критическая ошибка для {user_id}")
        await update.message.reply_text("Извините, произошла ошибка. Попробуйте позже.")

        
# === Запуск бота ===
async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот запущен! Можно написать ему в Telegram @lumen52_bot")
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

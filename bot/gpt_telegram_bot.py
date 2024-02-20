import os
import fitz
import openai
import logging
import aiohttp
import asyncio
from gtts import gTTS
from io import BytesIO
import tempfile
from helpers import download_audio, convert_audio_to_wav
from telegram import Update
import subprocess
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler, CallbackQueryHandler,
)
from telegram.ext import MessageHandler, filters
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Установка API ключа OpenAI
openai.api_key = "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"

# Токен Telegram Bot API
telegram_token = "6553657042:AAE6QoOa0MmYDNJ9PduF0NM1asuYntNsztY"

# Версия OpenAI API
openai_version = "gpt-3.5-turbo-1106"

messages_list = []

user_speed_settings = {}


def append_history(content, role):
    messages_list.append({"role": role, "content": content})
    return messages_list


def clear_history():
    messages_list.clear()
    return messages_list


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thinking = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="✨✨✨Генерируем ответ✨✨✨"
    )
    append_history(update.message.text, "user")

    response = generate_gpt_response()

    append_history(response, "assistant")
    if update.message.voice:
        await send_voice_message(update.effective_chat.id, response, context)
    else:
        await context.bot.deleteMessage(
            message_id=thinking.message_id, chat_id=update.message.chat_id
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def speed(update: Update, _: ContextTypes.DEFAULT_TYPE):

    keyboard = [
        [InlineKeyboardButton("Быстро (1.75x)", callback_data='fast'),
         InlineKeyboardButton("Обычно", callback_data='normal'),
         InlineKeyboardButton("Медленно (0.75x)", callback_data='slow')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Выберите скорость воспроизведения:', reply_markup=reply_markup)


def change_speed(input_audio_path, output_audio_path, speed=1.0):
    # Убедитесь, что speed передается корректно и используется в команде ниже
    command = [
        "ffmpeg",
        "-y",  # Автоматическое подтверждение перезаписи файлов
        "-i", input_audio_path,
        "-filter:a", f"atempo={speed}",
        "-vn", output_audio_path
    ]

    subprocess.run(command, check=True)

async def set_speed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    speed = query.data  # Получаем данные callback
    user_speed_settings[query.from_user.id] = speed  # Сохраняем предпочтение скорости для пользователя
    speed_text = "Быстро (1.75x)" if speed == "fast" else "Медленно (0.75x)" if speed == "slow" else "Обычно"
    await query.edit_message_text(text=f"Скорость воспроизведения установлена на: {speed_text}")


async def send_voice_message(chat_id, text, context):
    lang = 'ru'
    speed_setting = user_speed_settings.get(chat_id, 'normal')  # Получаем настройку скорости для пользователя

    # Определяем скорость воспроизведения
    speed = 1.0  # Стандартная скорость
    if speed_setting == 'fast':
        speed = 1.75
    elif speed_setting == 'slow':
        speed = 0.75

    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_input:
        tts.save(tmp_input.name)  # Сохраняем исходное аудио во временный файл

        # Создаем еще один временный файл для вывода
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_output:
            # Применяем изменение скорости к исходному файлу и сохраняем результат во временный файл
            change_speed(tmp_input.name, tmp_output.name, speed=speed)

            # После выполнения change_speed, результат будет сохранён в tmp_output.name,
            # и мы можем использовать этот файл для отправки аудио.
            with open(tmp_output.name, 'rb') as output_audio:
                await context.bot.send_voice(chat_id=chat_id, voice=output_audio)

    # Удаляем временные файлы
    os.remove(tmp_input.name)
    os.remove(tmp_output.name)


async def process_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    transcript = await get_audio_transcription(update, context)
    thinking = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="✨✨✨Генерируем ответ✨✨✨"
    )
    append_history(transcript, "user")

    response = generate_gpt_response()

    append_history(response, "assistant")
    await context.bot.delete_message(
        chat_id=update.effective_chat.id, message_id=thinking.message_id
    )
    await send_voice_message(update.effective_chat.id, response, context)


async def download_pdf(document, context):
    # Получаем объект File
    file = await context.bot.get_file(document.file_id)
    file_url = file.file_path

    # Используем aiohttp для скачивания файла
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            if resp.status == 200:
                # Задаем путь для сохранения файла, используя временные файлы
                file_path = os.path.join(tempfile.gettempdir(), document.file_unique_id + ".pdf")
                with open(file_path, 'wb') as fd:
                    while True:
                        chunk = await resp.content.read(1024)
                        if not chunk:
                            break
                        fd.write(chunk)
                return file_path


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text



async def process_pdf_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document  # Получаем документ из сообщения
    processing_message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Файл обрабатывается, пожалуйста, подождите..."
    )

    # Отправляем исчезающее сообщение о начале обработки файла
    pdf_file_path = await download_pdf(document, context)
    try:
        if pdf_file_path is None:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Ошибка при загрузке PDF файла. Пожалуйста, повторите попытку.")
            return

        # Извлекаем текст из PDF
        extracted_text = extract_text_from_pdf(pdf_file_path)

        if not extracted_text:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Не удалось извлечь текст из PDF файла.")
            return

        # Генерируем обобщение из извлеченного текста
        summary = await generate_gpt_summary(extracted_text)

        if summary is None:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Ошибка при генерации обобщения из PDF файла. Пожалуйста, повторите попытку.")
            return

        await send_voice_message(update.effective_chat.id, summary, context)  # Отправляем обобщение
    finally:
        # Удаляем сообщение о обработке после завершения всех операций
        await asyncio.sleep(5)  # Даем пользователю время увидеть сообщение о обработке
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=processing_message.message_id)


async def generate_gpt_summary(text):
    try:
        # Оборачиваем синхронный вызов в asyncio.to_thread для асинхронного выполнения
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            messages=[
                {"role": "system", "content": "Summarize the following text:"},
                {"role": "user", "content": text}
            ],
            model=openai_version,
            temperature=0.7,
            max_tokens=150
        )
        summary = response.choices[0].message["content"]
        return summary
    except Exception as e:
        logging.error(f"Ошибка при генерации обобщения: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

def generate_gpt_response():
    messages = [{"role": "user", "content": msg["content"]} for msg in messages_list]
    completion = openai.ChatCompletion.create(model=openai_version, messages=messages)
    return completion.choices[0].message["content"]



async def get_audio_transcription(update, context):
    new_file = await download_audio(update, context)
    voice = convert_audio_to_wav(new_file)
    transcript = openai.Audio.transcribe("whisper-1", voice)
    return transcript["text"]


async def reset_history(update, context):
    clear_history()
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Messages history cleaned"
    )
    return messages_list



if __name__ == "__main__":
    application = ApplicationBuilder().token(telegram_token).build()
    text_handler = MessageHandler(
        filters.TEXT & (~filters.COMMAND), process_text_message
    )

    application.add_handler(text_handler)
    application.add_handler(CommandHandler("reset", reset_history))

    audio_handler = MessageHandler(filters.VOICE, process_audio_message)
    application.add_handler(audio_handler)
    application.add_handler(CallbackQueryHandler(set_speed))
    application.add_handler(CommandHandler("speed", speed))
    document_handler = MessageHandler(filters.Document.MIME_TYPE("application/pdf"), process_pdf_message)
    application.add_handler(document_handler)

    application.run_polling()



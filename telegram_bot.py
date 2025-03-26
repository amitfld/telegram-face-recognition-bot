import os
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Define the buttons for the custom keyboard
button_labels = ['Hello', 'World', 'Telegram', 'Bot']
keyboard = ReplyKeyboardMarkup(
    [[label] for label in button_labels],  # one button per row
    resize_keyboard=True
)

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Choose one of the options below:",
        reply_markup=keyboard
    )

# General message handler
async def echo_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    if user_input in button_labels:
        await update.message.reply_text(user_input.lower())
    else:
        await update.message.reply_text("Please use the buttons 😊")

# Main function
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_selection))
    print("Bot is running... Press Ctrl+C to stop.")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


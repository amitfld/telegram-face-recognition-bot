import io
from PIL import Image
import os
import pickle
import cv2
import numpy as np
import face_recognition
import nest_asyncio
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler

nest_asyncio.apply()
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATA_FILE = "known_faces.pkl"

# States for conversation
WAITING_IMAGE, WAITING_NAME, WAITING_RECOGNITION_IMAGE = range(3)

# In-memory database (loaded from file at startup)
known_face_encodings = []
known_face_names = []

def save_known_faces():
    with open(DATA_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def load_known_faces():
    global known_face_encodings, known_face_names
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                known_face_encodings, known_face_names = pickle.load(f)
        except EOFError:
            print("Warning: known_faces.pkl is empty or corrupted. Starting fresh.")
            known_face_encodings, known_face_names = [], []

# Initial loading
load_known_faces()

main_keyboard = ReplyKeyboardMarkup([
    ["Add face"],
    ["Recognize faces"],
    ["Reset faces"]
], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=main_keyboard)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Clear any stored user data and reset conversation state
    context.user_data.clear()

    text = update.message.text

    if text == "Add face":
        await update.message.reply_text("Upload an image with a single face", reply_markup=main_keyboard)
        return WAITING_IMAGE

    elif text == "Recognize faces":
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in it", reply_markup=main_keyboard)
        return WAITING_RECOGNITION_IMAGE

    elif text == "Reset faces":
        global known_face_encodings, known_face_names
        known_face_encodings = []
        known_face_names = []
        save_known_faces()
        await update.message.reply_text("All faces have been forgotten.", reply_markup=main_keyboard)
        return ConversationHandler.END

    else:
        await update.message.reply_text("Please choose one of the options from the keyboard.", reply_markup=main_keyboard)
        return ConversationHandler.END

async def handle_add_face_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    image_path = "temp_add.jpg"
    await photo.download_to_drive(image_path)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        await update.message.reply_text("Please send an image with exactly one face.", reply_markup=main_keyboard)
        return ConversationHandler.END

    context.user_data["new_face_encoding"] = encodings[0]
    os.remove(image_path)
    await update.message.reply_text("Great. What’s the name of the person in this image?")
    return WAITING_NAME

async def handle_add_face_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.message.text
    known_face_encodings.append(context.user_data["new_face_encoding"])
    known_face_names.append(name)
    save_known_faces()
    await update.message.reply_text("Great. I will now remember this face.", reply_markup=main_keyboard)
    return ConversationHandler.END

async def handle_recognition_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    image_path = "temp_recognize.jpg"
    await photo.download_to_drive(image_path)

    image = cv2.imread(image_path)
    if image is None:
        await update.message.reply_text("❗ I failed to load the image file. It might be in an unsupported format or corrupted.", reply_markup=main_keyboard)
        os.remove(image_path)
        return ConversationHandler.END
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        await update.message.reply_text("I couldn’t detect any faces in the image.", reply_markup=main_keyboard)
        os.remove(image_path)
        return ConversationHandler.END

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_names = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0 and min(face_distances) < 0.5:
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]

        recognized_names.append(name)
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save and send annotated image
    result_path = "recognized_result.jpg"
    # Resize if image is too wide (e.g., >1280px)
    if image.shape[1] > 1280:
        scale_ratio = 1280 / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)

    # Convert image (BGR → RGB) and encode it in memory using Pillow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    bio = io.BytesIO()
    bio.name = 'recognized.jpg'
    pil_image.save(bio, 'JPEG')
    bio.seek(0)

    try:
        await update.message.reply_photo(photo=bio)
    except Exception as e:
        await update.message.reply_text(f"Error sending image: {str(e)}", reply_markup=main_keyboard)
        os.remove(image_path)
        return ConversationHandler.END

    if all(name == "Unknown" for name in recognized_names):
        await update.message.reply_text("I don’t recognize anyone in this image.", reply_markup=main_keyboard)
    else:
        await update.message.reply_text(
            f"I found {len(face_locations)} face(s). The people are: {', '.join(recognized_names)}",
            reply_markup=main_keyboard
        )

    os.remove(image_path)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.", reply_markup=main_keyboard)
    return ConversationHandler.END

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)],
        states={
            WAITING_IMAGE: [
                MessageHandler(filters.PHOTO, handle_add_face_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),  # allow button press override
            ],
            WAITING_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_face_name),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),  # override with button
            ],
            WAITING_RECOGNITION_IMAGE: [
                MessageHandler(filters.PHOTO, handle_recognition_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),  # override with button
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    print("Bot is running...")
    app.run_polling()

import io
from PIL import Image, ImageDraw, ImageFont
import os
import pickle
import cv2
import numpy as np
import face_recognition
import nest_asyncio
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler
import arabic_reshaper
from bidi.algorithm import get_display

nest_asyncio.apply()
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATA_FILE = "known_faces.pkl"

# States for conversation
WAITING_IMAGE, WAITING_NAME, WAITING_RECOGNITION_IMAGE,WAITING_CELEB_LOOKUP_IMAGE = range(4)

# In-memory database (loaded from file at startup)
# Known Faces - Structure: {name: [face_encoding1, face_encoding2, ...]}
known_faces = {}
# Celebs Encodings - Structure: {name: [face_encoding1, face_encoding2, ...]}
celeb_encodings = {}
# Celebs images - Structure: {name: [path_to_image_1, path_to_image_2, ...]}
celeb_images = {}

def save_known_faces():
    with open(DATA_FILE, "wb") as f:
        pickle.dump(known_faces, f)

def load_known_faces():
    global known_faces
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                known_faces = pickle.load(f)
        except EOFError:
            print("Warning: known_faces.pkl is empty or corrupted. Starting fresh.")
            known_faces = {}

def load_celeb_encodings(celebs_folder="celebs"):
    global celeb_encodings, celeb_images

    if not os.path.exists(celebs_folder):
        print(f"⚠️ Celebs folder '{celebs_folder}' not found.")
        return

    for celeb_name in os.listdir(celebs_folder):
        celeb_path = os.path.join(celebs_folder, celeb_name)
        if not os.path.isdir(celeb_path):
            continue

        celeb_encodings[celeb_name] = []
        celeb_images[celeb_name] = []

        for img_file in os.listdir(celeb_path):
            img_path = os.path.join(celeb_path, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    celeb_encodings[celeb_name].append(encodings[0])
                    celeb_images[celeb_name].append(img_path)
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

# Initial loading
load_known_faces()
load_celeb_encodings()

main_keyboard = ReplyKeyboardMarkup([
    ["Add face"],
    ["Recognize faces"],
    ["Reset faces"],
    ["Similar celebs"]
], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", 
                                    reply_markup=main_keyboard)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Clear any stored user data and reset conversation state
    context.user_data.clear()

    text = update.message.text

    if text == "Add face":
        await update.message.reply_text("Upload an image with a single face", 
                                        reply_markup=main_keyboard)
        return WAITING_IMAGE

    elif text == "Recognize faces":
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in it", 
                                        reply_markup=main_keyboard)
        return WAITING_RECOGNITION_IMAGE

    elif text == "Reset faces":
        global known_faces
        known_faces = {}
        save_known_faces()
        await update.message.reply_text("All faces have been forgotten.", 
                                        reply_markup=main_keyboard)
        return ConversationHandler.END
    
    elif text == "Similar celebs":
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person.", 
                                        reply_markup=main_keyboard)
        return WAITING_CELEB_LOOKUP_IMAGE

    else:
        await update.message.reply_text("Please choose one of the options from the keyboard.", 
                                        reply_markup=main_keyboard)
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
    global known_faces
    name = update.message.text
    encoding = context.user_data["new_face_encoding"]

    if name in known_faces:
        known_faces[name].append(encoding)
    else:
        known_faces[name] = [encoding]

    save_known_faces()
    await update.message.reply_text("Great. I will now remember this face.", reply_markup=main_keyboard)
    return ConversationHandler.END

async def handle_recognition_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global known_faces
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
        best_match_name = "Unknown"
        best_distance = float("inf")

        # Compare this encoding with all known encodings
        for name, encodings in known_faces.items():
            distances = face_recognition.face_distance(encodings, face_encoding)
            if len(distances) == 0:
                continue
            min_distance = np.min(distances)
            if min_distance < 0.5 and min_distance < best_distance:
                best_distance = min_distance
                best_match_name = name

        recognized_names.append(best_match_name)

        # Draw bounding box using OpenCV
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    # Resize image if needed
    if image.shape[1] > 1280:
        scale_ratio = 1280 / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)

    # Convert to RGB and use PIL for Unicode-safe text rendering
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 32)  # Use a font that supports Hebrew
    except:
        font = ImageFont.load_default()

    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # Reshape and reverse name if it's in Hebrew (RTL)
        reshaped_text = arabic_reshaper.reshape(name)
        bidi_text = get_display(reshaped_text)

        # Draw in blue with larger font
        draw.text((left, top - 40), bidi_text, font=font, fill=(0, 0, 255))

    # Save and send the annotated image
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

async def handle_celeb_lookup_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    image_path = "temp_celeb.jpg"
    await photo.download_to_drive(image_path)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        await update.message.reply_text("Please upload an image with exactly one clear face.", reply_markup=main_keyboard)
        os.remove(image_path)
        return ConversationHandler.END

    input_encoding = encodings[0]

    best_match_name = "Unknown"
    best_distance = float("inf")

    for name, encodings_list in celeb_encodings.items():
        if not encodings_list:
            continue
        distances = face_recognition.face_distance(encodings_list, input_encoding)
        min_distance = np.min(distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_match_name = name

    os.remove(image_path)

    # Get one of the celeb's images to send back
    celeb_image_path = celeb_images[best_match_name][0]  # you can randomize if you want
    with open(celeb_image_path, 'rb') as f:
        await update.message.reply_photo(photo=f)

    # Build a feedback message based on resemblance quality
    if best_distance < 0.54:
        feedback = "✅ Wow! You really resemble"
    elif best_distance < 0.60:
        feedback = "🟡 You kind of look like"
    elif best_distance < 0.68:
        feedback = "⚠️ With some imagination, you resemble"
    elif best_distance < 0.82:
        feedback = "❓ It's a bit of a stretch, but you remind me of"
    else:
        feedback = "🤷‍♂️ Honestly, I don't see it — but here’s who you matched"
    
    similarity_percent = distance_to_similarity_percent(best_distance)
    await update.message.reply_text(
        f"{feedback} {best_match_name}\n(similarity: {similarity_percent}%)",
        reply_markup=main_keyboard
    )

    return ConversationHandler.END

def distance_to_similarity_percent(distance, min_threshold=0.35, max_threshold=0.8):
    similarity = max(0.0, 1 - (distance - min_threshold) / (max_threshold - min_threshold))
    return round(similarity * 100)

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
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
            ],
            WAITING_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_face_name),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
            ],
            WAITING_RECOGNITION_IMAGE: [
                MessageHandler(filters.PHOTO, handle_recognition_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
            ],
            WAITING_CELEB_LOOKUP_IMAGE: [
                MessageHandler(filters.PHOTO, handle_celeb_lookup_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    print("Bot is running...")
    app.run_polling()

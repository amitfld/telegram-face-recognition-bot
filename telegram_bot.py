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
import uuid
import shutil
import stat
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from collections import defaultdict

nest_asyncio.apply()
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATA_FILE = "known_faces.pkl"

# States for conversation
WAITING_IMAGE, WAITING_NAME, WAITING_RECOGNITION_IMAGE, WAITING_CELEB_LOOKUP_IMAGE, WAITING_FIRST_IMAGE, WAITING_SECOND_IMAGE = range(6)

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
    ["Similar celebs"],
    ["Map"],
    ["Similarity check"]
], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Send the risograph poster image
    try:
        with open("Bot-Risograph.png", "rb") as poster:
            await update.message.reply_photo(
                photo=poster,
                caption="Welcome to ScanYourFaceBot!\nHere's what I can do 🧠👆"
            )
    except Exception as e:
        await update.message.reply_text("Welcome to ScanYourFaceBot!")
    # Then send the keyboard
    await update.message.reply_text("Choose an option:", reply_markup=main_keyboard)

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

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
        # Delete saved face images folder
        user_faces_dir = "user_faces"
        if os.path.exists(user_faces_dir):
            shutil.rmtree(user_faces_dir, onerror=remove_readonly)

        save_known_faces()
        await update.message.reply_text("All faces have been forgotten.", 
                                        reply_markup=main_keyboard)
        return ConversationHandler.END
    
    elif text == "Similar celebs":
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person.", 
                                        reply_markup=main_keyboard)
        return WAITING_CELEB_LOOKUP_IMAGE
    
    elif text == "Map":
        await update.message.reply_text("Generating the face similarity map... please wait ⏳")
        await generate_and_send_map(update)
        return ConversationHandler.END
    
    elif text == "Similarity check":
        await update.message.reply_text("Please send the first image with exactly one clear face.")
        return WAITING_FIRST_IMAGE

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
    face_locations = face_recognition.face_locations(image)

    if len(encodings) != 1:
        await update.message.reply_text("Please send an image with exactly one face.", reply_markup=main_keyboard)
        os.remove(image_path)
        return ConversationHandler.END

    encoding = encodings[0]
    top, right, bottom, left = face_locations[0]
    face_crop = image[top:bottom, left:right]
    pil_face = Image.fromarray(face_crop)

    # Save cropped image
    face_dir = "user_faces"
    os.makedirs(face_dir, exist_ok=True)
    filename = os.path.join(face_dir, f"{uuid.uuid4()}.jpg")
    pil_face.save(filename)

    # Save to context
    context.user_data["new_face_encoding"] = encoding
    context.user_data["new_face_image_path"] = filename

    os.remove(image_path)
    await update.message.reply_text("Great. What’s the name of the person in this image?")
    return WAITING_NAME

async def handle_add_face_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global known_faces
    name = update.message.text.strip()
    encoding = context.user_data["new_face_encoding"]
    original_path = context.user_data["new_face_image_path"]

    # Ensure directory exists for this person's folder
    person_dir = os.path.join("user_faces", name)
    os.makedirs(person_dir, exist_ok=True)

    # Move the image into the person's folder with a cleaner name
    new_filename = f"{uuid.uuid4()}.jpg"
    new_path = os.path.join(person_dir, new_filename)
    os.rename(original_path, new_path)

    # Store (encoding, image path)
    if name not in known_faces:
        known_faces[name] = []
    known_faces[name].append((encoding, new_path))

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

            only_encodings = [enc for (enc, _) in encodings]
            distances = face_recognition.face_distance(only_encodings, face_encoding)
            if len(distances) == 0:
                continue
            min_distance = np.min(distances)
            if min_distance < 0.45 and min_distance < best_distance:
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
        f"{feedback} {best_match_name}.\n(similarity: {similarity_percent}%)",
        reply_markup=main_keyboard
    )

    return ConversationHandler.END

async def handle_first_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    image_path = "temp_first.jpg"
    await photo.download_to_drive(image_path)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        await update.message.reply_text("❗ Please upload an image with exactly one clear face.")
        os.remove(image_path)
        return ConversationHandler.END

    context.user_data["first_encoding"] = encodings[0]
    os.remove(image_path)

    await update.message.reply_text("✅ Got it. Now please send the second image.")
    return WAITING_SECOND_IMAGE

async def handle_second_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    image_path = "temp_second.jpg"
    await photo.download_to_drive(image_path)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        await update.message.reply_text("❗ Please upload an image with exactly one clear face.")
        os.remove(image_path)
        return ConversationHandler.END

    second_encoding = encodings[0]
    first_encoding = context.user_data.get("first_encoding")
    os.remove(image_path)

    if first_encoding is None:
        await update.message.reply_text("❌ Something went wrong. Please start again.", reply_markup=main_keyboard)
        return ConversationHandler.END

    distance = face_recognition.face_distance([first_encoding], second_encoding)[0]
    similarity = distance_to_similarity_percent(distance)

    # Feedback message (you can customize this scale)
    if similarity >= 72:
        feedback = "✅ These faces look very similar!"
    elif similarity >= 65:
        feedback = "🟡 There's a decent resemblance."
    elif similarity >= 50:
        feedback = "⚠️ Some similarity, but not a strong match."
    else:
        feedback = "❌ These faces don’t look very similar."

    await update.message.reply_text(
        f"{feedback}\nSimilarity: {similarity}%",
        reply_markup=main_keyboard
    )

    context.user_data.clear()
    return ConversationHandler.END


def distance_to_similarity_percent(distance, min_threshold=0.35, max_threshold=0.8):
    raw = 1 - (distance - min_threshold) / (max_threshold - min_threshold)
    clamped = min(1.0, max(0.0, raw))  # Clamp between 0 and 1
    return round(clamped * 100)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.", reply_markup=main_keyboard)
    return ConversationHandler.END

async def generate_and_send_map(update: Update):
    if not known_faces and not celeb_encodings:
        await update.message.reply_text("I don’t have any faces to map yet.")
        return

    # Combine all encodings, labels, image paths
    all_encodings = []
    all_labels = []
    all_image_paths = []

    for name, enc_list in known_faces.items():
        for item in enc_list:
            if isinstance(item, tuple):
                encoding, img_path = item
            else:
                encoding = item
                img_path = None
            all_encodings.append(encoding)
            all_labels.append(name)
            all_image_paths.append(img_path)

    for name, enc_list in celeb_encodings.items():
        for i, encoding in enumerate(enc_list):
            all_encodings.append(encoding)
            all_labels.append(name)
            all_image_paths.append(celeb_images[name][i])

    # Reduce dimensions
    X = np.array(all_encodings)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=200).fit_transform(X) * 10

    # Group indices
    name_to_indices = defaultdict(list)
    for i, name in enumerate(all_labels):
        name_to_indices[name].append(i)

    # Spread similar names
    spread = 12.0
    for indices in name_to_indices.values():
        for j, idx in enumerate(indices):
            angle = (2 * np.pi * j) / len(indices)
            dx = np.cos(angle) * spread
            dy = np.sin(angle) * spread
            X_2d[idx][0] += dx
            X_2d[idx][1] += dy

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Face Similarity Map (similar faces are positioned closer)")

    def get_thumbnail(img_path, zoom=0.35, size=(60, 60)):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        bordered = Image.new("RGB", (size[0] + 4, size[1] + 4), "white")
        bordered.paste(img, (2, 2))
        return OffsetImage(np.array(bordered), zoom=zoom)

    # Draw images
    for i, (x, y) in enumerate(X_2d):
        img_path = all_image_paths[i]
        if img_path and os.path.exists(img_path):
            try:
                imagebox = get_thumbnail(img_path)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")

    # Draw name labels above each cluster
    for name, indices in name_to_indices.items():
        highest_idx = min(indices, key=lambda i: X_2d[i][1])
        x, y = X_2d[highest_idx]
        ax.text(x, y + 35, name, fontsize=4, ha="center",
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    ax.set_xlim(X_2d[:, 0].min() - 50, X_2d[:, 0].max() + 50)
    ax.set_ylim(X_2d[:, 1].min() - 50, X_2d[:, 1].max() + 50)
    plt.axis('off')

    # Save and send
    output_path = "similarity_map.jpg"
    plt.savefig(output_path, dpi=150)
    plt.close()

    with open(output_path, "rb") as f:
        await update.message.reply_photo(photo=f)


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
            WAITING_FIRST_IMAGE: [
                MessageHandler(filters.PHOTO, handle_first_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
            ],
            WAITING_SECOND_IMAGE: [
                MessageHandler(filters.PHOTO, handle_second_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    print("Bot is running...")
    app.run_polling()

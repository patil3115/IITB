import csv
import cv2
import os

# Directory containing video files
main_dir = r'C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\DataSet\Test'  # Main directory containing subfolders

# Directory to save frames
output_dir = 'frames_final'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Emotion-specific directories
emotions = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
for emotion in emotions:
    emotion_dir = os.path.join(output_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

# Function to extract maximum occurrence emotion from the CSV file
def get_max_occurrence_emotion(csv_file):
    max_occurrence_emotion = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        # Print column names for debugging
        header = reader.fieldnames
        print(f"CSV Header: {header}")
        for row in reader:
            clip_id = row['ClipID']
            emotions_count = {}
            for emotion in emotions:
                if emotion in row:
                    emotions_count[emotion] = int(row[emotion])
                else:
                    print(f"Emotion '{emotion}' not found in row: {row}")
            if emotions_count:
                max_value = max(emotions_count.values())
                max_emotions = [emotion for emotion, count in emotions_count.items() if count == max_value]
                if len(max_emotions) > 1:  # If there's a tie, use "Frustration"
                    max_emotion = 'Frustration'
                else:
                    max_emotion = max_emotions[0]
                max_occurrence_emotion[clip_id] = max_emotion
                print(f"Clip ID: {clip_id}, Emotions Count: {emotions_count}, Max Emotion: {max_emotion}")
    return max_occurrence_emotion

# Extract maximum occurrence emotion from CSV file
csv_file = r'C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\Labels\TestLabels.csv'  # Adjust path to your actual CSV file
max_occurrence_emotion = get_max_occurrence_emotion(csv_file)

def extract_and_save_frames(clip_id, video_path, max_occurrence_emotion, frame_interval=30):
    print(f"Processing video: {video_path}, Max occurrence emotion: {max_occurrence_emotion.get(clip_id)}")
    
    # Determine the emotion with maximum occurrence for this video
    max_emotion = max_occurrence_emotion.get(clip_id)
    if not max_emotion:
        print(f"No emotion found for video: {clip_id}")
        return
    
    # Create the directory for the max emotion if it doesn't exist
    emotion_dir = os.path.join(output_dir, max_emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print(f"Failed to read the video file: {video_path}")
        return

    frame_number = 0
    saved_frames = 0

    while success:
        if frame_number % frame_interval == 0:
            # Save frame as image
            frame_name = os.path.join(emotion_dir, f"{os.path.splitext(clip_id)[0]}_frame{frame_number}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frames += 1
            
        # Move to the next frame
        frame_number += 1
        success, frame = cap.read()

    cap.release()
    print(f"Finished processing video: {video_path}, Saved frames: {saved_frames}")

# Function to recursively get all video files in the directory
def get_all_video_files(dir_path):
    video_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv')):  # Add other video formats if needed
                video_files.append(os.path.join(root, file))
    return video_files

# Get all video files in the main directory
video_files = get_all_video_files(main_dir)
print(f"Found {len(video_files)} video files.")

# Limit the number of videos processed (optional)
max_videos = 1866  # Set this to None to process all videos, or set a limit
if max_videos:
    video_files = video_files[:max_videos]

# Process each video file found and save frames in the directory of max occurrence emotion
for video_file in video_files:
    clip_id = os.path.basename(video_file)
    extract_and_save_frames(clip_id, video_file, max_occurrence_emotion)

print("Frames extracted and saved successfully.")


import csv
import cv2
import os

# Directory containing video files
main_dir = r'C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\DataSet\Train'  # Main directory containing subfolders

# Directory to save frames
output_dir = 'frames_final1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Emotion-specific directories
emotions = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
for emotion in emotions:
    emotion_dir = os.path.join(output_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

# Function to extract maximum occurrence emotion from the CSV file
def get_max_occurrence_emotion(csv_file):
    max_occurrence_emotion = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        # Print column names for debugging
        header = reader.fieldnames
        print(f"CSV Header: {header}")
        for row in reader:
            clip_id = row['ClipID']
            emotions_count = {}
            for emotion in emotions:
                if emotion in row:
                    emotions_count[emotion] = int(row[emotion])
                else:
                    print(f"Emotion '{emotion}' not found in row: {row}")
            if emotions_count:
                max_value = max(emotions_count.values())
                max_emotions = [emotion for emotion, count in emotions_count.items() if count == max_value]
                if len(max_emotions) > 1:  # If there's a tie, use "Frustration"
                    max_emotion = 'Frustration'
                else:
                    max_emotion = max_emotions[0]
                max_occurrence_emotion[clip_id] = max_emotion
                print(f"Clip ID: {clip_id}, Emotions Count: {emotions_count}, Max Emotion: {max_emotion}")
    return max_occurrence_emotion

# Extract maximum occurrence emotion from CSV file
csv_file = r"C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\Labels\TrainLabels.csv"  # Adjust path to your actual CSV file
max_occurrence_emotion = get_max_occurrence_emotion(csv_file)

def extract_and_save_frames(clip_id, video_path, max_occurrence_emotion, frame_interval=30):
    print(f"Processing video: {video_path}, Max occurrence emotion: {max_occurrence_emotion.get(clip_id)}")
    
    # Determine the emotion with maximum occurrence for this video
    max_emotion = max_occurrence_emotion.get(clip_id)
    if not max_emotion:
        print(f"No emotion found for video: {clip_id}")
        return
    
    # Create the directory for the max emotion if it doesn't exist
    emotion_dir = os.path.join(output_dir, max_emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print(f"Failed to read the video file: {video_path}")
        return

    frame_number = 0
    saved_frames = 0

    while success:
        if frame_number % frame_interval == 0:
            # Save frame as image
            frame_name = os.path.join(emotion_dir, f"{os.path.splitext(clip_id)[0]}_frame{frame_number}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frames += 1
            
        # Move to the next frame
        frame_number += 1
        success, frame = cap.read()

    cap.release()
    print(f"Finished processing video: {video_path}, Saved frames: {saved_frames}")

# Function to recursively get all video files in the directory
def get_all_video_files(dir_path):
    video_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv')):  # Add other video formats if needed
                video_files.append(os.path.join(root, file))
    return video_files

# Get all video files in the main directory
video_files = get_all_video_files(main_dir)
print(f"Found {len(video_files)} video files.")

# Limit the number of videos processed (optional)
max_videos = 1866  # Set this to None to process all videos, or set a limit
if max_videos:
    video_files = video_files[:max_videos]

# Process each video file found and save frames in the directory of max occurrence emotion
for video_file in video_files:
    clip_id = os.path.basename(video_file)
    extract_and_save_frames(clip_id, video_file, max_occurrence_emotion)

print("Frames extracted and saved successfully.")



import csv
import cv2
import os

# Directory containing video files
main_dir = r'C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\DataSet\Validation'  # Main directory containing subfolders

# Directory to save frames
output_dir = 'frames_final2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Emotion-specific directories
emotions = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
for emotion in emotions:
    emotion_dir = os.path.join(output_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

# Function to extract maximum occurrence emotion from the CSV file
def get_max_occurrence_emotion(csv_file):
    max_occurrence_emotion = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        # Print column names for debugging
        header = reader.fieldnames
        print(f"CSV Header: {header}")
        for row in reader:
            clip_id = row['ClipID']
            emotions_count = {}
            for emotion in emotions:
                if emotion in row:
                    emotions_count[emotion] = int(row[emotion])
                else:
                    print(f"Emotion '{emotion}' not found in row: {row}")
            if emotions_count:
                max_value = max(emotions_count.values())
                max_emotions = [emotion for emotion, count in emotions_count.items() if count == max_value]
                if len(max_emotions) > 1:  # If there's a tie, use "Frustration"
                    max_emotion = 'Frustration'
                else:
                    max_emotion = max_emotions[0]
                max_occurrence_emotion[clip_id] = max_emotion
                print(f"Clip ID: {clip_id}, Emotions Count: {emotions_count}, Max Emotion: {max_emotion}")
    return max_occurrence_emotion

# Extract maximum occurrence emotion from CSV file
csv_file = r"C:\Users\Ashutosh Patil\Desktop\IITB\DAiSEE\Labels\ValidationLabels.csv"  # Adjust path to your actual CSV file
max_occurrence_emotion = get_max_occurrence_emotion(csv_file)

def extract_and_save_frames(clip_id, video_path, max_occurrence_emotion, frame_interval=30):
    print(f"Processing video: {video_path}, Max occurrence emotion: {max_occurrence_emotion.get(clip_id)}")
    
    # Determine the emotion with maximum occurrence for this video
    max_emotion = max_occurrence_emotion.get(clip_id)
    if not max_emotion:
        print(f"No emotion found for video: {clip_id}")
        return
    
    # Create the directory for the max emotion if it doesn't exist
    emotion_dir = os.path.join(output_dir, max_emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print(f"Failed to read the video file: {video_path}")
        return

    frame_number = 0
    saved_frames = 0

    while success:
        if frame_number % frame_interval == 0:
            # Save frame as image
            frame_name = os.path.join(emotion_dir, f"{os.path.splitext(clip_id)[0]}_frame{frame_number}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frames += 1
            
        # Move to the next frame
        frame_number += 1
        success, frame = cap.read()

    cap.release()
    print(f"Finished processing video: {video_path}, Saved frames: {saved_frames}")

# Function to recursively get all video files in the directory
def get_all_video_files(dir_path):
    video_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv')):  # Add other video formats if needed
                video_files.append(os.path.join(root, file))
    return video_files

# Get all video files in the main directory
video_files = get_all_video_files(main_dir)
print(f"Found {len(video_files)} video files.")

# Limit the number of videos processed (optional)
max_videos = 1866  # Set this to None to process all videos, or set a limit
if max_videos:
    video_files = video_files[:max_videos]

# Process each video file found and save frames in the directory of max occurrence emotion
for video_file in video_files:
    clip_id = os.path.basename(video_file)
    extract_and_save_frames(clip_id, video_file, max_occurrence_emotion)

print("Frames extracted and saved successfully.")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Directory containing frames for each dataset
train_dir = r"C:\Users\Ashutosh Patil\Desktop\IITB\frames_final1"
val_dir = r"C:\Users\Ashutosh Patil\Desktop\IITB\frames_final2"
test_dir = r"C:\Users\Ashutosh Patil\Desktop\IITB\frames_final"

# Parameters
img_height, img_width = 128, 128  # Image dimensions
batch_size = 32
epochs = 25  # Number of epochs
steps_per_epoch = 150  # Fixed number of steps per epoch
validation_steps = 150  # Fixed number of validation steps

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load test data
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ensure the order is maintained for evaluation
)

# Determine the number of classes
num_classes = train_generator.num_classes
print(f'Number of classes: {num_classes}')

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(150, activation='relu'),  # Updated dense layer to 150 units
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Fixed number of steps per epoch
    validation_data=val_generator,
    validation_steps=validation_steps,  # Fixed number of validation steps
    epochs=epochs  # Number of epochs
)

# Save the model
model.save('emotion_classification_model.h5')

# Evaluate the model on test data
print("Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Get predictions on test data
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Print classification report and confusion matrix
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

print("Model trained, evaluated on test data, and saved successfully as 'emotion_classification_model.h5'.")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Define the image size (same as used during training)
img_height, img_width = 128, 128

# Load the trained model
model = tf.keras.models.load_model('emotion_classification_model.h5')

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array

# Path to a sample image from your dataset
image_path =r"C:\Users\Ashutosh Patil\Desktop\IITB\9877360256_frame0.jpg"  # Replace with the actual path to the image

# Load and display the image
img_array = load_and_preprocess_image(image_path)
img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
plt.imshow(img)
plt.show()

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Get class indices
class_names = ['Boredom', 'Confusion', 'Engagement', 'Frustration']

# Print the predicted class and probability for each class
print(f"Predicted Emotion: {class_names[predicted_class[0]]} ({predictions[0][predicted_class[0]] * 100:.2f}%)")
print("Probabilities for each emotion:")
for i, prob in enumerate(predictions[0]):
    print(f"{class_names[i]}: {prob * 100:.2f}%")

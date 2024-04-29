# %%
"""
# LSTM + CNN
"""

# %%
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
from datetime import datetime
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
def features_extract(file):
    audio, sam = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sam, n_mfcc=40)
    mfcc_scaled = np.mean(mfccs_features.T, axis=0)
    return mfcc_scaled

# Path to the directory containing audio files
path = "C:\\Users\\abhih\\OneDrive\\Desktop\\tdl_project\\TESS Toronto emotional speech set data"

# Extract features from audio files
extracted_features = []
all_files = os.listdir(path)
for sets in tqdm(all_files, desc="Processing Directories"):
    cur_path = os.path.join(path, sets)
    for file_name in tqdm(os.listdir(cur_path), desc="Processing Files in {}".format(sets), leave=False):
        data = features_extract(os.path.join(cur_path, file_name))
        extracted_features.append([sets, data])

# %%
# Create DataFrame for extracted features
voice = pd.DataFrame(extracted_features, columns=['Mood', 'Data'])

# Preprocess data
X = np.array(voice['Data'].tolist())
y = np.array(voice['Mood'].apply(lambda x: x.lower().split('_')[1]))

# %%
label = LabelEncoder()
y = to_categorical(label.fit_transform(y))

# Initialize cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %%
def create_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(40, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model


# %%
# Initialize lists to store accuracy for each fold
acc_per_fold = []

# Convert one-hot encoded labels back to integer labels
y_labels = np.argmax(y, axis=1)

# Perform cross-validation
for index, (train_index, test_index) in enumerate(skf.split(X, y_labels)):
    print(f"Training on Fold {index + 1}...")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define callbacks
    filepath = f"Audio_Neuron_fold{index + 1}.keras"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    # Train model
    history = model.fit(X_train, y_train, batch_size=32, epochs=100,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpointer, early_stopping])
    
    # Evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {index + 1} - Accuracy: {scores[1]*100}%")
    acc_per_fold.append(scores[1] * 100)

# Print average accuracy across folds
print(f"Average Accuracy: {np.mean(acc_per_fold)}%")


# %%
model.summary()

# %%
print(f"Average Accuracy: {np.mean(acc_per_fold)}%")

# %%
# Save the model
keras.saving.save_model(model, 'model_main_cnn_lstm.keras')
print("Model saved successfully.")

# %%
data = "C:\\Users\\abhih\\OneDrive\\Desktop\\tdl_project\\voices_testing\\anger.wav" 
ipd.Audio(data) 

# %%
# predictu = features_extract(data)
# predictu = predictu.reshape(1, -1)
# op = model.predict(predictu) 
# for i in op:
#     print(np.round(i)) 
final = []
audio, sam = librosa.load(data, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sam, n_mfcc=40)
mfcc_scaled = np.mean(mfccs_features.T, axis=0)
# print(mfcc_scaled)
# Reshape mfcc_scaled to match the input shape of the model
mfcc_scaled = mfcc_scaled.reshape(1, mfcc_scaled.shape[0], 1)

# Print the shape of mfcc_scaled before prediction
print("Shape of mfcc_scaled before prediction:", mfcc_scaled.shape)

# Predict with the model
predicted = model.predict(mfcc_scaled)

# Process the predicted probabilities
final = []
for i in predicted:
    final.append(np.round(i))

# Define the emotion labels
d = label.classes_
print(d)



# %%
print(final) 

# %%
output = []
for i in predicted:
    kp = list(i)
    for i in kp:
        output.append(i)

# %%
for i in output:
    print(round(i))  

# %%
j = 0 
for i in range(len(output)):
    if round(output[i]) == 1:
        print(d[i])
        break

# %%
# import pickle
# with open('Emotions.pkl', 'wb') as sounds:
#     pickle.dump(model, sounds) 

# %%
# import pickle
# with open('EmotionLabels.pkl', 'wb') as plane:
#     pickle.dump(label, plane) 

# %%
# model.save("Neuron/") 
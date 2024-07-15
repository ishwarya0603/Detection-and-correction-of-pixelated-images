# Detection-and-correction-of-pixelated-images
Our main objective is to create a detection and correction model for detecting and correcting pixelated images using SRCNN model.
The primary objective of using a Super-Resolution Convolutional Neural Network (SRCNN) model is to improve the resolution and quality of low-resolution images. SRCNN achieves this by learning a mapping from low-resolution images to their high-resolution counterparts. 

https://github.com/user-attachments/assets/788f8f45-4892-499c-bee9-ec750798e875

DETECTION:

The below code loads images and assigns them labels, making it suitable for detection tasks.

```
def load_images_and_labels(hr_dir, lr_dirs):
    images = []
    labels = []
    # Load HR images
    for filename in os.listdir(hr_dir):
        img = load_img(os.path.join(hr_dir, filename), target_size=(64, 64))
        images.append(img_to_array(img))
        labels.append(0)  # 0 for non-pixelated (HR)
    # Load LR images from multiple directories
    for lr_dir in lr_dirs:
        for filename in os.listdir(lr_dir):
            img = load_img(os.path.join(lr_dir, filename), target_size=(64, 64))
            images.append(img_to_array(img))
            labels.append(1)  # 1 for pixelated (LR)
    return np.array(images), np.array(labels)

images, labels = load_images_and_labels(hr_dir, lr_dirs)
```
Create the SRCNN Detection model:
```
def build_classification_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

classification_model = build_classification_model()
```

DETECTION:

Import necessary libraries:
```
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.kera![Screenshot 2024-07-05 205244]
s.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
```
We use tensorflow and keras for creating the neural network while the module matplotlib is used for plotting and evaluating the results.

Specify paths to your dataset:

Create a custom generator to preprocess and match the low resolution images with its corresponding high resolution image so that there is no sample size error between the high resolution images and 5 low resolution images for each high resolution images.
```
def custom_image_generator(lr_dirs, hr_dir, batch_size=16, target_size=(256, 256)):
    while True:
        lr_images = []
        hr_images = []
        for _ in range(batch_size):
            lr_img_path = np.random.choice([os.path.join(dir, f) for dir in lr_dirs for f in os.listdir(dir)])
            hr_img_path = os.path.join(hr_dir, os.path.basename(lr_img_path))
            lr_img = load_img(lr_img_path, target_size=target_size)
            lr_img = img_to_array(lr_img) / 255.0
            hr_img = load_img(hr_img_path, target_size=target_size)
            hr_img = img_to_array(hr_img) / 255.0
            lr_images.append(lr_img)
            hr_images.append(hr_img)
        yield np.array(lr_images), np.array(hr_images)
```

PSNR calculation function:
```
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)
```

Building SRCNN model:
```
def build_srcnn():
    input_shape = (None, None, 3)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    outputs = Conv2D(3, (5, 5), activation='linear', padding='same')(conv3)
    model = Model(inputs, outputs)
    return model
The SRCNN model consists of several convolutional layers with different kernel sizes and activation functions.
```

Compiling the model:
```
learning_rate = 0.0001
deeper_srcnn_model = build_deeper_srcnn()
deeper_srcnn_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[psnr])
Low learning rate as the memory of our PC is low for higher learning learning rate.
```

PROBLEMS:

The model had a tendency to overfit, thus regularisation techniques like EarlyStopping and LR Scheduling were used.
```
Scheduling learning rate:
def lr_schedule(epoch):
    initial_lr = 0.0001
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * np.exp(0.1 * (10 - epoch))
lr_scheduler = LearningRateScheduler(lr_schedule)
This function defines a learning rate scheduler that decreases the learning rate after the first 10 epochs.

Early stopping:
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

Train the model:
```
epochs = 50
steps_per_epoch = sum([len(os.listdir(dir)) for dir in LR_traindir]) // batch_size
validation_steps = sum([len(os.listdir(dir)) for dir in LR_validdir]) // batch_size

model = deeper_srcnn_model.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)
```
Evaluate the model:
```
test_lr_images, test_hr_images = next(valid_generator)
test_loss, test_psnr = deeper_srcnn_model.evaluate(test_lr_images, test_hr_images)
print("Test Loss:", test_loss)
print("Test PSNR:", test_psnr)
```
Save the trained model:
```
srcnn_model.save(model_save_path)
```

Plotting validation and training loss:
```
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Valid Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''Plotting validation and training PSNR
plt.figure(figsize=(10, 5))
plt.plot(history.history['psnr'], label='Train PSNR')
plt.plot(history.history['val_psnr'], label='Valid PSNR')
plt.title('Training and Validation PSNR')
plt.xlabel('Epochs')
plt.ylabel('PSNR')
plt.legend()
plt.show()
```

![Screenshot 2024-07-08 193917](https://github.com/user-attachments/assets/9ed6e114-05c6-4666-ab0f-24b33c4df0ed)
![Screenshot 2024-07-08 193903](https://github.com/user-attachments/assets/c8af4c1c-0a3b-40b2-a0d9-f456f5f4b522)





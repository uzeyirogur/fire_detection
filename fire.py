from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Veri dizinleri
training_dir = "fire_dataset/Train_Data"
validation_dir = "fire_dataset/Test_Data"

# Veri artırma
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=45,
    fill_mode="nearest"
)

validation_data_generator = ImageDataGenerator(rescale=1./255)

# Eğitim ve doğrulama jeneratörleri
train_generetor = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=128  # Batch size 32
)

validation_generetor = validation_data_generator.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=128  # Batch size 32
)

input_shape = (224, 224, 3)

def firenet(input_shape):
    # Modeli oluşturma
    model = Sequential()

    # 1. Konvolüsyon katmanı
    model.add(Conv2D(96, (11, 11), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D((3, 3), strides=(2, 2)))

    # 2. Konvolüsyon katmanı
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPool2D((3, 3), strides=(2, 2)))

    # 3. Konvolüsyon katmanı
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(MaxPool2D((3, 3), strides=(2, 2)))

    # Düzleştirme katmanı
    model.add(Flatten())
    model.add(Dropout(0.3))

    # Tam bağlı katman
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))

    # Çıkış katmanı (örneğin, 2 sınıf için)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=["accuracy"])  # 'acc' yerine 'accuracy' kullanıldı
    return model

# Modeli oluşturma
model = firenet(input_shape)

# Modeli eğitme
history = model.fit(
    train_generetor,
    steps_per_epoch=len(train_generetor),
    epochs=1,  # Epoch sayısını 5'e düşürdük
    validation_data=validation_generetor,
    validation_steps=len(validation_generetor)
)

# Eğitim ve doğrulama doğruluğu ve kaybı
acc = history.history["accuracy"]  # 'acc' yerine 'accuracy'
val_acc = history.history["val_accuracy"]  # 'val_acc' yerine 'val_accuracy'
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Epoch sayısını doğruluğun uzunluğuna göre ayarlayın
epochs = range(len(acc))

# Eğitim ve doğrulama doğruluğunu grafiğe dökelim
plt.plot(epochs, acc, label="train_acc")
plt.plot(epochs, val_acc, label="val_acc")
plt.title("Training and Validation Accuracy")
plt.legend(loc=0)
plt.figure()

# Eğitim ve doğrulama kaybını grafiğe dökelim
plt.plot(epochs, loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.title("Training and Validation Loss")
plt.legend(loc=0)
plt.figure()

plt.show()

# Modeli kaydetme
model.save("model_fire.h5")

import cv2
import numpy as np
from keras.models import load_model

path = "fire_dataset/NF_17.jpg"

test_img = cv2.imread(path)
img=np.asarray(test_img)
img = cv2.resize(img,(224,224))
img = img/255

img = img.reshape(1,224,224,3)

prediction = model.predict(img)
pred = max(prediction[0])


pred = np.argmax(prediction[0])
prob = prediction[0][pred]
prob_y= "{:.2f}".format(prob)

if pred == 1:
    label = "yangin"
else :
    label = "normal"

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(test_img, label, (45,50), font, 1, (255,0,0),2)
cv2.putText(test_img, prob_y, (80,80), font, 1, (255,0,0),2)
cv2.imshow("sonuc", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

video_path = "fire_dataset/fire.mp4"
cap = cv2.VideoCapture(video_path)
while True:
    ret,frame = cap.read()
    
    img = np.asarray(frame)
    img = cv2.resize(img,(224,224))
    img = img/255
    img = img.reshape(1,224,224,3)
    pred = max(prediction[0])
    pred = np.argmax(prediction[0])
    prob = prediction[0][pred]
    prob_y= "{:.2f}".format(prob)
    if pred == 1:
        label = "yangin"
    else :
        label = "normal"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (45,50), font, 1, (255,0,0),2)
    cv2.putText(frame, prob_y, (80,80), font, 1, (255,0,0),2)
    cv2.imshow("sonuc", frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()    
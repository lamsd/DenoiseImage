from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from numpy import asarray, reshape
from sklearn.model_selection import train_test_split
from os.path import join
import src.codefun as codf
import cofig as coff

def process_image(path):
    img = imread(path)
    img = asarray(img, dtype="float32")
    img = resize(img, (640, 480))
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = img/255.0
    img = reshape(img, (480, 640, 1))
    return img

def prossing_data(pathtrain, pathtest, pathttrainclear):
    train = []
    train_cleaned = []
    test = []
    for f in codf.sortdata(pathtrain):
        train.append(process_image(join(pathtrain, f)))

    for f in codf.sortdata(pathttrainclear):
        train_cleaned.append(process_image(join(pathttrainclear, f)))

    for f in codf.sortdata(pathtest):
        test.append(process_image(join(pathtest, f)))

    X_train = asarray(train)
    Y_train = asarray(train_cleaned)
    X_test = asarray(test)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

    return X_train, X_val, Y_train, Y_val, X_test

def model():
    input_layer = Input(shape=(480, 640, 1))
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Dropout(0.5)(x)
    # decoding
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    
    output_layer = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def trainner(pathtrain, pathtest, pathttrainclear):
    model2 =model()
    model2.summary()
    X_train, X_val, Y_train, Y_val, X_test = prossing_data(pathtrain, pathtest, pathttrainclear)
    callback = EarlyStopping(monitor='loss', patience=30)
    history = model2.fit(X_train, Y_train,\
        validation_data=(X_val, Y_val), \
        epochs=coff.epochs, \
        batch_size=coff.batch_size, \
        verbose=coff.verbose, \
        callbacks=[callback])
    return history, model2
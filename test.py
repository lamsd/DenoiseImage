import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (480, 640, 1))
    return img

def main(model):
    # path_test = r"imagetest/as.png"
    path_test = r"imagetest/as4.jpg"
    data_sd = process_image(path_test)
    data_sd = np.asarray([data_sd])
    Y_test = model.predict(data_sd, batch_size=3)
    cv2.imwrite("data4.png",cv2.cvtColor(Y_test[0]*255, cv2.COLOR_GRAY2BGR))
    plt.figure(figsize=(15,25))
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data_sd[0][:,:,0], cmap='gray')
    plt.subplot(122)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Y_test[0][:,:,0], cmap='gray')
    plt.savefig("data3.png")
    plt.show()
    
if __name__ == "__main__":
    model = load_model('models/my_models1.h5')
    main(model)
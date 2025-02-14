from keras.models import load_model
import numpy as np
import cv2

model = load_model("Model.h5")

img = cv2.imread("") #image name
img = cv2.resize(img, (224, 224))
img = np.array(img)
img = img.astype("float32")/255.0
img = np.expand_dims(img, axis=0)

mapping = {0 : "angry", 1 : "fear", 2 : "happy", 3 : "sad", 4 : "surprise"}

prediction = model.predict(img)

index = list(prediction[0]).index(max(list(prediction[0])))

print(prediction)
print(mapping[index])
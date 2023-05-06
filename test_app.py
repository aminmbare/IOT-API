import requests 
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras 
import numpy as np
from PIL import Image
name_img = "leaf_detection.jpg"
image = keras.utils.load_img("/Users/aminembarek/Desktop/app/test/tomato.jpeg")
Serialized = pickle.dumps(image, protocol=pickle.DEFAULT_PROTOCOL)
files = {'image': (name_img, Serialized, 'multipart/form-data', {'Expires': '0'})}
resp = requests.post('http://127.0.0.2:8000/leaf_detection',files = files)
image= pickle.loads(resp.content)

plt.figure()
plt.imshow(image)
plt.show()
image = Image.fromarray(image)
image = image.resize((224,224))
name_img = "disease_detection.jpg"
image = keras.utils.load_img("/Users/aminembarek/Desktop/app/test/TomatoEarlyBlight3.JPG", target_size=(224, 224))
Serialized = pickle.dumps(image, protocol=pickle.DEFAULT_PROTOCOL)
files = {'image': (name_img, Serialized, 'multipart/form-data', {'Expires': '0'})}
resp = requests.post('http://127.0.0.2:8000/leaf_disease',files = files)
print(resp.json())
result = resp.json()








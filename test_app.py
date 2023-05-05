import requests 
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras 
import numpy as np
from PIL import Image
name_img = "leaf_detection.jpg"
#image = keras.utils.load_img("/Users/aminembarek/Desktop/app/test/tomato.jpeg", target_size=(448, 448))
#Serialized = pickle.dumps(image, protocol=pickle.DEFAULT_PROTOCOL)
#files = {'image': (name_img, Serialized, 'multipart/form-data', {'Expires': '0'})}
#resp = requests.post('https://leafdetection-kjlyzrfelq-og.a.run.app',files = files)
#image= pickle.loads(resp.content)
#
#plt.figure()
#plt.imshow(image)
#plt.show()
#image = Image.fromarray(image)
#image = image.resize((224,224))
name_img = "disease_detection.jpg"
image = keras.utils.load_img("/Users/aminembarek/Desktop/app/test/TomatoEarlyBlight3.JPG", target_size=(224, 224))
Serialized = pickle.dumps(image, protocol=pickle.DEFAULT_PROTOCOL)
files = {'image': (name_img, Serialized, 'multipart/form-data', {'Expires': '0'})}
resp = requests.post('https://ld-kjlyzrfelq-og.a.run.app',files = files)
print(resp.json())
result = resp.json()








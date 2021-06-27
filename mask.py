import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import base64
import tensorflow as tf


#face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'mask_recog.h5')
#model = load_model(r"C:\Users\Asus\Desktop\FMBProj\pyth\maskDetector\mask_recog.h5")
model = load_model(r"{}".format(filename))

#model = load_model("mask_recog.h5")
#model= tf.keras.models.load_model('mask_recog.h5') 

def face_mask_detector(frame):
  # frame = cv2.imread(fileName)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
  faces_list=[]
  preds=[]
  print(faces)
  for (x, y, w, h) in faces:
      face_frame = frame[y:y+h,x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))
      face_frame = img_to_array(face_frame)
      face_frame = np.expand_dims(face_frame, axis=0)
      face_frame =  preprocess_input(face_frame)
      faces_list.append(face_frame)
      print("first for loop")
      if len(faces_list)>0:
          preds = model.predict(faces_list)
      for pred in preds:
          (mask, withoutMask) = pred
      label = "Mask" if mask > withoutMask else "No Mask"
      print("label assign")
      color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
      print("label 2 assign")
      cv2.putText(frame, label, (x, y- 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
  # cv2_imshow(frame)
  #print("{}: {:.2f}%".format(label, max(mask, withoutMask) * 100))
  return label



#python facemask calling
from flask import Flask
from werkzeug.wrappers import request
from flask import request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, woekekekek!'

@app.route('/dus',methods=["POST"])
def hello_word():
    req=request.get_json()
    image_string=req["base"]
    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output = face_mask_detector(img_np)
    print(output)
    return output

app.run()









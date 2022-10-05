import os
from tkinter import E
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


model = tf.keras.models.load_model("C:/Users/yaman/Desktop/Drone_Notification/Spill_detection_model.h5")
img_path = "C:/Users/yaman/Desktop/Drone_Notification/clean_3.jpg"##new image path
test_image = image.load_img(img_path, target_size = [224,224])
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result < 0.2:
    print(f"no spill with {result} percentage")
    
    spill_img = cv2.imread(img_path)
    cv2.imshow("test" ,spill_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:

    print(f"spill with {result} percentage")
    from_address = 'yaman88887@gmail.com'
    to_address = 'yaman.aljnadi@gmail.com'
    message = MIMEMultipart('Foobar')
    content = MIMEText('Oil Spell detected at {24.8537014,46.7128969}', 'plain')
    message.attach(content)
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    mail.login(from_address, 'glqbpduleedkxdgj')
    mail.sendmail(from_address,to_address, message.as_string())
    mail.close()

    spill_img = cv2.imread(img_path)
    cv2.imshow("test" ,spill_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
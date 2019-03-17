from darkflow.net.build import TFNet
from cv2 import imread, imshow, waitKey, destroyAllWindows
import numpy as np
import cv2

options = {"model": "cfg/yolo.cfg",
           "load": "bin/yolov2.weights",
           "gpu": 0.7,
           "threshold": 0.01}

tfnet = TFNet(options)

def check_sensor(bot_x,top_x,x_line):
  status = ""
  loc = np.arange(top_x,bot_x)
  if x_line in loc:
    status = 1
  else:
    status = 0
  return status

def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    loc_x = int(newImage.shape[1])

    loc_y_top = int(0)

    loc_y_bot = int(newImage.shape[0])

    newImage = cv2.line(newImage, (int(loc_x/3), loc_y_top),
                     (int(loc_x/3), loc_y_bot), (0, 0, 255), 3)

    newImage = cv2.circle(
        newImage, (int(loc_x*(4/5)), int(loc_y_bot/6)), 30, (0, 0, 255), -1)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']
        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        if result["label"] == "car":
          if confidence > 0.01:
            if btm_x - top_x > 500:
                newImage = cv2.line(
                    newImage, (top_x, int((btm_y-top_y)/2)+top_y), (btm_x, int(((btm_y-top_y)/2)+top_y)), (255, 0, 0), 3)
                newImage = cv2.rectangle(
                    newImage, (top_x, top_y), (btm_x, btm_y), (0, 255, 0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y-5),
                                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 230, 0), 1, cv2.LINE_AA)
                status = check_sensor(btm_x, top_x, int(loc_x/3))
                print(status)
                if status == 1:
                  newImage = cv2.circle(
                      newImage, (int(loc_x*(4/5)), int(loc_y_bot/6)), 30, (0, 255, 0), -1)


    return newImage

cap = cv2.VideoCapture("2.MOV")

# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
  
    results = tfnet.return_predict(frame)
    newframe = boxing(frame, results)

    # Display the resulting frame
    cv2.imshow('Frame', newframe)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

waitKey(0)
destroyAllWindows()

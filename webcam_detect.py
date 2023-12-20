from ultralytics import YOLO
import cv2

# Load a model
# model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

model = YOLO("./best.pt")  # lataa oma malli


cap = cv2.VideoCapture(0)  # 0 is the index of the default camera)

while True:
    ret, frame = cap.read()  # read a frame from the camera
    if not ret:
        break  # exit if there's an error reading the frame

    results = model(frame, conf=0.5)  # pass the frame to the model

    # annotated_frame = results[0].plot(boxes=False)
    annotated_frame = results[0].plot(boxes=True)  # plot the results

    cv2.imshow("frame", annotated_frame)  # show the frame in a window
    if cv2.waitKey(1) == ord("q"):  # wait for a key press, exit if it's 'q'
        break

cap.release()  # release the camera
cv2.destroyAllWindows()  # close all windows

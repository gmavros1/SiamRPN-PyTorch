from siamRPNBIG import TrackerSiamRPNBIG
import cv2 as cv
import numpy as np

tracker = TrackerSiamRPNBIG("/home/gmavros/Desktop/sxolh_last/υπολογιστική όραση/tracker/siamRPN weights/SiamRPNOTB.pth")

#video = cv.VideoCapture("/home/gmavros/Desktop/sxolh_last/υπολογιστική όραση/tracker/videoTest/Coke.mp4") # coca cola
video = cv.VideoCapture("/home/gmavros/Desktop/sxolh_last/υπολογιστική όραση/tracker/videoTest/2022-05-01-181016.webm") # backround drifting

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480)) # save

if not video.isOpened():
    print("Cannot open camera")
    exit()

# Init tracker
_, frame = video.read()
#tracker.init(frame, np.array([298, 160, 48, 80])) # coca cola
tracker.init(frame, np.array([880, 450, 180, 100])) # backround drifting

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Last frame. Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find new roi
    nROI = tracker.update(frame)
    x = int(nROI[0])
    y = int(nROI[1])
    w = int(nROI[2])
    h = int(nROI[3])

    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    out.write(frame) # save
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
video.release()
out.release()
cv.destroyAllWindows()
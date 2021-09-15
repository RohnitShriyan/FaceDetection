import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capturing video
video_capture = cv2.VideoCapture(0)
while True:
    # Capturing frame-by-frame
    ret, frames = video_capture.read()

    # Converting to gray
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # detecting faces using the faceCascade
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frames)
    # Press "Q" to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # Stops video capturing and destroys all windows
        video_capture.release()
        cv2.destroyAllWindows()
import cv2 as cv
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    img = cv.resize(img, (640, 480))
    cv.imshow('main', img)
    key = cv.waitKey(1)
    if key == ord('w'):
        cv.imwrite(f"img{count}.jpg", img)
        count += 1
    elif key == ord('q'):
        quit()
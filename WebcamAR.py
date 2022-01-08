import cv2
import numpy
import math
from scipy.spatial import distance as dist

def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(4)
            h = camera.get(3)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                print(camera.getBackendName())

                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

# print(list_ports())

print("starting camera...")
cam = cv2.VideoCapture(3) # 1: internal webcam | 3: OBS Virtual Camera/Webcam
print("setting resolution...")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cam.set(cv2.CAP_PROP_FPS, 30)
print("finished starting camera")

cv2.namedWindow("display", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("webcam view")

insert = cv2.imread("Companion Cubes.jpg")
(insertHeight, insertWidth) = (insert.shape[0], insert.shape[1])

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    # frame = cv2.flip(frame, 1)
    (height, width) = (frame.shape[0], frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3) #apply blur to roi
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    canny = cv2.Canny(gray, 75, 15)

    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    largestQuad = None


    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4:
            if area > maxArea:
                largestQuad = approx
                maxArea = area

    if largestQuad is None:
        img = numpy.zeros((frame.shape[0], frame.shape[1], 3), numpy.uint8)
        img[:, :] = (255, 255, 255)
        cv2.imshow("display", img)
    else:
        try:
            M = cv2.moments(largestQuad)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            largestQuad = numpy.array(sorted(largestQuad, key=lambda coord: (math.atan2(coord[0][0] - cX, coord[0][1] - cY))))

            # xDiff = largestQuad[0][0][0] - largestQuad[1][0][0]
            # yDiff = largestQuad[0][0][1] - largestQuad[1][0][1]
            # h = math.sqrt(xDiff * xDiff + yDiff * yDiff)
            #
            # xDiff = largestQuad[0][0][0] - largestQuad[3][0][0]
            # yDiff = largestQuad[0][0][1] - largestQuad[3][0][1]
            # w = math.sqrt(xDiff * xDiff + yDiff * yDiff)

            [_, _, w, h] = cv2.boundingRect(largestQuad)

            scale = max(w / 1920, h / 1080)
            l = (0 - 1920 / 2) * scale + 1920 / 2
            r = (1920 / 2) * scale + 1920 / 2
            u = (0 - 1080 / 2) * scale + 1080 / 2
            d = (1080 / 2) * scale + 1080 / 2

            startingRect = numpy.float32([[[l, u]], [[l, d]], [[r, d]], [[r, u]]])

            cv2.drawContours(frame, [largestQuad], -1, (200, 200, 200), 2)

            insertMatrix = cv2.getPerspectiveTransform(numpy.float32([[[0, 0]], [[0, insertHeight]], [[insertWidth, insertHeight]], [[insertWidth, 0]]]), startingRect)
            img = cv2.warpPerspective(insert, insertMatrix, (1920, 1080), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            matrix = cv2.getPerspectiveTransform(numpy.float32(largestQuad), startingRect)
            newImg = cv2.warpPerspective(img, matrix, (1920, 1080), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            bordersize = 50
            border = cv2.copyMakeBorder(
                img,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            cv2.imshow("display", border)
        except Exception as e:
            print(e)

    cv2.imshow("webcam view", frame)# cv2.addWeighted(frame, 1, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR), 1, 0.0))

    # cv2.drawContours(thresh, [maxRect], -1, 127, -1)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()

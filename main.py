import cv2
import numpy as np
import os
import utlis

########################################################################
webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg = 480
########################################################################

# Prepare save directory and counter
save_dir = "Scanned"
os.makedirs(save_dir, exist_ok=True)
count = 0

utlis.initializeTrackbars()

while True:

    # 1) Grab frame
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    # 2) Preprocess
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # 3) Thresholding
    thres = utlis.valTrackbars()
    imgCanny = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # 4) Find and draw contours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, _ = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)

        # 5) Warp perspective
        pts1 = np.float32(biggest)
        pts2 = np.float32([
            [0, 0],
            [widthImg, 0],
            [0, heightImg],
            [widthImg, heightImg]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        # crop 20px margins & resize back
        imgWarpColored = imgWarpColored[20:-20, 20:-20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # 6) Adaptive threshold for a “scanned” look
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(
            imgWarpGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 7, 2
        )
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        imageArray = (
            [img, imgGray, imgThreshold, imgContours],
            [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre]
        )
    else:
        imageArray = (
            [img, imgGray, imgThreshold, imgContours],
            [imgBlank, imgBlank, imgBlank, imgBlank]
        )

    # 7) Stack and show
    labels = [
        ["Original", "Gray", "Threshold", "Contours"],
        ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]
    ]
    stackedImage = utlis.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    # 8) Key handling: save on ‘s’, quit on ‘q’
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and biggest.size != 0:
        # zero‑padded filename
        filename = f"myImage_{count:03d}.jpg"
        full_path = os.path.join(save_dir, filename)
        cv2.imwrite(full_path, imgWarpColored)

        # draw symmetric feedback box + text
        h, w = stackedImage.shape[:2]
        box_w, box_h = 460, 100
        box_tl = (w//2 - box_w//2, h//2 - box_h//2)
        box_br = (w//2 + box_w//2, h//2 + box_h//2)
        cv2.rectangle(stackedImage, box_tl, box_br, (0, 255, 0), cv2.FILLED)
        cv2.putText(
            stackedImage, "Scan Saved",
            (w//2 - 130, h//2 + 15),
            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA
        )
        cv2.imshow("Result", stackedImage)
        cv2.waitKey(300)

        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

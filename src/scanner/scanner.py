import time
from enum import Enum
import cv2
import numpy as np

from src.settings import (
    IMAGE_THRESHOLDS,
    IMAGE_CAPTURE_SIZE,
    CARD_DIMENSIONS,
    IMAGE_CAPTURE_FPS,
)


class State(Enum):
    UNSTABLE = 1
    SCANNING = 2
    DETECTED = 3
    FOUND = 4


def _biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def _reorder(biggest_contour):
    try:
        dots = biggest_contour.reshape((4, 2))
    except Exception:
        return None
    new_dots = np.zeros((4, 1, 2), dtype=np.int32)
    add = dots.sum(1)

    new_dots[0] = dots[np.argmin(add)]
    new_dots[3] = dots[np.argmax(add)]
    diff = np.diff(dots, axis=1)
    new_dots[1] = dots[np.argmin(diff)]
    new_dots[2] = dots[np.argmax(diff)]

    return new_dots


def process(
        input_source,
        thresholds=IMAGE_THRESHOLDS,
        capture_size=IMAGE_CAPTURE_SIZE,
        card_dimensions=CARD_DIMENSIONS,
):
    cam = cv2.VideoCapture(input_source)
    # set framerate
    cam.set(cv2.CAP_PROP_FPS, IMAGE_CAPTURE_FPS)
    count = 0
    control_time = time.time()
    while True:

        count += 1

        ret, frame = cam.read()
        if not ret:
            print("Error reading frame")
            break

        kernel = np.ones((5, 5))

        # apply filters
        frame = cv2.resize(frame, (capture_size.WIDTH, capture_size.HEIGHT))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 1)
        canny_frame = cv2.Canny(blur_frame, thresholds[0], thresholds[1])
        dilated_frame = cv2.dilate(canny_frame, kernel, iterations=2)
        eroded_frame = cv2.erode(dilated_frame, kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(
            eroded_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # find the largest with only 4 corners and draw it
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        biggest_contour, _ = _biggestContour(contours)

        blank_frame = np.zeros((capture_size.HEIGHT, capture_size.WIDTH, 3), np.uint8)

        if biggest_contour.size != 0:
            f_frame = frame.copy()
            reordered_corners = _reorder(biggest_contour)

            # use perspective transform to warp
            pts1 = np.float32(reordered_corners)
            pts2 = np.float32(
                [
                    [0, 0],
                    [capture_size.WIDTH, 0],
                    [0, capture_size.HEIGHT],
                    [capture_size.WIDTH, capture_size.HEIGHT],
                ]
            )
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped_frame = cv2.warpPerspective(
                frame, matrix, (capture_size.WIDTH, capture_size.HEIGHT)
            )

            # draw bounding box
            cv2.drawContours(f_frame, [biggest_contour], -1, (0, 255, 0), 5)
            imageArray = [f_frame, warped_frame]

            # save image one image per 2 seconds
            if time.time() - control_time > 2:
                # crop 10 pixels from each side
                cropped_frame = warped_frame[
                                10: capture_size.HEIGHT - 10,
                                10: capture_size.WIDTH - 10,
                                ]
                print("saving image", count, "at", time.time())
                cv2.imwrite("../photos/img_{}.png".format(count), cropped_frame)
                control_time = time.time()

        else:
            imageArray = [frame, blank_frame]

        # display
        cv2.imshow("frame", imageArray[0])
        cv2.imshow("warped", imageArray[1])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def run():
    process("http://10.0.0.149:8080/videofeed" or 0)


if __name__ == "__main__":
    run()

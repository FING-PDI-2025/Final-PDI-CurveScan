import cv2
import cv2 as cv
import numpy as np
from rich.console import Console

console = Console()

class Utils:
    @staticmethod
    def console_dump_bgr(image: np.ndarray) -> None:
        """
        Dump the image to the console using rich
        """

        concatenated = ""
        for i in range(0, image.shape[0], 1):
            for j in range(0, image.shape[1], 1):
                b, g, r = image[i, j]
                concatenated += f"[on rgb({r},{g},{b})]  [/on rgb({r},{g},{b})]"
            concatenated += "\n"
        console.print(concatenated)

    @staticmethod
    def show_image(*image: np.ndarray, wait=True, offset=0) -> None:
        """
        Show the image using OpenCV
        """
        for i, img in enumerate(image):
            window = f"Image {i + offset}"
            cv.namedWindow(window, cv2.WINDOW_NORMAL)
            cv.imshow(window, img)
            pos = (i + offset)
            x = pos % 12
            y = pos // 12
            size = 900
            cv.moveWindow(window, size * x + 1, size * y + 1)
            cv.resizeWindow(window, size, size)
        if not wait:
            return
        cv.waitKey(0)
        cv.destroyAllWindows()

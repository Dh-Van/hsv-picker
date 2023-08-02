import cv2 as cv
import numpy as np
from typing import *
from nptyping import NDArray

def find_contours(
    color_image: NDArray[(Any, Any, 3), np.uint8],
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
) -> List[NDArray]:
    """
    Finds all contours of the specified color range in the provided image.

    Args:
        color_image: The color image in which to find contours,
            with pixels represented in the bgr (blue-green-red) format.
        hsv_lower: The lower bound for the hue, saturation, and value of colors
            to contour.
        hsv_upper: The upper bound for the hue, saturation, and value of the colors
            to contour.

    Returns:
        A list of contours around the specified color ranges found in color_image.

    Note:
        Each channel in hsv_lower and hsv_upper ranges from 0 to 255.

    Example::

        # Define the lower and upper hsv ranges for the color blue
        BLUE_HSV_MIN = (90, 50, 50)
        BLUE_HSV_MAX = (110, 255, 255)

        # Extract contours around all blue portions of the current image
        contours = rc_utils.find_contours(
            rc.camera.get_color_image(), BLUE_HSV_MIN, BLUE_HSV_MAX
        )
    """
    assert (
        0 <= hsv_lower[0] <= 179 and 0 <= hsv_upper[0] <= 179
    ), f"The hue of hsv_lower ({hsv_lower}) and hsv_upper ({hsv_upper}) must be in the range 0 to 179 inclusive."

    assert (
        0 <= hsv_lower[1] <= 255 and 0 <= hsv_upper[1] <= 255
    ), f"The saturation of hsv_lower ({hsv_lower}) and hsv_upper ({hsv_upper}) must be in the range 0 to 255 inclusive."

    assert (
        0 <= hsv_lower[0] <= 255 and 0 <= hsv_upper[0] <= 255
    ), f"The value of hsv_lower ({hsv_lower}) and hsv_upper ({hsv_upper}) must be in the range 0 to 255 inclusive."

    assert (
        hsv_lower[1] <= hsv_upper[1]
    ), f"The saturation channel of hsv_lower ({hsv_lower}) must be less than that of hsv_upper ({hsv_upper})."

    assert (
        hsv_lower[2] <= hsv_upper[2]
    ), f"The value channel of hsv_lower ({hsv_lower}) must be less than that of of hsv_upper ({hsv_upper})."

    # Convert the image from a blue-green-red pixel representation to a
    # hue-saturation-value representation
    hsv_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

    # Create a mask containing the pixels in the image with hsv values between
    # hsv_lower and hsv_upper.
    mask: NDArray
    if hsv_lower[0] <= hsv_upper[0]:
        mask = cv.inRange(hsv_image, hsv_lower, hsv_upper)

    # If the color range passes the 255-0 boundary, we must create two masks
    # and merge them
    else:
        mask1 = cv.inRange(hsv_image, hsv_lower, (255, hsv_upper[1], hsv_upper[2]))
        mask2 = cv.inRange(hsv_image, (0, hsv_lower[1], hsv_lower[2]), hsv_upper)
        mask = cv.bitwise_or(mask1, mask2)

    # Find and return a list of all contours of this mask
    return cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]


def get_largest_contour(
    contours: List[NDArray], min_area: int = 30
) -> Optional[NDArray]:
    """
    Finds the largest contour with size greater than min_area.

    Args:
        contours: A list of contours found in an image.
        min_area: The smallest contour to consider (in number of pixels)

    Returns:
        The largest contour from the list, or None if no contour was larger
        than min_area.

    Example::

        # Extract the blue contours
        BLUE_HSV_MIN = (90, 50, 50)
        BLUE_HSV_MAX = (110, 255, 255)
        contours = rc_utils.find_contours(
            rc.camera.get_color_image(), BLUE_HSV_MIN, BLUE_HSV_MAX
        )

        # Find the largest contour
        largest_contour = rc_utils.get_largest_contour(contours)
    """
    # Check that the list contains at least one contour
    if len(contours) == 0:
        return None

    # Find and return the largest contour if it is larger than min_area
    greatest_contour = max(contours, key=cv.contourArea)
    if cv.contourArea(greatest_contour) < min_area:
        return None

    return greatest_contour


def draw_contour(
    color_image: NDArray[(Any, Any, 3), np.uint8],
    contour: NDArray,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> None:
    """
    Draws a contour on the provided image.

    Args:
        color_image: The color image on which to draw the contour.
        contour: The contour to draw on the image.
        color: The color to draw the contour, specified as
            blue-green-red channels each ranging from 0 to 255 inclusive.

    Example::

        image = rc.camera.get_color_image()

        # Extract the largest blue contour
        BLUE_HSV_MIN = (90, 50, 50)
        BLUE_HSV_MAX = (110, 255, 255)
        contours = rc_utils.find_contours(image, BLUE_HSV_MIN, BLUE_HSV_MAX)
        largest_contour = rc_utils.get_largest_contour(contours)

        # Draw this contour onto image
        if (largest_contour is not None):
            draw_contour(image, largest_contour)
    """
    for channel in color:
        assert (
            0 <= channel <= 255
        ), f"Each channel in color ({color}) must be in the range 0 to 255 inclusive."

    cv.drawContours(color_image, [contour], 0, color, 3)
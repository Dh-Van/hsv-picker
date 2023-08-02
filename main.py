import color_finder as cf


"""
INPUTS
"""
IMAGE_PATH = "resources/example.png"
EXPECTED_HUE_RANGE = [40, 80]
HSV_RANGE = [(40, 50, 150), (80, 255, 255)]
MANUAL = True

if(MANUAL):
    color_finder = cf.ColorFinder(IMAGE_PATH, HSV_RANGE, True)

if(not MANUAL):
    color_finder = cf.ColorFinder(IMAGE_PATH, EXPECTED_HUE_RANGE, False)

running = True

while(running):
    running = color_finder.refresh()
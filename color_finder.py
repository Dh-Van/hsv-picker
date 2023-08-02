import cv2, numpy as np, contours

class ColorFinder:
    image = None
    cache = None
    hsv_image = None

    hue_range = []
    # current_hsv_range = [(90, 50, 80), (110, 200, 255)]
    current_hsv_range = []

    mouse_x, mouse_y = 0, 0
    drawing = False

    def __init__(self, path, hue_range):
        self.image = cv2.imread(path)
        self.cache = self.image.copy()
        self.hue_range = hue_range
        cv2.namedWindow("Color Finder")
        cv2.setMouseCallback("Color Finder", self.crop_image)

    def crop_image(self, event, x, y, flags, param):
        if(event == cv2.EVENT_LBUTTONDOWN):
            self.drawing = True
            self.mouse_x, self.mouse_y = x, y
        elif(event == cv2.EVENT_MOUSEMOVE):
            if(self.drawing): cv2.rectangle(self.cache, (self.mouse_x, self.mouse_y), (x, y), (0, 255, 0), 2)
        elif(event == cv2.EVENT_LBUTTONUP):
            self.drawing = False
            cv2.rectangle(self.image, (self.mouse_x, self.mouse_y), (x, y), (0, 255, 255), 2)
            if(y < self.mouse_y):
                self.image = self.image[y + 3:self.mouse_y-3, x+3:self.mouse_x-3]
            else:
                self.image = self.image[self.mouse_y+3:y-3, self.mouse_x+3:x-3]

            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def get_hsv_range(self):
        output_h, output_s, output_v = [], [], []
        for col in self.hsv_image:
            for row in col:
                if(row[0] >= self.hue_range[0] and row[0] <= self.hue_range[1]):
                    output_h.append(row[0])
                    output_s.append(row[1])
                    output_v.append(row[2])


        if(len(output_h) < 1):
            print("Invalid expected hsv range. Try again")
            return (0, 0, 0), (0, 0, 0)

        output_h.sort()
        output_s.sort()
        output_v.sort()

        return [(int(output_h[0]), int(output_s[0]), int(output_v[0])), (int(output_h[-1]), int(output_s[-1]), int(output_v[-1]))]

    def refresh(self):
        cv2.imshow("Color Finder", self.cache)
        self.cache = self.image.copy()

        if(self.hsv_image is not None):
            if(len(self.current_hsv_range) < 1):
                self.current_hsv_range = self.get_hsv_range()
            print(self.current_hsv_range)
            list_contours = contours.find_contours(self.cache, self.current_hsv_range[0], self.current_hsv_range[1])
            lg_contour = contours.get_largest_contour(list_contours, 30)
            contours.draw_contour(self.cache, lg_contour)


        if cv2.waitKey(10) == ord('q'):
            return False
        else:
            return True
        

color_finder = ColorFinder("resources/purpleLine2.png", (110, 150))
running = True

while(running):
    running = color_finder.refresh()

cv2.destroyAllWindows()

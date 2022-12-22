
class BoundingBox:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


    def get_x1(self):
        return self.x1


    def get_y1(self):
        return self.y1


    def get_x2(self):
        return self.x2


    def get_y2(self):
        return self.y2


    def get_width(self):
        return self.x2 - self.x1

    
    def get_height(self):
        return self.y2 - self.y1

    # to string
    def __str__(self):
        return "x1: {}, y1: {}, x2: {}, y2: {}".format(self.x1, self.y1, self.x2, self.y2)


    
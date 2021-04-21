from detect_v import detect_v
from detect_to_wear_mask import detect_to_wear_mask

class mask:
    def __init__(self):
        #self.detect_mask()
        s = self.d_mask()


    def d_mask(self):
        m = detect_to_wear_mask()
        self.mask = m.test()

        if self.mask == 1:
            #어쩌구저쩌구 말하고
            self.d_v()

    def d_v(self):
        print('d_v')
        de = detect_v()
        v = de.test()
        if v == 1:
            print('ok')


if __name__ == '__main__':
    detect = mask()



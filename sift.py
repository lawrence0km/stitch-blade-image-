import cv2
import numpy as np

class Matcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher()
        self.raw_match = None
        self.goods = None

    def knn_match(self, desc1, desc2, k=2):
        # KNN Matching
        self.raw_match = self.matcher.knnMatch(desc1, desc2, k)
        return self.raw_match

    def good_matching(self, ratio=0.7):
        # filter good matching points
        self.goods = []
        for first, second in self.raw_match:
            if first.distance < second.distance * ratio:
                self.goods.append((first.trainIdx, first.queryIdx))
        return self.goods

    def form(self, kp1, kp2):
        if len(self.goods) > 0:
            psta = np.array([kp1[i].pt for _, i in self.goods])
            src_match = psta[:, :2]
            pstb = np.array([kp2[i].pt for i, _ in self.goods])
            dst_match = pstb[:, :2]
        else:
            src_match = []
            dst_match = []
        return src_match, dst_match # [len, 2], [len, 2]

    def detect(self, *args, **kwargs):
        raise NotImplementedError

    def raw_match(self):
        return self.raw_match

    def good_match(self):
        return self.goods

    def thread(self, img1, img2, ratio=0.7):
        kp1, desc1 = self.detect(img1)
        kp2, desc2 = self.detect(img2)
        if np.any(desc1) == None or np.any(desc2) == None:
            return [], []
        self.knn_match(desc1, desc2)
        self.good_matching(ratio)
        src_match, dst_match = self.form(kp1, kp2)
        return src_match, dst_match
    
class SIFTMatcher(Matcher):
    def __init__(self):
        super().__init__()
        self.agent = cv2.SIFT_create()

    def detect(self, img):
        # return (kp, desc)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.agent.detectAndCompute(gray, None)
    


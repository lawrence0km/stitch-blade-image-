import cv2
import numpy as np
import math
from sift import SIFTMatcher
from utils import EstimateRANSAC, AffineLine, GetJointPoint
from transform import AffineTransform

class StitchBase:
    def __init__(self) -> None:
        self.imgs = []
        self.masks = []

    def Push(self, img, mask):
        self.imgs.append(img)
        self.masks.append(mask)

class LineManager(StitchBase):
    def __init__(self) -> None:
        super().__init__()
        self.mask_lines = []

    def EdgesDetect(self, reference_angle, std_angle=10):
        masks_lines = []
        min_angle, max_angle = reference_angle - std_angle, reference_angle + std_angle
        for mask in self.masks:
            mask_lines = []
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(mask_gray, threshold1=50, threshold2=150)
            mask_height, mask_width = mask.shape[:2]
            lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=25, minLineLength=mask_height//8, maxLineGap=mask_height//4) 
            if lines is None or len(lines) < 2:
                continue
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < min_angle or angle_deg > max_angle:
                    continue
                intercept = y1 - slope * x1
                length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                mask_lines.append([angle_deg, intercept, length])
            edge1, edge2 = self.EdgesNMS(mask_lines)
            edge = self.GetMidEdge([edge1, edge2])
            masks_lines.append([edge, edge1, edge2])
        self.mask_lines = masks_lines
        return masks_lines
    
    def EdgesNMS(self, edges):
        edges = np.array(edges)
        mean = np.mean(edges[:, 1])
        index1 = np.where(edges[:, 1] <= mean)
        index2 = np.where(edges[:, 1] > mean)
        edges1, edges2 = edges[index1], edges[index2]
        edge1 = edges1[np.argmax(edges1[:, 2])]
        edge2 = edges2[np.argmax(edges2[:, 2])]
        return edge1, edge2
    
    def GetMidEdge(self, edges):
        edge1, edge2 = edges[0], edges[1]
        # edge = [(edge1[0] + edge2[0]) / 2.0, (edge1[1] + edge2[1]) / 2.0]

        angle1_radians = np.deg2rad(edge1[0])
        angle2_radians = np.deg2rad(edge2[0])
        A1 = -np.tan(angle1_radians)
        B1 = 1
        C1 = -edge1[1]
        A2 = -np.tan(angle2_radians)
        B2 = 1
        C2 = -edge2[1]
        A = np.array([[A1, B1],
                    [A2, B2]])

        b = np.array([[-C1],
                    [-C2]])
        intersection = np.linalg.solve(A, b)
        x_intersection = intersection[0, 0]
        y_intersection = intersection[1, 0]
        
        theta = (edge1[0] + edge2[0]) / 2.0
        interect = (edge1[1] + edge2[1]) / 2.0
        if x_intersection < 0:
            theta += 180
        return [theta, interect]
    

    # init_scale 有camera-blade相对距离决定
    def Align(self, idx1, idx2, init_scale=1.0):
        # 默认idx1比idx2宽
        assert len(self.mask_lines) > 0

        transform1 = AffineTransform(1.0, self.mask_lines[idx1][0][0] - 90.0, (0, 0))
        transform2 = AffineTransform(init_scale, self.mask_lines[idx2][0][0] - 90.0, (0, 0))

        lines1 = AffineLine((self.masks[idx1].shape[1], self.masks[idx1].shape[0]), transform1.GetMatrix(), self.mask_lines[idx1])
        lines2 = AffineLine((self.masks[idx2].shape[1], self.masks[idx2].shape[0]), transform2.GetMatrix(), self.mask_lines[idx2])

        # 求纵坐标的平移
        line_y0 = [0.0, 0.0, np.inf]
        line_yheight = [0.0, self.masks[idx1].shape[0], np.inf]
        w11 = abs(GetJointPoint(line_y0, lines1[1])[0] - GetJointPoint(line_y0, lines1[2])[0])
        w21 = abs(GetJointPoint(line_y0, lines2[1])[0] - GetJointPoint(line_y0, lines2[2])[0])
        w12 = abs(GetJointPoint(line_yheight, lines1[1])[0] - GetJointPoint(line_yheight, lines1[2])[0])
        offset_y = (w21 - w11) * (self.masks[idx1].shape[0] - 0) / (w12 - w11) + 0
        
        # 求横坐标的平移
        x1 = lines1[0][2]
        x2 = lines2[0][2]
        offset_x = x1 - x2

        angle1 = self.mask_lines[idx1][0][0] - 90.0
        angle2 = self.mask_lines[idx2][0][0] - 90.0
        scale = init_scale
        offset = (offset_x, offset_y)
        return angle1, angle2, scale, offset

    def AlignImages(self, init_scales=[]):
        if len(init_scales) < len(self.masks):
            init_scales = [1.0] * len(self.masks)
        affine_collects = []
        affine_collects.append({"angle": self.mask_lines[0][0][0] - 90.0, "scale": 1.0, "offset": (0, 0)})
        for i in range(len(self.masks) - 1):
            angle1, angle2, scale, offset = self.Align(i, i+1, init_scales[i+1])
            while offset[1] < 0:
                init_scales[i+1] *= 0.8
                angle1, angle2, scale, offset = self.Align(i, i+1, init_scales[i+1])
            affine_collects.append({"angle": angle2, "scale": scale, "offset": offset})
        return affine_collects


class CoarseStitch(StitchBase):
    def __init__(self) -> None:
        super().__init__()
        self.images_path = []
        self.matcher = SIFTMatcher()

    def PushImagePath(self, img_path):
        self.images_path.append(img_path)

    def BladeExtract(self, idx):
        img = cv2.imread(self.images_path[idx])
        img_height, img_width = img.shape[:2]
        mask_gray = cv2.cvtColor(self.masks[idx], cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.resize(mask_gray, (img_width, img_height), None, interpolation=cv2.INTER_NEAREST)
        img = cv2.bitwise_and(img, img, mask=mask_gray)
        kernel = np.ones((15, 15), np.uint8)
        opened_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
        return opened_image
    
    def ShowMatch(self, idx1, idx2, points1, points2):
        img1 = self.imgs[idx1]
        img2 = self.imgs[idx2]

        width = img1.shape[1] + img2.shape[1]
        height = max(img2.shape[0], img2.shape[0])

        img = np.zeros((height, width, img1.shape[2]), dtype=np.uint8)

        img[0: img1.shape[0], 0: img1.shape[1], :] = img1
        img[0: img2.shape[0], img1.shape[1]: img1.shape[1] + img2.shape[1], :] = img2

        points2[:, 0] +=  img1.shape[1]

        points1 = points1.astype(np.int32)
        points2 = points2.astype(np.int32)

        for p1, p2 in zip(points1, points2):
            cv2.circle(img, (p1[0], p1[1]), 3, (255, 0, 0), -1)
            cv2.circle(img, (p2[0], p2[1]), 3, (0, 0, 255), -1)
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 3)
        
        cv2.imshow("match points", img)
        cv2.waitKey(-1)

    
    def DetectMatchKeypoints(self, idx1, idx2, thresh=0.3):
        img1 = self.BladeExtract(idx1)
        img2 = self.BladeExtract(idx2)
        match1, match2 = self.matcher.thread(img1, img2, thresh)

        if len(match1) < 1:
            return None, None

        fx = self.masks[idx1].shape[1] / img1.shape[1]
        fy = self.masks[idx1].shape[0] / img1.shape[0]
        match1_resize = match1 * np.array([fx, fy])

        fx = self.masks[idx2].shape[1] / img1.shape[1]
        fy = self.masks[idx2].shape[0] / img1.shape[0]
        match2_resize = match2 * np.array([fx, fy])


        dis_threshold = 100
        best_transform, best_inliers = EstimateRANSAC(match2_resize, match1_resize, 500, inlier_threshold = dis_threshold)
        match1_resize = match1_resize[best_inliers]
        match2_resize = match2_resize[best_inliers]
        return match1_resize, match2_resize, best_transform

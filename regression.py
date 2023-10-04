import numpy as np
import matplotlib.pyplot as plt
import cv2
from sift import SIFTMatcher
from transform import AffineTransform
from utils import GetFinalSize
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.geometry import mapping

class Regression:
    def __init__(self, aligen_params) -> None:
        self.aligen_params = aligen_params
    
    def SetPair(self, idx1, idx2, img1, img2, mask1, mask2):
        self.idx1 = idx1
        self.idx2 = idx2
        self.img1 = img1
        self.img2 = img2
        self.mask1 = mask1
        self.mask2 = mask2

        self.match1 = None
        self.match2 = None

        self.r = self.aligen_params[idx2]["angle"]
        self.s = self.aligen_params[idx2]["scale"]
        self.tx = self.aligen_params[idx2]["offset"][0]
        self.ty = self.aligen_params[idx2]["offset"][1]


    def ExtractBlade(self, img, mask):
        img_height, img_width = img.shape[:2]
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.resize(mask_gray, (img_width, img_height), None, interpolation=cv2.INTER_NEAREST)
        img = cv2.bitwise_and(img, img, mask=mask_gray)
        kernel = np.ones((15, 15), np.uint8)
        opened_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
        return opened_image

    def DetectMatchKeypoints(self, thresh=0.22):
        img1 = self.ExtractBlade(self.img1, self.mask1)
        img2 = self.ExtractBlade(self.img2, self.mask2)
        matcher = SIFTMatcher()
        match1, match2 = matcher.thread(img1, img2, thresh)
        return match1, match2

    def GetMatchLoss(self):
        if np.any(self.match1) == None or np.any(self.match2) == None:
            match1, match2 = self.DetectMatchKeypoints()
            self.match1, self.match2 = match1, match2
        else:
            match1, match2 = self.match1, self.match2
        match_points = len(match1)
        if match_points == 0:
            return 0, match_points
        img1_height, img1_width = self.img1.shape[:2]
        mask1_height, mask1_width = self.mask1.shape[:2]

        img2_height, img2_width = self.img2.shape[:2]
        mask2_height, mask2_width = self.mask2.shape[:2]

        fx1, fy1 = mask1_width / img1_width, mask1_height / img1_height
        fx2, fy2 = mask2_width / img2_width, mask2_height / img2_height
        mask_match1 = match1 * np.array([fx1, fy1])
        mask_match2 = match2 * np.array([fx2, fy2])

        aligen_params1 = self.aligen_params[self.idx1]

        affine_transform1 = AffineTransform(1.0, aligen_params1["angle"], (0, 0))
        affine_transform2 = AffineTransform(self.s, self.r, (self.tx, self.ty))

        mask_match1 = affine_transform1.WarpPoints(mask_match1)
        mask_match2 = affine_transform2.WarpPoints(mask_match2)

        match_loss = np.sum(np.linalg.norm(mask_match2 - mask_match1, axis=1)) / match_points
        return match_loss, match_points
    
    def GetShapeLoss(self):
        aligen_params1 = self.aligen_params[self.idx1]
        affine_transform1 = AffineTransform(1.0, aligen_params1["angle"], (0, 0))
        affine_transform2 = AffineTransform(self.s, self.r, (self.tx, self.ty))
        (new_width, new_height), (offset_x, offset_y) = affine_transform1.GetWarpSize(self.mask1)
        mask1 = cv2.warpAffine(self.mask1, affine_transform1.GetOffsetMatrix((offset_x, offset_y)), (new_width, new_height))
        mask1[mask1 > 0] = 1

        (new_width, new_height), (offset_x, offset_y) = affine_transform2.GetWarpSize(self.mask2)
        mask2 = cv2.warpAffine(self.mask2, affine_transform2.GetOffsetMatrix((offset_x, offset_y)), (new_width, new_height))
        mask2[mask2 > 0] = 2

        (width, height), (image_offset_x, image_offset_y) = \
            GetFinalSize([self.mask1, self.mask2], [affine_transform1, affine_transform2])
        fusion_mask = np.zeros((height, width, 3), dtype=np.uint8)

        affine_transform1 = AffineTransform(1.0, aligen_params1["angle"], (image_offset_x, image_offset_y))
        affine_transform2 = AffineTransform(self.s, self.r, (self.tx + image_offset_x, self.ty + image_offset_y))

        (min_x, min_y), (max_x, max_y) = affine_transform1.GetImageRange(self.mask1)
        mask1 = cv2.resize(mask1, (max_x - min_x, max_y - min_y), interpolation=cv2.INTER_NEAREST)
        fusion_mask[min_y: max_y, min_x: max_x, :] += mask1
        
        (min_x, min_y), (max_x, max_y) = affine_transform2.GetImageRange(self.mask2)
        mask2 = cv2.resize(mask2, (max_x - min_x, max_y - min_y), interpolation=cv2.INTER_NEAREST)
        fusion_mask[min_y: min_y+mask2.shape[0], min_x: max_x, :] += mask2

        # 提取公共区域
        height, width = self.mask1.shape[:2]
        corners1 = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ])
        height, width = self.mask2.shape[:2]
        corners2 = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ])

        corners1 = affine_transform1.GetMatrix().dot(corners1.T).T.astype(np.int32)
        corners2 = affine_transform2.GetMatrix().dot(corners2.T).T.astype(np.int32)

        polygon1 = Polygon(corners1)
        polygon2 = Polygon(corners2)
        intersection_polygon = polygon1.intersection(polygon2)
        intersection_geojson = mapping(intersection_polygon)
        coordinates = np.array(intersection_geojson['coordinates']).squeeze()
        if len(coordinates) < 3:
            return 1
        intersect = np.array(intersection_polygon.exterior.coords).astype(np.int32)
        intersect_mask = np.zeros((fusion_mask.shape[0], fusion_mask.shape[1], 3), dtype=np.uint8)
        intersect_mask = cv2.fillPoly(intersect_mask, [intersect], (1, 1, 1))
        fusion_mask = cv2.bitwise_and(fusion_mask, fusion_mask, intersect_mask)
        count_1 = np.sum(fusion_mask == 1)
        count_2 = np.sum(fusion_mask == 2)
        count_3 = np.sum(fusion_mask == 3)
        epli = 0.000001 
        Ap2c = count_3 / (count_3 + count_1 + epli)
        Ac2p = count_3 / (count_2 + count_3 + epli)
        shape_loss = 0.5 * ((1 - Ap2c)**2 + (1 - Ac2p) ** 2 + (Ap2c - Ac2p) ** 2)
        return shape_loss
    
    def ScaleConstraint(self):
        init_s = self.aligen_params[self.idx2]["scale"]
        return max(init_s, self.s) / min(init_s, self.s) - 1.0
    
    def RotationConstraint(self):
        init_r = self.aligen_params[self.idx2]["angle"]
        return abs(init_r - self.r)
    
    def OverlapConstraint(self):
        pass

    def OptimizeFunction(self, params, M_lower=3, M_upper=8):
        self.r, self.s, self.tx, self.ty = params
        match_loss, match_points = self.GetMatchLoss()
        shape_loss = self.GetShapeLoss()
        scale_constraint = self.ScaleConstraint()
        rotation_constraint = self.RotationConstraint()

        if match_points < M_lower:
            weight = [0, 500, 1, 1, 1]
        elif match_points >= M_upper:
            weight = [5, 0, 1, 1, 1]
        else:
            weight = [5, 500 * M_lower / match_points, 1, 1, 1]

        loss = weight[0] * match_loss + weight[1] * shape_loss + weight[2] * scale_constraint + weight[3] * rotation_constraint
        # print("match loss: ", match_loss, "shape loss: ", shape_loss, "scale constraint: ", scale_constraint, "rotation constraint: ", rotation_constraint)
        # print("R S T :", self.r, self.s, self.tx, self.ty)
        return loss
    
    def Optimize(self):
        self.GetMatchLoss()
        initial_params = [self.r, self.s, self.tx, self.ty]
        print("before optimize: ", initial_params)
        bounds = [(self.r-1, self.r+1), (self.s-0.02, self.s+0.02), (self.tx-2, self.tx+2), (self.ty-2, self.ty+2)]
        result = minimize(self.OptimizeFunction, initial_params, method='Nelder-Mead', bounds=bounds)
        theta_opt, s_opt, tx_opt, ty_opt = result.x
        print("after optimize: ",  theta_opt, s_opt, tx_opt, ty_opt)
        return theta_opt, s_opt, tx_opt, ty_opt
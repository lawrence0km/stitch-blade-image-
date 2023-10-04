import numpy as np
import cv2
from copy import copy

class AffineTransform:
    def __init__(self, scale : float, angle : float, translate: tuple) -> None:
        self.scale = scale
        self.angle = angle
        self.translate = translate
        self.affine_matrix = cv2.getRotationMatrix2D((0, 0), angle=angle, scale=scale)
        self.affine_matrix[0, 2], self.affine_matrix[1, 2] = translate[0], translate[1]

        
    def GetMatrix(self):
        return self.affine_matrix
    
    def GetOffsetMatrix(self, image_offset):
        affine_matrix = copy(self.affine_matrix)
        affine_matrix[0, 2] = image_offset[0]
        affine_matrix[1, 2] = image_offset[1]
        return affine_matrix

    def WarpPoints(self, points: np.ndarray):
        points = np.array([points])
        points = cv2.transform(points, self.affine_matrix)[0]
        return points
    
    def GetWarpSize(self, img):
        (min_x, min_y), (max_x, max_y) = self.GetImageRange(img)
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        new_width, new_height = max_x - min_x, max_y - min_y

        return (new_width, new_height), (offset_x, offset_y)

    def GetImageRange(self, img):
        height, width = img.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [0, height, 1],
            [width, height, 1]
        ])

        corners = self.affine_matrix.dot(corners.T).T.astype(np.int32)
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        return (min_x, min_y), (max_x, max_y)
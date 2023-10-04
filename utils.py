import numpy as np
import random
import cv2
from transform import AffineTransform


def EuclDistance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def EstimateRANSAC(source_points, target_points, num_iterations, inlier_threshold):
    best_transform = None
    best_inliers = []
    
    for _ in range(num_iterations):
        # 随机选择两个匹配点
        sample_indices = random.sample(range(len(source_points)), 2)
        source_sample = source_points[sample_indices]
        target_sample = target_points[sample_indices]
        
        # 计算欧氏变换参数
        translation = target_sample - source_sample
        scale = np.linalg.norm(translation)
        rotation = np.arctan2(translation[1], translation[0])
        
        # 应用欧氏变换
        transformed_source = source_points * scale
        rotated_source = np.dot(transformed_source, np.array([[np.cos(rotation), -np.sin(rotation)],
                                                             [np.sin(rotation), np.cos(rotation)]]))
        translated_source = rotated_source + translation
        
        # 计算内点（距离小于阈值的点）
        inliers = []
        for i in range(len(source_points)):
            if EuclDistance(target_points[i], translated_source[i]) < inlier_threshold:
                inliers.append(i)
        
        # 更新最佳变换和内点
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = (scale, rotation, translation)
    
    return best_transform, best_inliers

def AffineLine(img_size, affine_matrix, lines):
    # affine_matrix = np.vstack((affine_matrix, [0, 0, 1]))
    image_width, image_height = img_size
    new_lines = []
    for index, line in enumerate(lines):
        angle_radians = np.deg2rad(line[0])
        A = -np.tan(angle_radians)
        B = 1
        C = line[1]

        x_top = (C - B * 0) / A
        y_top = 0

        x_bottom = (C - B * image_height) / A
        y_bottom = image_height

        x_left = 0
        y_left = (C - A * 0) / B

        x_right = image_width
        y_right = (C - A * image_width) / B


        points = np.array([
            [x_top, y_top],
            [x_bottom, y_bottom],
            [x_left, y_left],
            [x_right, y_right]
        ])

        # points = np.dot(affine_matrix, points.T).T
        points = cv2.transform(points.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)

        x_top, y_top = points[0, 0], points[0, 1]
        x_bottom, y_bottom = points[1, 0], points[1, 1]

        theta = np.rad2deg(np.arctan2(y_bottom - y_top, x_bottom - x_top))
        if theta == 90.0 or theta == -90.0 or x_bottom - x_top == 0:
            x_interact = x_top
            y_interact = np.inf
        
        elif theta == 0.0 or theta == -0.0:
            x_interact = np.inf
            y_interact = 0

        else:
            slope = (y_bottom - y_top) / (x_bottom - x_top)
            y_interact = y_top - slope * x_top
            x_interact = -y_interact / slope
        new_lines.append([theta, y_interact, x_interact])
    return new_lines


def GetLineParam(line):
    angle_radians = np.deg2rad(line[0])
    if 89.91 < line[0] < 90.01 or -89.91 > line[0] > -90.01:
        A = -np.tan(angle_radians)
        B = 0
        C = line[2]
    else:
        A = -np.tan(angle_radians)
        B = 1
        C = line[1]
    return A, B, C

def GetJointPoint(line1, line2):
    A1, B1, C1 = GetLineParam(line1)
    A2, B2, C2 = GetLineParam(line2)
    determinant = A1 * B2 - A2 * B1

    if determinant != 0:
        x = (C1 * B2 - C2 * B1) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return [x, y]
    else:
        return None
    
def GetFinalSize(imgs, transforms):
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    for img, transform in zip(imgs, transforms):
        if isinstance(transform, AffineTransform):
            transform = transform.GetMatrix()
        height, width = img.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [0, height, 1],
            [width, height, 1]
        ])
        corners = transform.dot(corners.T).T.astype(np.int32)
        x, y = np.max(corners, axis=0)
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        x, y = np.min(corners, axis=0)
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
    width, height = max_x - min_x, max_y - min_y
    image_offset = [-min_x if min_x < 0 else 0, -min_y if min_y < 0 else 0]
    return (width, height), (image_offset[0], image_offset[1])
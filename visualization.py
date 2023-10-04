import numpy as np
import cv2
from transform import AffineTransform
from shapely.geometry import Polygon
from shapely.geometry import mapping
from copy import deepcopy

class Visualization:
    def __init__(self) -> None:
        self.imgs, self.masks = [], []
    
    def Push(self, img, mask):
        self.imgs.append(img)
        self.masks.append(mask)

    def FusionImage(self, xs, ys, origin_img, target_img, fusion_mask, method="cover"):
        '''
            The method contains cover and mean 
        '''
        fusion_mask_ = cv2.cvtColor(fusion_mask, cv2.COLOR_GRAY2BGR)
        fusion_mask_weight = fusion_mask / 255.0 / 4
        top, left = ys[0], xs[0]
        for x in range(xs[1] - xs[0]):
            for y in range(ys[1] - ys[0]):
                p1 = origin_img[top + y, left + x, :]
                p2 = target_img[y, x, :]
                fm = fusion_mask_[top + y, left + x, :]
                if method == "weight_fusion":
                    if np.all(p1) != 0 and np.all(p2) != 0:
                        w2 = fusion_mask_weight[top + y, left + x]
                        w1 = 1 - w2
                        origin_img[top + y, left + x, :] = np.uint8(w1 * p1 + w2 * p2)
                    elif np.all(p1) == 0 and np.all(p2) != 0:
                        origin_img[top + y, left + x, :] = p2
                    elif np.all(p1) != 0 and np.all(p2) == 0:
                        origin_img[top + y, left + x, :] = p1
                elif method == "vis_fusion":
                    if fm[0] > 0:
                        origin_img[top + y, left + x, :] = fm
                    elif np.all(p1) == 0 and np.all(p2) != 0:
                        origin_img[top + y, left + x, :] = p2
                    elif np.all(p1) != 0 and np.all(p2) == 0:
                        origin_img[top + y, left + x, :] = p1
                    else:
                        origin_img[top + y, left + x, :] = p2
                else:
                    if np.all(p1) == 0 and np.all(p2) != 0:
                        origin_img[top + y, left + x, :] = p2
                    elif np.all(p1) != 0 and np.all(p2) == 0:
                        origin_img[top + y, left + x, :] = p1
                    else:
                        origin_img[top + y, left + x, :] = p2
        return origin_img

    def GetIntercet(self, idx1, idx2, transforms):
        mask1 = self.masks[idx1]
        mask2 = self.masks[idx2]
        corners1 = np.array([[
            [0, 0],
            [mask1.shape[1], 0],
            [mask1.shape[1], mask1.shape[0]],
            [0, mask1.shape[0]]
        ]])

        corners2 = np.array([[
            [0, 0],
            [mask2.shape[1], 0],
            [mask2.shape[1], mask2.shape[0]],
            [0, mask2.shape[0]]
            ]])
        transform1 = transforms[idx1]
        transform2 = transforms[idx2]
        corners1 = cv2.transform(corners1, transform1)[0]
        corners2 = cv2.transform(corners2, transform2)[0]
        polygon1 = Polygon(corners1)
        polygon2 = Polygon(corners2)
        intersection_polygon = polygon1.intersection(polygon2)
        intersection_geojson = mapping(intersection_polygon)
        coordinates = np.array(intersection_geojson['coordinates']).squeeze()[:-1]
        if(len(coordinates) < 4):
            return None

        min_y_index = np.argmin(coordinates[:, 1])
        indexes_min = np.arange(0, min_y_index, step=1)
        indexes_max = np.arange(min_y_index, len(coordinates), step=1)
        coordinates = np.vstack((coordinates[indexes_max], coordinates[indexes_min]))
        return coordinates

    def BuildFusionMask(self, intercet, coarse_img):
            offset = intercet[0]
            points = np.array(intercet, dtype=np.float32)
            rect = cv2.minAreaRect(points)
            rotation_angle = rect[-1]
            img_height, img_width = int(np.linalg.norm(intercet[1] - intercet[0])), int(np.linalg.norm(intercet[3] - intercet[0]))
            image1 = np.zeros((img_height, img_width))
            for x in range(img_width):
                value = x / (img_width - 1)  # 计算每一列的值，范围从0到1
                image1[:, x] = value

            image2 = np.zeros((img_height, img_width))
            for y in range(img_height):
                value = y / (img_height - 1)  # 计算每一列的值，范围从0到1
                image2[y, :] = value

            image = image1 * 0.5 + image2 * 0.5
            image = np.uint8(image * 255)
            transform = AffineTransform(1.0, -rotation_angle, (offset))
            image = cv2.warpAffine(image, transform.GetMatrix(), (coarse_img.shape[1], coarse_img.shape[0]))
            return image

    def WarpImgMask(self, affine_collects):
        transforms = []
        last_scale, last_angle, last_translate = affine_collects[0]["scale"], \
                                                 affine_collects[0]["angle"], \
                                                 affine_collects[0]["offset"]
        affine_transform = AffineTransform(last_scale, last_angle, last_translate)
        transforms.append(affine_transform.GetMatrix())
        for idx, param in enumerate(affine_collects[1:]):
            idx += 1
            cur_scale = param["scale"] * last_scale
            cur_angle = param["angle"]
            offset = (param["offset"][0] * last_scale, param["offset"][1] * last_scale)
            cur_offset = (last_translate[0] + offset[0], last_translate[1] + offset[1])
            affine_transform = AffineTransform(cur_scale, cur_angle, cur_offset)
            transforms.append(affine_transform.GetMatrix())
            last_scale, last_translate = cur_scale, cur_offset
        
        self.transforms = deepcopy(transforms)

        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf 
        masks, imgs = [], []
        for index, transform in enumerate(transforms):
            mask = self.masks[index]
            img = self.imgs[index]
            masks.append(mask)
            imgs.append(img)

            corners = np.array([[
                [0, 0],
                [mask.shape[1], 0],
                [0, mask.shape[0]],
                [mask.shape[1], mask.shape[0]]]
            ])
            corners = cv2.transform(corners, transform)[0]
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

        width0, height0 = max_x - min_x, max_y - min_y
        image_offset = [-min_x if min_x < 0 else 0, -min_y if min_y < 0 else 0]

        imgs_, masks_, xs, ys = [], [], [], []
        for idx, (img, mask) in enumerate(zip(imgs, masks)):
            transforms[idx][0, 2] += image_offset[0]
            transforms[idx][1, 2] += image_offset[1]
            self.transforms[idx][0, 2] += image_offset[0]
            self.transforms[idx][1, 2] += image_offset[1]
            corners_origin = np.array([[
                [0, 0],
                [mask.shape[1], 0],
                [0, mask.shape[0]],
                [mask.shape[1], mask.shape[0]]]
            ])
            corners = cv2.transform(corners_origin, transforms[idx])[0]
            max_x, max_y = np.max(corners, axis=0)
            min_x, min_y = np.min(corners, axis=0)
            width, height = max_x - min_x, max_y - min_y
            xs.append((min_x, max_x))
            ys.append((min_y, max_y))
            
            # 在没有平移的情况下进行变换
            transforms[idx][0, 2], transforms[idx][1, 2] = 0, 0
            corners = cv2.transform(corners_origin, transforms[idx])[0]
            sub_min_x, sub_min_y = np.min(corners, axis=0)
            sub_image_offset = [-sub_min_x if sub_min_x < 0 else 0, -sub_min_y if sub_min_y < 0 else 0]
            transforms[idx][0, 2] += sub_image_offset[0]
            transforms[idx][1, 2] += sub_image_offset[1]
            img = cv2.warpAffine(img, transforms[idx], (width, height))
            mask = cv2.warpAffine(mask, transforms[idx], (width, height))
            imgs_.append(img)
            masks_.append(mask)
        return imgs_, masks_, (width0, height0), (xs, ys)
    
    def Visual(self, affine_collects):
        imgs_, masks_, (width, height), (xs, ys) = self.WarpImgMask(affine_collects)
        coarse_mask = np.zeros((height, width, 3), dtype=np.uint8)
        coarse_img = np.zeros((height, width, 3), dtype=np.uint8)
        fusion_masks = np.zeros((height, width), dtype=np.uint8)
        for idx, (img, mask) in enumerate(zip(imgs_, masks_)):
            fusion_mask = np.zeros((height, width), dtype=np.uint8)
            if idx > 1:
                interect = self.GetIntercet(idx-1, idx, self.transforms)
                fusion_mask = self.BuildFusionMask(interect, coarse_img)
                fusion_masks += fusion_mask
            coarse_img = self.FusionImage(xs[idx], ys[idx], coarse_img, img, fusion_mask, "cover")
            coarse_mask[ys[idx][0]: ys[idx][1], xs[idx][0]: xs[idx][1], :] += mask
        return coarse_img, coarse_mask

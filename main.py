from dataset import Dataset
from linemanager import LineManager, CoarseStitch
from visualization import Visualization
from regression import Regression
import cv2

if __name__ == "__main__":
    image_dir = r"D:\WorkSpace\BladeStitch\data\3_blade_1_15\Blade_2\1_0"
    mask_dir = r"D:\WorkSpace\BladeStitch\data\3_blade_1_15\Blade_2\1_0\mask"

    line_manager = LineManager()
    coarse_stitcher = CoarseStitch()
    visualizer = Visualization()
    dataset = Dataset(image_dir, mask_dir, is_reverse=True)
    for img, mask, img_path in dataset:
        line_manager.Push(img, mask)
        coarse_stitcher.Push(img, mask)
        visualizer.Push(img, mask)
        coarse_stitcher.PushImagePath(img_path)
    
    line_manager.EdgesDetect(reference_angle=35)
    init_scales = [0.977] * 25
    affine_collects = line_manager.AlignImages(init_scales)
   

    regressor = Regression(affine_collects)
    for index in range(len(dataset)-1):
        idx1 = index
        idx2 = index + 1
        img1, img2 = cv2.imread(dataset.images_path[idx1]),  cv2.imread(dataset.images_path[idx2])
        regressor.SetPair(idx1, idx2, img1, img2, line_manager.masks[idx1], line_manager.masks[idx2])
        angle, scale, offsetx, offsety = regressor.Optimize()
        affine_collects[idx2] = {"angle": angle, "scale": scale, "offset": (offsetx, offsety)}
    coarse_img, coarse_mask =  visualizer.Visual(affine_collects[0:5])
    cv2.imwrite("coarse_img.jpg", coarse_img)
    cv2.imwrite("coarse_mask.jpg", coarse_mask)
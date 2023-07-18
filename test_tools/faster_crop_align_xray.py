import numpy as np
import cv2
from .warp_for_xray import (
    estimiate_batch_transform,
    estimiate_batch_transform_selfblend,
    transform_landmarks,
    std_points_256,
)


class FasterCropAlignXRay:
    """
    修正到统一坐标系，统一图像大小到标准尺寸
    """

    def __init__(self, size=256, return_ldm5=False):
        self.image_size = size
        self.std_points = std_points_256 * size / 256.0
        self.return_ldm5 = return_ldm5

    def __call__(self, landmarks, images=None, jitter=False):
        landmarks = [landmark[:4] for landmark in landmarks]
        ori_boxes = np.array([ori_box for _, _, _, ori_box in landmarks]) # big box
        # print(ori_boxes)
        five_landmarks = np.array([ldm5 for _, ldm5, _, _ in landmarks]) # 相对于big box的5个点
        landmarks68 = np.array([ldm68 for _, _, ldm68, _ in landmarks]) # 相对于big box的68个点
        # assert landmarks68.min() > 0

        left_top = ori_boxes[:, :2].min(0) # 所有帧的左上角最小值

        right_bottom = ori_boxes[:, 2:].max(0) # 所有帧的右下角最大值
        
        size = right_bottom - left_top
        # print(ori_boxes,size)
        w, h = size

        diff = ori_boxes[:, :2] - left_top[None, ...] # 每一帧各自的原点-新的原点

        new_five_landmarks = five_landmarks + diff[:, None, :] # 针对新的原点的landmark
        new_landmarks68 = landmarks68 + diff[:, None, :]

        landmark_for_estimiate = new_five_landmarks.copy()
        if jitter:
            landmark_for_estimiate += np.random.uniform(
                -4, 4, landmark_for_estimiate.shape
            )

        tfm, trans = estimiate_batch_transform(
            landmark_for_estimiate, tgt_pts=self.std_points
        )

        transformed_landmarks68 = np.array(
            [transform_landmarks(ldm68, trans) for ldm68 in new_landmarks68]
        ) # 计算转换过后的landmark68

        transformed_landmarks5 = np.array(
            [transform_landmarks(ldm5, trans) for ldm5 in new_five_landmarks]
        ) # 计算转换过后的landmark5

        if images is not None:
            # 计算转换过后的images
            transformed_images = [
                self.process_single(tfm, image, d, h, w)
                for image, d in zip(images, diff)
            ]  # 拼接 func 的参数
            transformed_images = np.stack(transformed_images)
            if self.return_ldm5:
                return transformed_landmarks5, transformed_landmarks68, transformed_images
            else:
                return transformed_landmarks68, transformed_images
        else:
            if self.return_ldm5:
                return transformed_landmarks5, transformed_landmarks68
            else:
                return transformed_landmarks68

    def process_single(self, tfm, image, d, h, w):
        assert isinstance(image, np.ndarray)
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = d
        ih, iw, _ = image.shape
        # print(image.shape, new_image.shape)
        new_image[y : y + ih, x : x + iw] = image
        transformed_image = cv2.warpAffine(
            new_image, tfm, (self.image_size, self.image_size)
        )
        # print(transformed_image.shape)
        return transformed_image


class FasterCropAlignXRay_SelfBlend:
    """
    修正到统一坐标系，统一图像大小到标准尺寸
    """

    def __init__(self, size=256):
        self.image_size = size

    def __call__(self, source_landmarks, target_five_source_landmarks, source_images=None, jitter=False):
        source_landmarks = [landmark[:4] for landmark in source_landmarks]
        ori_boxes = np.array([ori_box for _, _, _, ori_box in source_landmarks]) # big box
        # print(ori_boxes)
        five_source_landmarks = np.array([ldm5 for _, ldm5, _, _ in source_landmarks]) # 相对于big box的5个点
        source_landmarks68 = np.array([ldm68 for _, _, ldm68, _ in source_landmarks]) # 相对于big box的68个点
        # assert source_landmarks68.min() > 0
        left_top = ori_boxes[:, :2].min(0) # 所有帧的左上角最小值
        right_bottom = ori_boxes[:, 2:].max(0) # 所有帧的右下角最大值
        size = right_bottom - left_top
        # print(ori_boxes,size)
        w, h = size
        diff = ori_boxes[:, :2] - left_top[None, ...] # 每一帧各自的原点-新的原点
        new_five_source_landmarks = five_source_landmarks + diff[:, None, :] # 针对新的原点的landmark
        new_source_landmarks68 = source_landmarks68 + diff[:, None, :]
        src_ldms = new_five_source_landmarks.copy()
        
        tgt_ldms = target_five_source_landmarks.copy()
        tfm, trans = estimiate_batch_transform_selfblend(src_ldms,tgt_ldms)
        transformed_source_landmarks68 = np.array(
            [transform_landmarks(ldm68, trans) for ldm68 in new_source_landmarks68]
        ) # 计算转换过后的landmark68

        if source_images is not None:
            # 计算转换过后的images
            transformed_images = [
                self.process_single(tfm, image, d, h, w)
                for image, d in zip(source_images, diff)
            ]  # 拼接 func 的参数
            transformed_images = np.stack(transformed_images)
            return transformed_source_landmarks68, transformed_images
        else:
            return transformed_source_landmarks68

    def process_single(self, tfm, image, d, h, w):
        assert isinstance(image, np.ndarray)
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = d
        ih, iw, _ = image.shape

        new_image[y : y + ih, x : x + iw] = image
        transformed_image = cv2.warpAffine(
            new_image, tfm, (self.image_size, self.image_size)
        )
        # print(transformed_image.shape)
        return transformed_image

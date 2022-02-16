import cv2
import numpy as np


def cyberpunk(image):
    # 反转色相
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_hls = np.asarray(image_hls, np.float32)
    hue = image_hls[:, :, 0]
    hue[hue < 90] = 180 - hue[hue < 90]
    image_hls[:, :, 0] = hue

    image_hls = np.asarray(image_hls, np.uint8)
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_lab = np.asarray(image_lab, np.float32)
    
    # 提高像素亮度，让亮的地方更亮
    light_gamma_high = np.power(image_lab[:, :, 0], 0.8)
    light_gamma_high = np.asarray(light_gamma_high / np.max(light_gamma_high) * 255, np.uint8)

     # 降低像素亮度，让暗的地方更暗
    light_gamma_low = np.power(image_lab[:, :, 0], 1.2)
    light_gamma_low = np.asarray(light_gamma_low / np.max(light_gamma_low) * 255, np.uint8)

    # 调色至偏紫
    dark_b = image_lab[:, :, 2] * (light_gamma_low / 255) * 0.1
    dark_a = image_lab[:, :, 2] * (1 - light_gamma_high / 255) * 0.3

    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_b, 0, 255)
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_a, 0, 255)

    image_lab = np.asarray(image_lab, np.uint8)
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)


if __name__ == "__main__":
    # 设置窗口可缩放
    cv2.namedWindow('origin', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('cyberpunk', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    image = cv2.imread("./city.png")
    cv2.imshow("origin", image)
    image = cyberpunk(image)
    cv2.imshow("cyberpunk", image)
    cv2.waitKey()

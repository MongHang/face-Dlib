import dlib
from math import hypot
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def rotate(image, angle, center=None, scale=1.0):
    # 取畫面寬高
    (h, w) = image.shape[:2]

    # 若中心點為無時，則中心點取影像的中心點
    if center is None:
        center = (w / 2, h / 2)

    # 產生旋轉矩陣Ｍ(第一個參數為旋轉中心，第二個參數旋轉角度，第三個參數：縮放比例)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 透過旋轉矩陣進行影像旋轉
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



# 讀取人臉辨識模型
detector = dlib.get_frontal_face_detector()

# 讀取人臉辨識之特徵模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 讀取影像（人臉與豬鼻子）
img = cv2.imread("./img/face_3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
black_glasses = cv2.imread('./img/glasses_1.png', cv2.IMREAD_UNCHANGED)
# 影像維度
(h, w, c) = img.shape

# 影像代入人臉辨識模型，需帶入RGB影像
face = detector(img)
print("人臉之位置", face)

# 轉成灰階影像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 將多個人臉帶入迴圈中
for face in face:

    # 使用灰階影像偵測臉特徵的詳細位置
    landmarks = predictor(img_gray, face)

    # 獲得鼻子上方、中間、左邊及右邊的座標位置
    top_glasses = (landmarks.part(24).x, landmarks.part(24).y)
    center_glasses = (landmarks.part(27).x, landmarks.part(27).y)
    left_glasses = (landmarks.part(36).x, landmarks.part(36).y)
    right_glasses = (landmarks.part(45).x, landmarks.part(45).y)

    # 計算角度
    glasses_angle = float(-np.arctan((right_glasses[1] - left_glasses[1]) / (right_glasses[0] - left_glasses[0])) * 180 / np.pi)
    black_glasses_rotated = rotate(black_glasses, glasses_angle)

    # 計算鼻子寬(鼻子左邊至右邊的歐式距離)，高度則是一樣（原因鼻子大小高寬相同），並定於2.2倍
    glasses_width = int(hypot(left_glasses[0] - right_glasses[0], left_glasses[1] - right_glasses[1]) * 2.2)
    glasses_height = int(glasses_width)

    # 鼻子左上與右下的位置，即正方形
    top_left_glasses = (int(center_glasses[0] - glasses_width / 2), int(center_glasses[1] - glasses_height / 2))
    bottom_right_glasses = (int(center_glasses[0] + glasses_width / 2), int(center_glasses[1] + glasses_height / 2))

    # 改變豬鼻子大小，與我鼻子同寬高
    black_glasses_resized = cv2.resize(black_glasses_rotated, (glasses_width, glasses_height))
    # plt.imshow(black_glasses_resized)
    # plt.show()

    # 豬鼻子變成灰階, 使用閥值變成二值化
    # nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
    # _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(nose_mask, cmap ='gray')
    # plt.show()

    # # 條件式判斷貼圖是否會貼超出畫面
    # if top_left[1]<0 or top_left[0]<0 or bottom_right[1]>h or bottom_right[0]>w:
    #     continue

    # # 豬鼻子預放入的區域大小之鼻子部分
    # nose_area = img[top_left[1]: top_left[1]+nose_height, top_left[0]: top_left[0]+nose_width]
    # plt.imshow(nose_area)
    # plt.show()

    # # 每個畫素值進行二進位制“&”操作，1&1=1，1&0=0，0&1=0，0&0=0，
    # nose_area_no_nose = cv2.bitwise_and(nose_area,nose_area,mask=nose_mask)
    # plt.imshow(nose_area_no_nose)
    # plt.show()

    # # 將豬鼻子與真鼻子外影像結合的矩形
    # final_nose = cv2.add(nose_area_no_nose, nose_pig)
    # plt.imshow(final_nose)
    # plt.show()

    # # 將矩形放入原來影像之矩形
    # img[top_left[1]: top_left[1]+nose_height, top_left[0]: top_left[0]+nose_width] = final_nose
    for c in range(0, 3):
        img[top_left_glasses[1]: top_left_glasses[1] + glasses_height,
        top_left_glasses[0]: top_left_glasses[0] + glasses_width, c] = \
            black_glasses_resized[:, :, c] * (black_glasses_resized[:, :, 3] / 255.0) + img[top_left_glasses[1]:
                                                                                            top_left_glasses[
                                                                                                1] + glasses_height,
                                                                                        top_left_glasses[0]:
                                                                                        top_left_glasses[
                                                                                            0] + glasses_width, c] * (
                        1.0 - black_glasses_resized[:, :, 3] / 255.0)
    plt.figure(figsize=(15, 13))
    plt.imshow(img)
    plt.show()

    plt.imsave("newface.jpg", img)
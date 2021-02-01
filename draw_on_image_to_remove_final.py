import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from scipy import ndimage
import skimage.io as skio

windowName = 'Draw on region which you want to remove pres escape to leave'

img_=skio.imread("data/grpgray.jpg")
img_org=img_

img_=img_/1.0+1
temp=(img_> 255)*1.
img_=img_-temp
img_= img_.astype(np.uint8)
img=img_

cv2.namedWindow(windowName)
drawing = False
(ix, iy) = (-1, -1)
# mouse callback function
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
cv2.setMouseCallback(windowName, draw_shape)
def main():

    mod=[[]]
    while (True):
        cv2.imshow(windowName, img)
        k = cv2.waitKey(1)

        if k == 27:
            mod= img
            break

    cv2.destroyAllWindows()

    patch=(mod[:,:,1]==0)*1
    reduction=0
    for i in range(patch.shape[0]):
        if reduction<np.sum(patch[i,:]):
            reduction=np.sum(patch[i,:])
    res_img, seam_img = column_carving(np.mean(img_org,-1), reduction,patch)
    print("\nprevious size of image ", (img_org.shape[0],img_org.shape[1]))
    print("resized image shape : ", res_img.shape)
    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.imshow(img_org, cmap='gray')
    plt.title("original image")
    plt.subplot(222)
    plt.imshow(res_img, cmap='gray')
    plt.title("objects removed")
    plt.subplot(223)
    plt.imshow(seam_img)
    plt.title("seams removed")
    plt.show()


def apply_sobel(img):
    du = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    dv = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])

    img = img.astype('float32')
    convolved = np.absolute(ndimage.filters.convolve(img, du)) + np.absolute(ndimage.filters.convolve(img, dv))
    return convolved

def min_sum_DP(e_map):
    # e_map = apply_sobel(img)
    m, n = e_map.shape
    min_energy = e_map.copy()
    find_seam_backtrack = np.zeros((m, n), dtype=np.int)

    for i in range(1, m):
        for j in range(0, n):
            if j == 0:
                temp = np.argmin(min_energy[i - 1, j:j + 2])
                find_seam_backtrack[i, j] = temp + j
                min_energy[i, j] += min_energy[i - 1, temp + j]
            else:
                temp = np.argmin(min_energy[i - 1, j - 1:j + 2])
                find_seam_backtrack[i, j] = temp + j - 1
                min_energy[i, j] += min_energy[i - 1, temp + j - 1]
    return min_energy, find_seam_backtrack

def column_carving(img, column_factor,patch):
    reduced_n = int(img.shape[1] - column_factor)
    img1 = img.copy()
    mask_red = np.zeros((img.shape[0], img.shape[1]))
    e_map = apply_sobel(img)
    e_map=e_map-10000000*patch
    temp=img1.shape[1]
    for i in trange(img.shape[1] - reduced_n):
        min_energy, find_seam_backtrack = min_sum_DP(e_map)
        min_index = np.argmin(min_energy[-1])
        kk=0
        if(min_index<=temp):
            kk=0
        else:
            kk=i
        temp = min_index
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.bool)
        for k in reversed(range(img.shape[0])):
            mask[k, min_index] = 0
            mask_red[k, min_index +kk] = 1

            min_index = find_seam_backtrack[k, min_index]
        img = img[mask].reshape((img.shape[0], img.shape[1] - 1))
        e_map = e_map[mask].reshape((e_map.shape[0], e_map.shape[1] - 1))
    mask_inv = 1 - mask_red
    img1=img1/255.0
    mod = np.zeros((img1.shape[0], img1.shape[1], 3))
    mod[:, :, 0] = img1 * mask_inv+ mask_red
    mod[:, :, 1] = img1 * mask_inv
    mod[:, :, 2] = img1 * mask_inv
    return img, mod


if __name__ == "__main__":
    main()

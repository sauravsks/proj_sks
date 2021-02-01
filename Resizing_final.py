
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import trange
from scipy import ndimage
from scipy.ndimage.interpolation import rotate

def main():
    img_ = io.imread("data/1gray.jpg",as_gray=True)
    
    
    val = input("Want to resize ( c (for columns) / r (for rows)) : ")
    if(val=='c'):
        
        column_factor = float(input("Enter reducing factor(0-1) : "))
        img1 = img_.copy()
        res_img, seam_img , seam_e_map= column_carving(img1,column_factor)
        
    elif(val=='r'):
        row_factor = float(input("Enter reducing factor(0-1) : "))
        img1 = img_.copy()
        img1 = rotate(img1,90,reshape=True)
        img1 = (-np.min(img1)+img1)/np.max(img1)
        res_img, seam_img , seam_e_map= column_carving(img1,row_factor)
        res_img = rotate(res_img,-90,reshape=True)
        res_img = (-np.min(res_img)+res_img)/np.max(res_img)
        seam_img = rotate(seam_img,-90,reshape=True)
        seam_img = (-np.min(seam_img)+seam_img)/np.max(seam_img)
        seam_e_map = rotate(seam_e_map,-90,reshape=True)
        seam_e_map = (-np.min(seam_e_map)+seam_e_map)/np.max(seam_e_map)
    else:
        print("Please enter correct value")
        return
    print("\nprevious size of image ", img_.shape)
    print("resized image shape : ",res_img.shape)
    plt.figure(figsize=(15,15))
    plt.subplot(221)
    plt.imshow(img_,cmap='gray')
    plt.title("Original Image")
    plt.subplot(222)
    plt.imshow(res_img,cmap='gray')
    plt.title("Carved Image")
    plt.subplot(223)
    plt.imshow(seam_img)
    plt.title("Seams on Image")
    plt.subplot(224)
    plt.imshow(seam_e_map)
    plt.title("seams on energy map")
    plt.show()
def apply_sobel(img):

    du = np.array([[1.0, 2.0, 1.0],[0.0, 0.0, 0.0],[-1.0, -2.0, -1.0]])

    dv = np.array([[1.0, 0.0, -1.0],[2.0, 0.0, -2.0],[1.0, 0.0, -1.0]])

    img = img.astype('float32')
    convolved = np.absolute(ndimage.filters.convolve(img,du))+ np.absolute(ndimage.filters.convolve(img,dv))
    
    return convolved

def min_sum_DP(e_map):
    m,n = e_map.shape
    min_energy = e_map.copy()
    find_seam_backtrack = np.zeros((m,n),dtype=np.int)
    
    for i in range(1,m):
        for j in range(0,n):
            if j==0:
                temp = np.argmin(min_energy[i-1,j:j+2])
                find_seam_backtrack[i,j] = temp+j
                min_energy[i,j]+=min_energy[i-1,temp+j]
            else:
                temp = np.argmin(min_energy[i-1,j-1:j+2])
                find_seam_backtrack[i,j] = temp+j-1
                min_energy[i,j]+=min_energy[i-1,temp+j-1]
    return min_energy,find_seam_backtrack

def column_carving(img,column_factor):
    reduced_n = int(img.shape[1]*column_factor)
    img1 =img.copy()
    
    mask_red = np.zeros((img.shape[0],img.shape[1]))
    e_map = apply_sobel(img)
    min_energy, find_seam_backtrack = min_sum_DP(e_map)
    seam_e_map = min_energy.copy()
    
    temp = img1.shape[1]
    
    seam_e_map = seam_e_map/np.max(seam_e_map)
    for i in trange(img.shape[1]-reduced_n):
        min_energy, find_seam_backtrack = min_sum_DP(e_map)
        min_index = np.argmin(min_energy[-1])
        kk=0
        if(min_index<=temp):
            kk=0
        else:
            kk=i
        temp = min_index
        mask = np.ones((img.shape[0],img.shape[1]),dtype=np.bool)
        for k in reversed(range(img.shape[0])):
            mask[k,min_index] = 0
            mask_red[k,min_index+kk] =1
            min_index = find_seam_backtrack[k,min_index]
        img = img[mask].reshape((img.shape[0],img.shape[1]-1))
        e_map = e_map[mask].reshape((e_map.shape[0],e_map.shape[1]-1))
    mask_inv=1-mask_red
    mod=np.zeros((img1.shape[0],img1.shape[1],3))
    mod[:,:,0]=img1*mask_inv
    mod[:,:,1]=img1*mask_inv+mask_red
    mod[:,:,2]=img1*mask_inv
    mod1=np.zeros((img1.shape[0],img1.shape[1],3))
    mod1[:,:,0]=seam_e_map*mask_inv
    mod1[:,:,1]=seam_e_map*mask_inv+mask_red
    mod1[:,:,2]=seam_e_map*mask_inv
    return img,mod,mod1
            

if __name__ == '__main__':
    main()

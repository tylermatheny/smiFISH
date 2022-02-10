#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import bigfish as bf
import bigfish.plot as plot
import bigfish.segmentation as segmentation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2


# In[2]:


path = '/Users/katievaeth/2022FEB7_smiFISH/2022_FEB7_HealthyControl/HC_CLN7cy3_NET1cy5/'
img = bf.stack.read_dv(path + 'control_CLN7_NET1_04_R3D_D3D.dv')
nuc = img[0,...]
cy3 = img[1,...]
cy5 = img[2,...]


# In[4]:


plt.imshow(np.max(nuc, axis = 0), cmap = 'Blues')
nuc_max = bf.stack.maximum_projection(nuc)
nuc_max = bf.stack.rescale(nuc_max, channel_to_stretch=0)
bf.plot.plot_images(nuc_max)


# In[12]:


flattend_nuc = nuc_max.flatten()
plt.hist(flattend_nuc, bins =100)

upper_limit = flattend_nuc.mean() + 3*flattend_nuc.std()
print(upper_limit)
nuc_thresh = nuc_max[nuc_max < upper_limit]
thresh = cv2.threshold(nuc_thresh,0, max(flattend_nuc), cv2.THRESH_BINARY + 
                       cv2.THRESH_OTSU)
print(thresh[0])
Threshold = thresh[0]
ret, otsu = cv2.threshold(nuc,0,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
#print( "{} {}".format(thresh,ret) )


# In[13]:


nuc_mask = bf.segmentation.thresholding(nuc_max, threshold= Threshold)
nuc_mask = bf.segmentation.clean_segmentation(nuc_mask, small_object_size= 2000, fill_holes=True)
nuc_label = bf.segmentation.label_instances(nuc_mask)


# In[14]:


images = [nuc_max, nuc_mask, nuc_label]
titles = ['max_proj', 'masked nuclei', 'labeled nuclei']
bf.plot.plot_segmentation(nuc_max, np.uint8(nuc_label))


# In[38]:


cy5_max = bf.stack.maximum_projection(cy5)
cy5_max = bf.stack.rescale(cy5_max, channel_to_stretch=0)

flattened_cy5 = cy5_max.flatten()
upper_limit_cy5 = flattened_cy5.mean()
lower_limit_cy5 = flattened_cy5.mean()- 
print(upper_limit_cy5)

cy5_thresh = cy5_max[cy5_max < upper_limit_cy5]
thresh = cv2.threshold(cy5_thresh,0, upper_limit_cy5, cv2.THRESH_BINARY + 
                       cv2.THRESH_OTSU)
print(thresh[0])
Threshold_cy5 = thresh[0]

ret, otsu = cv2.threshold(cy5,0,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

cell_label = bf.segmentation.cell_watershed(cy5_max, np.int64(nuc_label), threshold=Threshold_cy5, alpha=0.9)
bf.plot.plot_segmentation_boundary(cy5_max, cell_label, np.int64(nuc_label), contrast=True, boundary_size=4)


# In[51]:


cy5_max = bf.stack.maximum_projection(cy5)
cy5_max = bf.stack.rescale(cy5_max, channel_to_stretch=0)

cell_label = bf.segmentation.cell_watershed(cy5_max, np.int64(nuc_label), threshold=500, alpha=0.9)
bf.plot.plot_segmentation_boundary(cy5_max, cell_label, np.int64(nuc_label), contrast=True, boundary_size=4)



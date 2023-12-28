#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage.metrics import structural_similarity
import cv2


# In[ ]:


def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    
    keyPoints_a, descriptor_a =orb.detectAndCompute(img1,None)
    keyPoints_b, descriptor_b =orb.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    matches = bf.match(descriptor_a, descriptor_b) #perform matches 
    
    similar_regions = [i for i in matches if i.distance < 60]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches) 


# In[ ]:


def structural_sim(img1, img2):
    sim, diff = structural_similarity(img1, img2, full=True,data_range=img1.max()-img2.min())
    return sim


# In[ ]:


image0 = cv2.imread("o1.PNG",0)
image1 = cv2.imread("c1.PNG",0)


# In[ ]:


orb_similarity = orb_sim(image0, image1)


# In[ ]:


print(orb_similarity)


# In[ ]:


cv2.imshow('image1',image0)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


from skimage.transform import resize


# In[ ]:


image2 = cv2.imread("o1.PNG",0) #small size
image3 = cv2.imread("c1.PNG",0) 


# In[ ]:


image4 = cv2.resize(image1, (image0.shape[1], image0.shape[0]))


# In[ ]:


ssim = structural_sim(image0, image4)
print("similarity using SSIM is:",ssim)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


digits = load_digits()


# In[24]:

#data{ndarray, dataframe} of shape (1797, 64)
#The flattened data matrix. If as_frame=True, data will be a pandas DataFrame.

print("Image data shape",digits.data.shape) # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)


# In[25]:
#target: {ndarray, Series} of shape (1797,)
#The classification target. If as_frame=True, target will be a pandas Series.

print("Label data shape",digits.target.shape) # Print to show there are 1797 labels (integers from 0–9)


# In[26]:


plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# In[28]:




# In[29]:


# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()


# In[30]:


logisticRegr.fit(x_train, y_train)


# In[31]:


# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(x_test[0].reshape(1,-1))


# In[32]:


#Predict for Multiple Observations (images) at Once
logisticRegr.predict(x_test[0:10])


# In[33]:


#Make predictions on entire test data
predictions = logisticRegr.predict(x_test)


# In[34]:


# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)


# In[35]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[36]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[ ]:

#refered document: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html


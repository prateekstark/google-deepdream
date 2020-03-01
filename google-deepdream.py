#!/usr/bin/env python
# coding: utf-8

# In[57]:


from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image
import sys

# In[58]:


base_image_path = 'images/' + sys.argv[1]
result_prefix = 'results/' + sys.argv[2]


# In[59]:


def preprocess(fp):
    img = Image.open(fp)
    img = np.array(img)
    img = np.array([img])
    img = inception_v3.preprocess_input(img)
    return img


# In[60]:


preprocess(base_image_path).shape


# In[61]:


def deprocess(x):
    if(K.image_data_format() == 'channels_first'):
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape(x.shape[1], x.shape[2], 3)
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[62]:


K.set_learning_phase(0)


# In[63]:


from keras.models import load_model
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
dream = model.input
print('Model loaded.')


# In[64]:


# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


# In[65]:


settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}


# In[66]:


loss = K.variable(0)
for layer_name in settings['features']:
    if(layer_name not in layer_dict):
        raise ValueError('Layer' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if(K.image_data_format() == 'channels_first'):
        loss = loss + coeff * K.sum(K.square(x[:, :, 2:, -2, 2:, -2])) / scaling
    else:
        loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
    
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


# In[67]:


def eval_loss_and_grads(x):
    out = fetch_loss_and_grads([x])
    return out[0], out[1]


# In[68]:


def resize(img, size):
    img = np.copy(img)
    if(K.image_data_format() == 'channels_first'):
        factors = (1, 1, float(size[0])/img.shape[2], float(size[1])/img.shape[3])
    else:
        factors = (1, float(size[0])/img.shape[1], float(size[1])/img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


# In[69]:


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if(max_loss is not None and loss_value > max_loss):
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


# In[70]:


step = 0.01
num_octave = 3
octave_scale = 3
iterations = 20
max_loss = 10
img = preprocess(base_image_path)
if(K.image_data_format() == 'channels_first'):
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim/(octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize(img, successive_shapes[0])
for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    upscaled_shrunk_original_img = resize(original_img, shape)
    same_size_original = resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = resize(original_img, shape)

save_img(result_prefix + '.png', deprocess(np.copy(img)))


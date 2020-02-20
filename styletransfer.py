from keras.preprocessing.image import load_img,img_to_array

target_image_path = 'content2.jpg'
style_reference_image_path = 'style1.jpg'

width,height = load_img(target_image_path).size
img_height = 400
img_width = int(width*img_height/height)
img_nrows = img_height
img_ncols = img_width

import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path,target_size=(img_height,img_width))
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]#BGR---> RGB
    x = np.clip(x,0,255).astype('uint8')
    return x

from keras import backend as K

target_image = K.variable(preprocess_image(target_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))
#combination_image = K.placeholder((1,img_height,img_width,3))
input_tensor = K.concatenate([target_image,style_reference_image,combination_image],axis=0)
model = vgg19.VGG19(input_tensor = input_tensor,weights='imagenet',include_top=False)

#定义内容loss
def content_loss(base,combination):
    return K.sum(K.square(combination-base))

#定义style loss
def gram_matrix(x):
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    #features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
    gram = K.dot(features,K.transpose(features))
    return gram

def style_loss(style,combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height*img_width
    return K.sum(K.square(C-S))/(4.*(channels**2)*(size**2))

#total_variation_loss总变异损失 paper里未提及
def total_variation_loss(x):
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#total_loss
outputs_dict = dict([(layer.name,layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

total_variation_weight = 1
style_weight = 1.
content_weight = 0.025
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_imaget_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]
loss += content_weight*content_loss(target_imaget_features,combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_reference_features,combination_features)
    loss += (style_weight/len(style_layers))*sl
    
loss += total_variation_weight*total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

#grads = K.gradients(loss, combination_image)[0]
#fetch_loss_and_grads = K.function([combination_image], [loss, grads])
fetch_loss_and_grads = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_height, img_width))
    else:
        x = x.reshape((1, img_height, img_width, 3))
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        #x = x.reshape((1, img_height, img_width, 3))
        #outs = fetch_loss_and_grads([x])
        #loss_value = outs[0]
        #grad_values = outs[1].flatten().astype('float64')
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
        
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
        
evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b
from imageio import imsave
import time 
import keras
result_prefix = 'stytran/my_result'
iterations = 20
import tensorflow as tf
x = preprocess_image(target_image_path)#目标图片路径
x = x.flatten()#展开，应用l-bfgs
#keras.backend.get_session().run(tf.global_variables_initializer())
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    #在生成图片上运行L-BFGS优化；注意传递计算损失和梯度值必须为两个不同函数作为参数
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads,maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    #img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
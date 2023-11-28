from keras.models import Model
import keras
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
import numpy as np
import h5py

def find_target_layer(model_pretrained):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model_pretrained.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

def compute_heatmap_inceptionV3(model_pretrained, image, eps=1e-8):
    # construct our gradient model by supplying (1) the inputs
    # to our pre-trained model, (2) the output of the (presumably)
    # final 4D layer in the network, and (3) the output of the
    # softmax activations from the model
    
    gradModel = Model(
        inputs=[model_pretrained.get_layer('inception_v3').inputs],
        outputs=[model_pretrained.get_layer('inception_v3').get_layer(find_target_layer(model_pretrained.get_layer('inception_v3'))).output,
            model_pretrained.get_layer('inception_v3').output])
    
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(image, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, 0]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap
  
fine_tuned_inceptionv3 = keras.models.load_model('./data/model_archive/best_iv3_model_simple(2).hdf5', compile=False)

with h5py.File('./data/2017_2019_images_pv_processed.hdf5', 'r') as hf:
        train_imarr = np.array(hf['trainval']['images_log'])
        train_pvlog = np.array(hf['trainval']['pv_log'])

for i, img in enumerate(train_imarr[500:503]):
    print(img.shape)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    print(img.shape)
    img = tf.keras.layers.UpSampling2D((2, 2))(img)
    img = tf.keras.layers.UpSampling2D((2, 2))(img)
    print(img.shape)
    img = img.numpy()
    
    img_heatmap = compute_heatmap_inceptionV3(fine_tuned_inceptionv3, img, eps=1e-8)
    
    colormap=cv2.COLORMAP_TURBO
    alpha = 0.43
    heatmap = cv2.applyColorMap(img_heatmap, colormap, 0)[:, :, 0]
    plt.imshow(heatmap)
    plt.savefig(f'./data/heatmap{i}.png')
    plt.imsave(f'./data/img{i}.png', img.reshape((img.shape[1], img.shape[2], 3)), cmap='rainbow')
    print(1)
    # plt.show()
    plt.close()

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import morphology, color
from skimage import transform, io
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop

from model.losses import bce_dice_loss, dice_coeff


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red,
	predicted lung field filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))

    min_val = gt.min()
    max_val = gt.max()
    # print('min_val = {} max_val = {}\n'.format(min_val, max_val))

    # boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    # boundary = morphology.dilation(gt, morphology.disk(1)) ^ gt

    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def load_imgs(im_names, im_shape):
    X = []
    for im_name in im_names:
        img = io.imread(im_name)
        img = transform.resize(img, im_shape, mode='constant')
        img = np.expand_dims(img, -1)
        X.append(img)

    X = np.array(X)

    X -= X.mean()
    X /= X.std()

    return X


class Unet(object):

    def __init__(self):
        # load json and create model
        json_file = open('models/model_bk.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("model_from_json() finished ...")

        # load weights into new model
        loaded_model.load_weights('models/trained_model.hdf5')
        print("Loaded model from disk")

        # evaluate loaded model on test data
        self.UNet = loaded_model
        model = loaded_model
        model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
        print("model compiled ")

    def predict(self, file_path):
        X = load_imgs([file_path], im_shape=(512, 256))
        xx_ = X[0, :, :, :]
        xx = xx_[None, ...]
        inp_shape = X[0].shape

        pred = self.UNet.predict(xx)[..., 0].reshape(inp_shape[:2])

        # Binarize masks
        pr = pred > 0.5

        pr_bin = img_as_ubyte(pr)
        pr_openned = morphology.opening(pr_bin)

        im_x_ray_original_size = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        height, width = im_x_ray_original_size.shape[:2]  # height, width  -- original image size
        ratio = float(height) / width
        new_shape = (4 * 256, int(4 * 256 * ratio))
        im_x_ray_4x = cv2.resize(im_x_ray_original_size, new_shape)
        pr_openned_4x = cv2.resize(pr_openned, new_shape)
        gt_4x = cv2.resize(img_as_ubyte(pr), new_shape)
        gt_4x = gt_4x > 0.5
        pr_openned_4x = pr_openned_4x > 0.5
        im_masked_4x = masked(im_x_ray_4x, gt_4x, pr_openned_4x, 0.5)  # img.max()=1.0 gt.max()=True pr.max()=True
        im_masked_4x = img_as_ubyte(im_masked_4x)
        io.imsave(file_path, im_masked_4x)


if __name__ == '__main__':
    file_path = "dataset_bow-legs/mask_050/!002115_2.png"
    unet = Unet()
    unet.predict(file_path)

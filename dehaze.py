import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import os


def estimate_atmospheric_light(img, w_size=15):

    size = img.shape[:2]
    k = int(0.001*np.prod(size))
    j_dark = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
    idx = np.argpartition(-j_dark.ravel(), k)[:k]
    x, y = np.hsplit(np.column_stack(np.unravel_index(idx, size)), 2)

    A = np.array([img[x, y, 0].max(), img[x, y, 1].max(), img[x, y, 2].max()])
    return A


def atm(img, mask, intensity=5):

    in_img = img.copy()

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    intensity = np.interp(intensity, (1, 10), (1.4, 0.5))
    print(intensity)
    mask = mask * intensity
    mask = mask / 255
    mask[mask == 0] = 0.01

    A = estimate_atmospheric_light(img)

    in_img = ((in_img.astype(mask.dtype) - A.astype(mask.dtype)) /
              mask) + A.astype(mask.dtype)

    return in_img


class Dehazer:

    def __init__(self):
        self.model = ort.InferenceSession('model.onnx')

    def dehaze(self, img, intensity):

        # Preprocess
        in_size = (460, 460)

        in_img = img.resize(in_size)
        in_img = np.array(in_img.getdata()).reshape(
            in_img.size[0], in_img.size[1], 3).astype(np.float32)
        in_img = in_img / 255

        arrays = []
        for array in in_img:
            arrays.append(array.transpose(1, 0))

        in_img = np.array(arrays)
        in_img = in_img.transpose(1, 0, 2)
        in_img = np.expand_dims(in_img, 0)

        # Transmission map estimation
        onnx_input = {'x': in_img}
        mask = self.model.run(None, onnx_input)[0]

        mask = mask[0][0] * 255
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, img.size)

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Image dehazing
        dehazed_img = atm(img, mask, intensity=intensity)
        cv2.imwrite('tmp.png', dehazed_img)
        dehazed_img = cv2.imread('tmp.png')
        os.remove('tmp.png')
        dehazed_img = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB)
        dehazed_img = Image.fromarray(dehazed_img)

        return dehazed_img


# if __name__ == "__main__":

    # # device = 'cuda' if cuda.is_available() else 'cpu'
    # device = 'cpu'
    # model = load('model.pth', map_location=device)
    # model = model.eval()

    # img_path = '../dataset/sots_reside/outdoor/hazy/0002.jpg'

    # in_img = Image.open(img_path)
    # out_img = dehaze(in_img, model, device)

    # out_img.save('out.png')

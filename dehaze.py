import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort


def guided_dehaze(img, mask):

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv_plane = cv2.split(yuv)
    dtype = mask.dtype

    # Pre-Processing
    mask = 255 - mask
    mask = (mask / 1.5).astype(dtype)

    yuv_plane[0] = cv2.subtract(yuv_plane[0], mask)
    yuv = cv2.merge(yuv_plane)

    final = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    v = np.mean(hsv[:, :, 2])

    # Post-Processing
    brightness = np.interp(v, (0, 255), (100, 50))
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    c = gray.std()
    contrast = np.interp(c, (0, 255), (100, 50))
    final = np.int16(final)
    final = final * (contrast/127+1) - contrast + brightness
    final = np.clip(final, 0, 255)
    final = np.uint8(final)

    return final


class Dehazer:

    def __init__(self):
        self.model = ort.InferenceSession('model.onnx')

    def dehaze(self, img):

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

        onnx_input = {'x': in_img}
        mask = self.model.run(None, onnx_input)[0]

        mask = mask[0][0] * 255
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, img.size)

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dehazed_img = guided_dehaze(img, mask)
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

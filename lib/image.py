import numpy as np
import matplotlib.pyplot as plt

def MyFunction():
    print ('My imported function')
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def inv_transform_img(input_tensor):
  """
    Args:
    input_tensor: input tensor
    Return:
    inp: original image
  """
  inp = input_tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  return inp
"""def get_images(mode='train'):
  fpattern = os.path.join(DATAPATH, f'{mode}/*/*.png')
  fpaths = sorted(glob(fpattern))
  if mode == 'test':
      fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
      fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

      images1 = np.asarray(list(map(imread, fpaths1)))
      images2 = np.asarray(list(map(imread, fpaths2)))
      images = np.concatenate([images1, images2])

  else:
      images = np.asarray(list(map(imread, fpaths)))

  images = np.asarray(images)
  return images"""
def get_images(mode='train'):
  fpattern = os.path.join(DATAPATH, f'{mode}/*.png')
  fpaths = sorted(glob(fpattern))
  images = np.asarray(list(map(imread, fpaths)))
  return images


"""def get_label():
    fpattern = os.path.join(DATAPATH, f'test/*/*.png')
    fpaths = sorted(glob(fpattern))
    fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
    fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

    nr_anomaly = len(fpaths1)
    nr_normal = len(fpaths2)
    labels = np.zeros(nr_anomaly + nr_normal, dtype=np.int32)
    labels[:nr_anomaly] = 1
    return labels"""


def get_mask():
    fpattern = os.path.join(DATAPATH, f'ground_truth/*/*.png')
    fpaths = sorted(glob(fpattern))
    masks = np.asarray(list(map(lambda fpath: resize(imread(fpath), (256, 256)), fpaths)))
    nr_anomaly = masks.shape[0]
    nr_normal = len(glob(os.path.join(DATASET_PATH, f'test/good/*.png')))

    masks[masks <= 128] = 0
    masks[masks > 128] = 255
    results = np.zeros((nr_anomaly + nr_normal,) + masks.shape[1:], dtype=masks.dtype)
    results[:nr_anomaly] = masks

    return results
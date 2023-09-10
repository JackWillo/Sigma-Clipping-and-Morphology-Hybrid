def custom_log_stretch(image, a = 0.008):
    """Custom logarithmic stretch with adjustable intensity."""
    return np.log(a * (image - np.min(image)) + 1) / np.log(a * (np.max(image) - np.min(image)) + 1)

def binary(log_img, threshold_value = 0.053):
  binary_image = (log_img > threshold_value).astype(int)
  return binary_image
import os
import numpy as np
from scipy.ndimage import label, median_filter, binary_dilation, binary_erosion
from astropy.io import fits
from scipy.signal import find_peaks

def CR_del(input_data, sigma = 3, block_size = 180, size_threshold = 20, filter_num = 40,
           max_iters = 4, dilation_structure = np.ones((5,5), bool), erosion_structure = np.ones((5,5), bool), save_path = None):

    # Input Data: This is the image, it can come in the form of a fits file, fits.gz file or just an array
    # sigma: This is for the sigma clipping. This is defaulted to 7 standard deviation
    # block_size: This is used to determine how large or small to break up the area on which the standard deviation and median are calculated.
    # size_threshold: Defaulted = 20.This is used to determine if a pixel is deemed small or large.
    ## If the pixel is small, the morphology dilation will be used to fill it. If it is large, the median filter will be applied.
    # filter_num: Defaulted at 5. This is used to change the size of the median filter
    # max_iters: This is used to break the function if the sigma mask cannot find enough CRs to add.
    # dilation_structure: This is used for the morphology dilation. It is defaulted to a 5 x 5 boolean matrx.
    # erosion_structure: This is used for the morphology erosion. It is defaulted to a 3 x 3 boolean matrix
    # save_path: The location for where the processed fits files will be downloaded to.

    def localized_sigma_clip(data, sigma, block_size):
        clipped_data = data.copy()
        mask = np.zeros_like(data, dtype=bool)
        for i in range(0, data.shape[0], block_size):
            for j in range(0, data.shape[1], block_size):
                block = data[i:i+block_size, j:j+block_size]
                median = np.nanmedian(block)
                std_dev = np.nanstd(block)
                local_mask = np.abs(block - median) > sigma * std_dev
                clipped_data[i:i+block_size, j:j+block_size][local_mask] = np.nan
                mask[i:i+block_size, j:j+block_size] = local_mask
        return clipped_data, mask

    def replace_with_neighbours(data, mask):
        output = data.copy()
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if mask[i, j]:
                    neighborhood = data[i-1:i+2, j-1:j+2]
                    neighborhood_mask = mask[i-1:i+2, j-1:j+2]

                    # Considering only non-masked neighbors
                    non_masked_neighs = neighborhood[~neighborhood_mask]

                    # If all neighbors are masked, skip this pixel
                    if len(non_masked_neighs) == 0:
                        continue

                    # Replacing the cosmic ray pixel with the median of the non-masked neighbors
                    output[i, j] = np.nanmedian(non_masked_neighs)

        return output

    def detect_skylines(data, prominence_threshold = 9):
        spectrum = np.sum(data, axis=1)
        peaks, _ = find_peaks(spectrum, prominence=prominence_threshold)
        return peaks

    # Load the data
    if isinstance(input_data, str) and (input_data.endswith(".fits") or input_data.endswith(".fits.gz")):
        with fits.open(input_data) as hdul:
            data = hdul[0].data.astype(float)
            original_header = hdul[0].header
            hdul.close()

        base_filename = os.path.splitext(os.path.basename(input_data))[0]
        if ".fits" in base_filename:
            base_filename = os.path.splitext(base_filename)[0]

        directory = os.path.dirname(input_data)

    else:
        data = np.array(input_data, dtype=float)
        original_header = None
        base_filename = None
        directory = None

    original = data.copy()
    mask_total = np.zeros_like(data, dtype=bool)

    skyline_indices = detect_skylines(data)
    skyline_mask = np.zeros_like(data, dtype=bool)
    for index in skyline_indices:
        skyline_mask[index, :] = True

    for _ in range(max_iters):
        copy_data, mask = localized_sigma_clip(data, sigma, block_size)
        mask &= ~skyline_mask  # Remove skylines from the mask
        mask_total |= mask  # Combine the masks from each iteration
        data = replace_with_neighbours(data, mask)

    # Labeling and morphology steps
    eroded_mask = binary_erosion(mask_total, erosion_structure)
    labeled_mask, num_features = label(eroded_mask)

    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        component_size = np.sum(component)

        if component_size <= size_threshold:
            dilated_component = binary_dilation(component, structure=dilation_structure)
            data[dilated_component & ~component] = data[component].mean()
        else:
           # Dynamic filter size based on component size
            dynamic_filter_size = int(np.sqrt(component_size))  # Proportional to sqrt of size
            dynamic_filter_size = max(5, dynamic_filter_size)  # Ensure it's not smaller than 5
            dynamic_filter_size = min(50, dynamic_filter_size)  # Ensure it's not larger than 50

            region_coords = np.array(np.where(component))
            min_coords = region_coords.min(axis=1)
            max_coords = region_coords.max(axis=1)
            x_min, y_min = min_coords
            x_max, y_max = max_coords
            region = data[x_min:x_max + 1, y_min:y_max + 1]
            filtered_region = median_filter(region, size=(dynamic_filter_size, dynamic_filter_size))
            data[min_coords[0]:max_coords[0] + 1, min_coords[1]: max_coords[1] + 1] = filtered_region

    # Save the outputs if save_path is provided
    if save_path:
        if original_header:
            hdu = fits.PrimaryHDU(data, header=original_header)
            hdu.writeto(save_path + base_filename + '_CR_rem.fits', overwrite=True)
            hdu_mask = fits.PrimaryHDU(mask_total.astype(int), header=original_header)
            hdu_mask.writeto(save_path + base_filename + '_mask.fits', overwrite=True)
        else:
            fits.writeto(save_path + '_output_data.fits', data, overwrite=True)
            fits.writeto(save_path + '_mask.fits', mask_total.astype(int), overwrite=True)
    '''
    x_start, x_end, y_start, y_end = 50, 100, 50, 100  # Change these to a region where you know a cosmic ray exists

    print("Original Section:\n", original[x_start:x_end, y_start:y_end])
    print("Modified Section:\n", data[x_start:x_end, y_start:y_end])
    '''
    return original, data, mask_total

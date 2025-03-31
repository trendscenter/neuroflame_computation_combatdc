import base64
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting

np.seterr(divide='ignore')

## TODO: Need to fix this based on usage
MASK = os.path.join('/computation', 'mask_4mm.nii')

def encode_png(args):
    # Begin code to serialize png images
    png_files = sorted(os.listdir(args["state"]["outputDirectory"]))

    encoded_png_files = []
    for file in png_files:
        if file.endswith('.png'):
            mrn_image = os.path.join(args["state"]["outputDirectory"], file)
            with open(mrn_image, "rb") as imageFile:
                mrn_image_str = base64.b64encode(imageFile.read())
            encoded_png_files.append(mrn_image_str)

    return dict(zip(png_files, encoded_png_files))


def print_beta_images(args, avg_beta_vector, X_labels):
    beta_df = pd.DataFrame(avg_beta_vector, columns=X_labels)

    images_folder = args["state"]["outputDirectory"]

    mask = nib.load(MASK)

    for column in beta_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = beta_df[column]

        image_string = 'beta_' + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + '.nii')

        plotting.plot_stat_map(clipped_img,
                               output_file=output_file,
                               display_mode='ortho',
                               colorbar=True)


def print_pvals(args, ps_global, ts_global, X_labels):
    p_df = pd.DataFrame(ps_global, columns=X_labels)
    t_df = pd.DataFrame(ts_global, columns=X_labels)

    # TODO manual entry, remove later
    images_folder = args["state"]["outputDirectory"]

    mask = nib.load(MASK)

    for column in p_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = -1 * np.log10(p_df[column]) * np.sign(
            t_df[column])

        image_string = 'pval_' + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + '.nii')

        #        thresholdh = max(np.abs(p_df[column]))
        plotting.plot_stat_map(clipped_img,
                               output_file=output_file,
                               display_mode='ortho',
                               colorbar=True)

def saveBin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)
import napari
import numpy as np
import nibabel as nib

nifti_path = r"BraTS20_Training_001_flair.nii"
labelmap_path = r"output\20260307_001134_3D_maxtree_auto_shreshold_segmentation\labelmap.npy"

volume = nib.load(nifti_path).get_fdata()
labelmap = np.load(labelmap_path)

viewer = napari.Viewer()

viewer.add_image(volume, name='MRI_Volume', colormap='gray', blending='additive')
viewer.add_labels(labelmap, name='Instances')

viewer.dims.ndisplay = 3

napari.run()
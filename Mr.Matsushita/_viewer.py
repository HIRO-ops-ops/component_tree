import napari
import numpy as np
import nibabel as nib

ans_path = r"Mr.Matsushita\HU_A0001_ans.npy"
labelmap_path = r"Mr.Matsushita\output\20260304_143734_load_filter1_3D_altitude\labelmap_10_0.9.npy"

volume = np.load(ans_path)
labelmap = np.load(labelmap_path)

viewer = napari.Viewer()

viewer.add_image(volume, name='CT_Volume', colormap='gray', blending='additive')
viewer.add_labels(labelmap, name='Instances')

viewer.dims.ndisplay = 3

napari.run()
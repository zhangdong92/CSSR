import pydicom

from logger import log

def get_dcm_spacing_mm(dcm_path):
    """get slice_spacing and pixel_spacing from dcm file"""
    ds = pydicom.dcmread(dcm_path)
    # slice_spacing = ds.get((0x0018, 0x0088)).value
    # pixel_spacing = ds.get((0x0028, 0x0030)).value
    slice_spacing = ds.get((0x0018, 0x0088))
    slice_mm = float(str(slice_spacing).split("'")[1])
    pixel_spacing = ds.get((0x0028, 0x0030))
    # log.info(f"Pixel Spacing: {pixel_spacing}")
    pixel_mm_array = str(pixel_spacing).split("[")[1].replace("]", "").replace(" ", " ").split(",")
    pixel_mm, pixel_mm2 = float(pixel_mm_array[0]), float(pixel_mm_array[1])
    if pixel_mm != pixel_mm2:
        log.info("exception, pixel_mm!=pixel_mm, {} {}".format(pixel_mm, pixel_mm2))
        exit(0)
    return pixel_mm, slice_mm

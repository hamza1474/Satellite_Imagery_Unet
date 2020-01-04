import zipfile
import pandas as pd
import numpy as np
import cv2
import tifffile

from config import N_SPLIT, PATCH_SIZE, CLASS_TYPE

print("===========================")
print("This script will load a large amount of data so it will take a few minutes")
print("After Script finishes you can find data_numpy.npy file in data/ directory which contains all necessary data for training")
print("===========================")

N_split = N_SPLIT
Patch_size = PATCH_SIZE
Class_Type = CLASS_TYPE

#Directory to extract and look for data
Dir = 'data/'

print("===========================")
print("EXTRACTING DATA....")
print("===========================")
with zipfile.ZipFile(Dir + 'grid_sizes.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(Dir)   

with zipfile.ZipFile(Dir + 'train_wkt_v4.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(Dir)

with zipfile.ZipFile(Dir + 'three_band.zip', 'r') as zip_ref:
    zip_ref.extractall(Dir)

print("===========================")
print("LOADING IMPORTANT FILES....")
print("===========================")
train_wkt = pd.read_csv(Dir + 'train_wkt_v4.csv')
grid_size = pd.read_csv(Dir + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
submission = pd.read_csv(Dir + 'sample_submission.csv')
Image_ID = sorted(train_wkt.ImageId.unique())
ClassName = ['Building','Misc','Road','Track','Trees','Crops','Waterway','Standing water',' Vehicle Large','Vehicle Small']


def Get_Image(image_id, Scale_Size=Patch_size*N_split):
    
    filename = os.path.join(Dir,'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)   
    img = cv2.resize(img,(Scale_Size,Scale_Size))
    
    return img


def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=grid_size, wkt_list_pandas=train_wkt):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    
    return mask


def get_patch(img_id,pos = 1):
    N_patch = N_split**2
    x_all = []
    img = Get_Image(img_id)
    mask = generate_mask_for_image_and_class((img.shape[0], img.shape[1]),img_id, Class_Type)
    for i in range(N_split):
        for j in range(N_split):   
            y = mask[Patch_size*i:Patch_size*(i + 1), Patch_size*j:Patch_size*(j + 1)]
            if ((pos == 1) and (np.sum(y) > 0)) or (pos == 0) :
                x = img[Patch_size*i:Patch_size*(i + 1), Patch_size*j:Patch_size*(j + 1),:]
                x_all.append(np.concatenate((x,y[:,:,None]),axis = 2))
    return x_all


def get_all_patches():
    array = np.zeros((1, 224, 224, 4))
    for i in range(len(Image_ID)):
        if(i==0):
            continue
    raw_patch = get_patch(Image_ID[i])
    if len(raw_patch)>0:
        for patch in raw_patch:
            if patch.shape == (224,224,4):
                patch_con = np.expand_dims(patch, axis=0)
                array = np.concatenate((array, patch_con),axis=0)
                array = np.delete(array, 0)
    return array

print("===========================")
print("LOADING DATA...")
print("===========================")
array = load_data.get_all_patches()
print("===========================")
print("SAVING DATA...")
print("===========================")
np.save('data/data_numpy.npy', array)
print("===========================")
print("Successfully saved data_numpy.npy in data/ directory")
print("===========================")
del(array)
print("===========================")
print("Exit")
print("===========================")







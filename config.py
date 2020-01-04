# Define global variable

N_SPLIT = 15
PATCH_SIZE = 224
CLASS_TYPE = 1
SEED = 40

def post_normalize_image(img, mean=0.229798,std =0.097015):
    img = (img - mean)/std
    return img
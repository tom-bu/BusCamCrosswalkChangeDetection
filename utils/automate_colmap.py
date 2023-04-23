import os
import shutil

class Argument():
    def __init__(self, image_path, mask_path, reconstruction_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.reconstruction_path = reconstruction_path



def sfm(image_path, maskpath, reconstruction_path):
    args = Argument(image_path, maskpath, reconstruction_path)

    os.makedirs(args.reconstruction_path, exist_ok=True) 
    sparse = args.reconstruction_path
    img = args.image_path
    mask = args.mask_path
    
    #extract sift features etc.
    cmd = 'colmap feature_extractor --database_path {}/db.db --image_path {} --ImageReader.mask_path {} --ImageReader.single_camera_per_folder 1'.format(sparse, img, mask)
    os.system(cmd)
    
    #find matches between points
    cmd = 'colmap exhaustive_matcher --database_path {}/db.db'.format(sparse)
    os.system(cmd)

    #sparse reconstruction
    cmd = 'colmap mapper --database_path {}/db.db --image_path {} --output_path {}'. format(sparse, img,  sparse + '/')
    os.system(cmd)

    #find the model with the most images registered. Not always necessary
    models = os.listdir(sparse)
    models = [x for x in models if 'db' not in x]
    max_model = '0'
    max_img_cnt = 0
    for model in models:
            cmd = 'colmap model_converter --input_path {} --output_path {} --output_type TXT'.format(os.path.join(sparse,model), os.path.join(sparse,model))    
            os.system(cmd)
            with open(os.path.join(sparse,model,'images.txt')) as f:
                lines = f.read().splitlines()
            curr_img_cnt = int(lines[3].split(',')[0][19:])
                    
            if curr_img_cnt > max_img_cnt:
                max_model = model
                max_img_cnt = curr_img_cnt
    os.rename(os.path.join(sparse, model, 'cameras.txt'), os.path.join(sparse, 'cameras.txt'))
    os.rename(os.path.join(sparse, model, 'images.txt'), os.path.join(sparse, 'images.txt'))
    os.rename(os.path.join(sparse, model, 'points3D.txt'), os.path.join(sparse, 'points3D.txt'))
    os.rename(os.path.join(sparse, model, 'project.ini'), os.path.join(sparse, 'project.ini'))
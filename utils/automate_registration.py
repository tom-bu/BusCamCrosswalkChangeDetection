import os
import shutil

def sfm(image_path, mask_path, query_sparse, offline_sparse):

    os.makedirs(query_sparse, exist_ok=True) 
    query_sparse = query_sparse


    shutil.copy(os.path.join(offline_sparse.replace('/plane_segmentation', ''), 'db.db'), os.path.join(query_sparse, 'db.db'))    

    #custom
    #extract sift features etc.
    cmd = 'colmap feature_extractor --database_path {}/db.db --image_path {} --ImageReader.mask_path {} --ImageReader.single_camera_per_folder 1'.format(query_sparse, image_path, mask_path)
    os.system(cmd)
    
    #find matches between points
    cmd = 'colmap exhaustive_matcher --database_path {}/db.db'.format(query_sparse)
    os.system(cmd)

    #sparse reconstruction
    cmd = 'colmap image_registrator --database_path {}/db.db --input_path {} --output_path {}'.format(query_sparse, offline_sparse, query_sparse)
    os.system(cmd)


    cmd = 'colmap model_converter --input_path {} --output_path {} --output_type TXT'.format(query_sparse, query_sparse)
    os.system(cmd)


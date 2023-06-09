## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/shape_predictor_68_face_landmarks.dat'

## Dirs for output files
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = './output'

## Keywords
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'

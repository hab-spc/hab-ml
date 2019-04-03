""" """
# Standard dist imports
from collections import OrderedDict

# Third party imports

# Project level imports

# Module level constants

def read_image():
    pass

def center_crop():
    pass

def verify_dates():
    import glob
    import os

    data_dict = {}
    for data in ['good_proro', 'bad_proro']:
        img = {}
        images = glob.glob (
            '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/images_orig/{}/*'.format(data))
        img['images'] = sorted([os.path.basename (i).replace ('jpg', 'tif') for i in images])
        img['times'] = sorted([i.split('-')[1] for i in img['images']])
        data_dict[data] = img
    data_dict['bad_proro']['spc2'] = [i.split('-')[1] for i in data_dict['bad_proro']['images'] if 'SPC2' in i]
    data_dict['bad_proro']['spcp2'] = [i.split ('-')[1] for i in data_dict['bad_proro']['images'] if 'SPCP2' in i]

def save_predictions(img_paths, gtruth, predictions, probs, label_file, output_fn):
    # df = pd.DataFrame({
    #     'image':img_paths,
    #     'gtruth':gtruth,
    #     'predictions':predictions,
    #     'confidence_level': calculate_confidence_lvl(probs)
    # })
    data = OrderedDict()
    data['image'] = img_paths
    data['gtruth'] = gtruth
    data['predictions'] = predictions
    data['confidence_level'] = calculate_confidence_lvl(probs)
    df = pd.DataFrame(data)
    df = map_labels(df, label_file, 'predictions')
    try:
        df.to_csv(output_fn)
    except:
        print('{} does not exist'.format(output_fn))
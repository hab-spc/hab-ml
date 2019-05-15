""" """
# Standard dist imports
from collections import OrderedDict
import logging

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

def train_val_split(df, label_col='label', partition=0.15, seed=42,
                    logger=None):
    from sklearn.model_selection import train_test_split

    logger = logger if logger else logging.getLogger('train-val-split')
    logger.debug('Training validation split (partition={})'.format(partition))

    train, val, _, _ = train_test_split(df, df[label_col],
                                        test_size=partition, random_state=seed)
    logger.info('Training size: {} | Validation size: {}\n'.format(
        train.shape[0], val.shape[0]))
    logger.debug('Training distribution')
    logger.debug(train[label_col].value_counts())
    logger.debug('Validation distribution')
    logger.debug(val[label_col].value_counts())

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    return train, val

def preprocess_dataframe(df, logger=None, proro=False, enable_uniform=False):
    logger = logger if logger else logging.getLogger('preprocess-df')

    if proro:
        df_ = df.copy()
        logger.debug(df_.head())
        logger.debug('Size: {}\n'.format(df_.shape))

        df_ = filter_classes(df_)

        if enable_uniform:
            df_ = sample_uniformly(df_)

        return df_

def filter_classes(df, logger=None):
    logger = logger if logger else logging.getLogger('filter-cls')
    logger.debug('Filtering classes')

    curr_size = df.shape
    df = df[(df['label'] == 'good_proro') |
              (df['label'] == 'bad_proro')].reset_index(drop=True)
    filtered_size = df.shape
    logger.debug('Filtered for binary prorocentrum classes. {} -> {} ({'
                 '})\n'.format(curr_size[0], filtered_size[0],
                             curr_size[0] - filtered_size[0]))
    return df


def sample_uniformly(df, label_col='label', sample_avg=False, seed=42,
                     logger=None):
    logger = logger if logger else logging.getLogger('sample-uniformly')
    logger.debug('Sampling uniformly')

    img_counts = df[label_col].value_counts().to_dict()
    logger.debug('Img Counts: {}'.format(img_counts))

    sample_size = 0
    if sample_avg:
        sample_size = int(df[label_col].value_counts().mean())
    else:
        sample_size = int(df[label_col].value_counts().min())
    logger.debug('Sample size (sample_avg={}): {}'.format(sample_avg, sample_size))

    sampled_classes = {k: v for k,v in img_counts.items() if v > sample_size}
    for cls in sampled_classes:
        n = sampled_classes[cls]-sample_size
        temp = df[df[label_col] == cls].sample(n=n, random_state=seed)
        df = df.drop(temp.index)

        cls_size = df[df[label_col] == cls].shape[0]
        logger.debug('Cls {}: removed {} samples | size: {}'.
                     format(cls, n, cls_size))

    final_img_counts = df[label_col].value_counts().to_dict()
    logger.debug('Final distribution: {}'.format(final_img_counts))
    logger.debug('\n')
    return df

def clean_up(data, check_exist=False):
    """Clean up dataframe before loading"""
    assert 'image_url' in data.columns.values
    import os
    df = data.copy()
    image_dir = '/data6/lekevin/hab-spc/phytoplankton-db/field_2017' #HACKEY
    df['images'] = df['image_url'].apply(
        lambda x: os.path.join(image_dir, os.path.basename(x) + '.jpg'))
    #TODO include flag to check if image file is readable
    if check_exist:
        df['exists'] = df['image'].map(os.path.isfile)
        df = df[df['exists'] == True].reset_index(drop=True)
    # TEMPORARY
    #TODO get label from the SPICI 'user_labels'
    df['label'] = 1 # unknown labels
    return df






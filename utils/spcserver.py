""" Script to download and upload images to spc.ucsd.edu

With this script you can download images from the spc.ucsd.edu website provided by
the configuration settings set in the text file. You could also upload images with
machine labeled annotations.

Author: Kevin Le
contact: kevin.le@gmail.com

"""
from __future__ import print_function, division

import sys
import json
import os
import numpy as np
import datetime
from lxml import html
import urllib
import urllib2
import cookielib
import csv
import pandas as pd
import argparse

CAMERAS = ['SPC2' , 'SPCP2', 'SPC-BIG']
IMG_PARAM = ['image_filename', 'image_id', 'user_labels', 'image_timestamp', 'tags']

def parse_cmds():
    parser = argparse.ArgumentParser(description='Accessing spc.ucsd.edu pipeline')
    parser.add_argument('--search-param-file', default=None, help='spc.ucsd.edu search param path')
    parser.add_argument('--image-output-path', default=None, help='Downloaded images output path')
    parser.add_argument('--meta-output-path', default=None, help='Meta data output path')
    parser.add_argument('-d', '--download', default=False, help='Download flagging option')
    args = parser.parse_args(sys.argv[1:])
    return args



def validate_arguments(args):
    def fatal_error(msg):
        sys.stderr.write('%s\n' % msg)
        exit(-1)

    if (args.search_param_file is None):
        fatal_error("No search param file provided")
    if (args.search_param_file is not None) and ((args.meta_output_path is None) or (args.image_output_path is None)):
        fatal_error("No meta/image output path provided")
    if (args.search_param_file is not None) and (args.download == False) and (args.image_output_path is not None):
        fatal_error("Download option not flagged")
    if (args.image_output_path is None) and (args.download == True):
        fatal_error('No output image path provided.')



def main(args):
    from spcserver import SPCServer

    print("Downloading images...")
    spc = SPCServer()
    spc.retrieve (textfile=args.search_param_file,
                  output_dir=args.image_output_path,
                  output_csv_filename=args.meta_output_path,
                  download=args.download)



if __name__ == '__main__':
    main(parse_cmds())



class SPCServer(object):
    """ Represents a wrapper class for accessing the spc.ucsd.edu pipeline

    A 'SPCServer' can be used to represent as an input/output pipeline,
    where it could be a collection of elements as inputs or a collection
    of images as outputs from the website.

    """

    def __init__(self):
        """ Creates a new SPCServer.

        """

        # Dates initialized for uploading purposes
        date = datetime.datetime.now().strftime('%s')
        self.date_submitted = str(int(date)*1000)
        self.date_started = str((int(date)*1000)-500)

        # Data for uploading images
        self.submit_dict = {"label": "",                            # 'str' DENOTING LABEL ASSOCIATED WITH IMAGES
                            "tag": "",                              # 'str' DENOTING TAG FOR FILTERING
                            "image_ids": [],                        # 'list' REPRESENTING COLLECTION OF IMAGE IDS
                            "name": "",                             # 'str' DENOTING NAME OF LABELING INSTANCE
                            "machine_name":"",                      # 'str' DENOTING MACHINE NAME
                            "started":self.date_started,            # PREFILLED START DATE
                            "submitted": self.date_submitted,       # PREFILLED SUBMISSION DATE
                            "is_machine": False,                    # 'bool' FLAGGING MACHINE OR HUMAN
                            "query_path": "",                       # MISC.
                            "notes": ""                             # MISC.
        }

        # Url links stored into list from _build_url()
        self.url_list = []

        # Camera resolution configuration
        self.cam_res = [7.38 / 1000, 0.738 / 1000]



    def upload(self, login_url, account_info, textfile, label_file):
        ''' Uploads submit dictionary to initialized url from prep_for_upload()

        Args:
            login_url: A 'str', representing the administrative link to spc.ucsd.edu
            account_info: A 'dict', containing the 'username' and 'password' to access the spc pipeline
            textfile: A 'str' representing the path to a textfile of images and ground truth/predicted labels (machine or human)

        Usage:
            ==> account_info = {'username':'kevin', 'password': 'plankt0n'}
            ==> login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
            ==> predictions = 'examples/prorocentrum/predictions.txt'
            ==> label_file = 'examples/prorocentrum/labels.txt'
            ==> spc.upload(login_url=login_url,
                   account_info=account_info,
                   textfile=predictions,
                   label_file=label_file)
        :return:
        '''
        # Access server pipeline using account credentials
        self.label_file = label_file

        self._access_server(login_url=login_url, account_info=account_info)

        if not os.path.exists(textfile):
            raise ValueError("{} does not exist".format(textfile))

        # Gather images and labels
        grouped_labels = self._read_text_file(textfile=textfile)

        # Check for valid configuration for uploading
        self._assert_submission_dict()

        # Image limit for uploading images in a session
        maximum_images = 15000

        # Loop over number of labels
        total_labels = 0
        for label in self.labels:

            # Submit label
            self.submit_dict['label'] = label

            # Group images based on labels
            image_ids = grouped_labels.get_group(label)['image'].tolist()

            # Number of images uploaded
            image_size = len(image_ids)

            # Check for necessary batching to avoid upload limit
            if image_size > maximum_images:
                for i in range(0, image_size, maximum_images):
                    batch = image_ids[i:i+maximum_images]
                    self.submit_dict['image_ids'] = batch
                    self._push_submission_dict(label=label)

            else:
                self.submit_dict['image_ids'] = image_ids
                self._push_submission_dict(label=label)

            total_labels += image_size
            print("Uploaded {} {} images".format(image_size, label))
        print("Uploaded {} total labels for {}".format(total_labels, self.labels))



    def retrieve(self, textfile, output_dir, output_csv_filename, download=False):
        """Retrieves images from url and outputs images and meta data to desired output dir and filename respectively

        Usage:
        ==> spc = SPCServer()
        ==> spc.retrieve(textfile='examples/prorocentrum/time_period.txt',
                        output_dir='examples/proroentrum/images',
                        output_csv_filename='examples/prorocentrum/meta_data.csv',
                        download=True)

        :param textfile: 'str' representing path to text file for parsing download configurations
        :param output_dir: 'str' representing path to desired output directory for downloaded images
        :param output_csv_filename: 'str' representing where to output meta csv file
        :param download: 'bool' to flag downloading option
        :return:
        """
        # Output directory
        if output_dir is not None:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Desired output filename
            self.output_dir = os.path.join (self.output_dir, '{!s}.jpg')

        # Read text file and output dir
        self._prep_for_retrieval(textfile=textfile)

        # Initialization for retrieving image urls from spc.ucsd.edu
        next_page = 'empty'

        # Source url
        self.inurl = 'http://{}.ucsd.edu{!s}'

        # Image url
        self.imgurl = 'http://{}.ucsd.edu{!s}.jpg'


        with open(output_csv_filename, 'w') as csv_file:

            # Initialize output file
            labelwriter = csv.DictWriter (csv_file, fieldnames=IMG_PARAM)
            labelwriter.writeheader ()

            # Loop over number of urls
            for i in self.data['url']:
                print('Starting download {}'.format(i))
                url_to_open = i

                if 'planktivore.ucsd.edu' in url_to_open:
                    server = 'planktivore'
                else:
                    server = 'spc'

                # Loop over number of pages per url
                while(next_page):

                    # Load json data for url
                    json_data = json.load(urllib2.urlopen(url_to_open))
                    next_page = json_data['image_data']['next']

                    # Prepare next page of images to open
                    if next_page:
                        url_to_open = self.inurl.format(server, next_page[21::])
                    else:
                        pass

                    img_dicts = json_data['image_data']['results']
                    for ii in range(len(img_dicts)):

                        # Parse for image data
                        img_url = img_dicts[ii]['image_url']
                        img = img_url.split ('/')[6] + ".jpg"
                        img_id = img_dicts[ii]['image_id']
                        img_label = [str(i) for i in img_dicts[ii]['user_labels']]
                        img_timestamp = img_dicts[ii]['image_timestamp'].encode('utf-8')
                        tags = [str(i) for i in img_dicts[ii]['tags']]

                        # Store into output csv file
                        labelwriter.writerow ({'image_filename': str (img),
                                               'image_id': str (img_id),
                                               "image_timestamp": img_timestamp,
                                               'user_labels': img_label,
                                               'tags': tags})

                        # Download images
                        if download:
                            self._download(server=server, img_url=img_url)

                next_page = 'empty'
        csv_file.close()



    @staticmethod
    def map_labels(dataframe, label_file, mapped_column):
        """ Map enumerated labels into class names

        :param dataframe: 'pandas' dataframe containing image filenames and labels
        :param label_file: 'str' representing path to label text file
        :param mapped_column: 'str' representing column of dataframe to perform mapping
        :return: dataframe with new column of the mapped class names
        """
        with open(label_file, "r") as f:
            # Map labels to class names
            mapped_labels = {int(k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}

        # Store into dataframe
        dataframe['class'] = dataframe[mapped_column].map(mapped_labels)

        return dataframe



    def _access_server(self, login_url, account_info):
        ''' Authorizes access to the server

        Usage:
            ==> account_info = {'username':'kevin', 'password': 'plankt0n'}
            ==> login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
            ==> prep_for_upload(login_url=login_url)

        :return:
        '''
        assert isinstance(account_info, dict)
        assert isinstance(login_url, str)

        cj = cookielib.CookieJar()
        self.opener = urllib2.build_opener(
            urllib2.HTTPCookieProcessor(cj),
            urllib2.HTTPHandler(debuglevel=1)
        )

        if login_url == None:
            login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'

        self.parsed_url = '/'.join(login_url.split('/')[:4])
        login_form = self.opener.open (login_url).read ()

        self.csrf_token = html.fromstring(login_form).xpath(
            '//input[@name="csrfmiddlewaretoken"]/@value')[0]

        params = json.dumps(account_info)
        req = urllib2.Request ('{}/rois/login_user'.format(self.parsed_url),
                               params, headers={'X-CSRFToken': str (self.csrf_token),
                                                'X-Requested-With': 'XMLHttpRequest',
                                                'User-agent': 'Mozilla/5.0',
                                                'Content-type': 'application/json'
                                                }
                               )
        self.resp = self.opener.open(req)
        print('Successfully logged in {}'.format(self.resp.read()))



    def _assert_submission_dict(self):
        """ Validate submission dictionary

        :return:
        """

        # Check for valid types and filled variables
        if not isinstance(self.submit_dict['image_ids'], list):
            raise TypeError("'image_ids' of 'submission_dict' must be a 'list'")
        if not isinstance(self.submit_dict['label'], str):
            raise TypeError("'label' of 'submit_dict' must be 'str'")
        if self.submit_dict['name'] == "":
            raise ValueError("'name' of 'submit_dict' must not be left empty")

        # Check for correct image id and correct if wrong
        if any(item.endswith("jpg") for item in self.submit_dict['image_ids']):
            self.submit_dict['image_ids'] = [item.replace('jpg', 'tif') for item in self.submit_dict['image_ids']]



    def _push_submission_dict(self, label):
        """ Pushes data up to spc.ucsd.edu pipeline

        :param label: 'str' representing organism label
        :return:
        """

        # Log errors with label uploads
        try:
            self.submit_json = json.dumps(self.submit_dict)
            self.req1 = urllib2.Request('{}/rois/label_images'.format(self.parsed_url),
                           self.submit_json, headers={'X-CSRFToken': str(self.csrf_token),
                                                 'X-Requested-With': 'XMLHttpRequest',
                                                 'User-agent': 'Mozilla/5.0',
                                                 'Content-type': 'application/json'
                                                 }
                           )
            self.resp1 = self.opener.open(self.req1)
        except:
            print('{} labels written to error log'.format(label))
            error_log = open ('error_log.txt', 'a')
            error_log.write('{}\n'.format(label))



    def _read_text_file(self, textfile):
        """ Parse text file containing image file names and respective labels for uploading

        :param textfile: 'str' representing path to text file of images and machine/human labels
        :return: 'pandas' group object containing images organized by their labels
        """

        try:
            # Read text file
            df = pd.read_csv(textfile, sep=',', names=['image', 'label'])
        except:
            raise IndexError("{} could not be parsed correctly. Please check formatting".format(textfile))

        # Map enumerated labels into class names
        df = self.map_labels(dataframe=df, label_file=self.label_file, mapped_column='label')

        # Labels
        self.labels = sorted(df['class'].unique())

        return df.groupby(df['class'])



    def _prep_for_retrieval(self, textfile):
        ''' Parses desired text file for url configurations and builds the url

        :param textfile: 'str' representing path to text filename to parse from.
                         Expecting items to be separated by ', ' and ordered in such fashion:
                         ['start_time', 'end_time', 'min_len', 'max_len', 'cam']
        :return:
        '''

        try:
            # Read textfile
            self.data = pd.read_csv(textfile, sep=', ', names=['start_time', 'end_time', 'min_len', 'max_len', 'cam'])
        except:
            print('{} could not be parsed correctly. Check formatting.'.format(os.path.basename(textfile)))

        #TODO ensure that min len and max len are numbers

        if not self.data.cam.isin(CAMERAS).all():
            raise ValueError('Camera specification(s) in ./{} not listed in camera choices. Options: {}'.
                             format(os.path.basename(textfile), CAMERAS))

        # Convert all at once and build url as new column
        self._build_url()



    def _build_url(self):
        """ Builds url for accessing spc.ucsd.edu pipeline for retrieving images and meta data

        :return:
        """
        def convert_date(date):
            """ Converts dates to Epoch Unix Time for Pacific West time zone

            :param date:
            :return:
            """
            import calendar
            import datetime
            import pytz

            utc_date = calendar.timegm (pytz.timezone ('America/Los_Angeles').localize (
            datetime.datetime.strptime (date, '%Y-%m-%d %H:%M:%S')).utctimetuple ())
            return (utc_date+3600)*1000

        # Convert date & time to UTC & daylight savings
        self.data.start_time = self.data.start_time.apply(convert_date)
        self.data.end_time = self.data.end_time.apply(convert_date)

        # Convert camera resolution
        min_len = self.data.min_len.astype(float)
        max_len = self.data.max_len.astype(float)

        # Initialize min/max len based off camera
        if self.data.cam.any() == 'SPC2':
            self.data.min_len = np.floor(min_len / self.cam_res[0])
            self.data.max_len = np.ceil (max_len/ self.cam_res[0])
        elif self.data.cam.any() == 'SPCP2':
            self.data.min_len = np.floor (min_len / self.cam_res[1])
            self.data.max_len = np.ceil (max_len / self.cam_res[1])
        elif self.data.cam.any() == 'SPC-BIG':
            self.data['min_len'] = 1485936000000
            self.data['max_len'] = 1488441599000

        #TODO include option for parsing labels and type of annotator

        # Build url
        pattern = "http://{}.ucsd.edu/data/rois/images/{}/{!s}/{!s}/0/24/500/{!s}/{!s}/0.05/1/noexclude/ordered/skip/any/any/any/"
        self.data['url'] = self.data.apply(
            lambda row: pattern.format('spc' if row.start_time < 1501488000000 else 'planktivore',
                                                row.cam, row.start_time, row.end_time,
                                                int(row.min_len), int(row.max_len)), axis=1)

    def _download(self, server, img_url):
        ''' Downloads image to desired output directory

        :param img_url: string parsed from json_data during retrieval
        :return:
        '''
        srcpath = self.imgurl.format(server, img_url)
        destpath = self.output_dir.format(os.path.basename(img_url))
        urllib.urlretrieve(srcpath, destpath)





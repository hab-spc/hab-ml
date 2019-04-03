"""OUTDATED"""
import json
import os
import glob
import numpy as np
import datetime
from lxml import html
import urllib
import urllib2
import cookielib
import csv
import pandas as pd

class SPCServer(object):
    def __init__(self):
        # Dates initialized for uploading purposes
        date = datetime.datetime.now().strftime('%s')
        self.date_submitted = str(int(date)*1000)
        self.date_started = str((int(date)*1000)-500)

        # Data for uploading images
        self.submit_dict = {"label": "",
                            "tag": "",
                            "image_ids": [],
                            "name": "",
                            "machine_name":"",
                            "started":self.date_started,
                            "submitted": self.date_submitted,
                            "is_machine": False,
                            "query_path": "",
                            "notes": ""
        }

        self.url_list = []
        self.cam_res = [7.38 / 1000, 0.738 / 1000]

    def prep_for_upload(self, login_url, account_info):
        '''
        Authorizes access to the server

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

    def upload(self):
        '''
        Uploads submit dictionary to initialized url from prep_for_upload()

        Usage:
            ==> spc.submit_dict['name'] = 'brian'       # NAME OF LABELING INSTANCE
            ==> spc.submit_dict['label'] = label        # STRING DENOTING LABEL ASSOCIATED WITH IMAGES
            ==> spc.submit_dict['image'] = images       # LIST OF IMAGE IDS
            ==> spc.submit_dict['is_machine'] = True    # BOOL FOR MACHINE UPLOAD. IF FALSE, USES USER LOG-IN AS ANNOTATOR NAME
            ==> spc.submit_dict['machine_name'] = mach  # NAME OF MACHINE IF IS_MACHINE = TRUE
            ==> spc.upload()
        :return:
        '''
        assert isinstance(self.submit_dict['image_ids'], list)
        assert isinstance(self.submit_dict['label'], str)
        assert self.submit_dict['name'] != ""

        self.submit_json = json.dumps(self.submit_dict)
        self.req1 = urllib2.Request('{}/rois/label_images'.format(self.parsed_url),
                       self.submit_json, headers={'X-CSRFToken': str(self.csrf_token),
                                             'X-Requested-With': 'XMLHttpRequest',
                                             'User-agent': 'Mozilla/5.0',
                                             'Content-type': 'application/json'
                                             }
                       )
        self.resp1 = self.opener.open(self.req1)

    def prep_for_retrieval(self, textfile, output_dir):
        '''
        Parses desired

        :param textfile: string of text filename to parse from. Expecting items to be separated by ', ' and ordered in such fashion:
                        ['start_time', 'end_time', 'min_len', 'max_len', 'cam']
        :param output_dir:
        :return:
        '''
        self.output_dir = output_dir
        try:
            self.data = pd.read_csv(textfile, sep=', ', names=['start_time', 'end_time', 'min_len', 'max_len', 'cam'])
        except:
            print('{} could not be parsed correctly. Check formatting.'.format(textfile))

        # Convert all at once and build url as new column
        self.build_url()

    def build_url(self):
        def convert_date(date):
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
        # if self.data.cam.any() == 'SPC2':
        #     self.data.min_len = np.floor(float(min_len) / self.cam_res[0])
        #     self.data.max_len = np.ceil (float (max_len) / self.cam_res[0])
        # elif self.data.cam.any() == 'SPCP2':
        #     self.data.min_len = np.floor (float (min_len) / self.cam_res[1])
        #     self.data.max_len = np.ceil (float (max_len) / self.cam_res[1])
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
        pattern = "http://spc.ucsd.edu/data/rois/images/{}/{!s}/{!s}/0/24/300/{!s}/{!s}/0.3/1/noexclude/ordered/skip/any/any/any/"
        self.data['url'] = self.data.apply(lambda row: pattern.format(row.cam, row.start_time, row.end_time,
                                                            int(row.min_len), int(row.max_len)), axis=1)

    def retrieve(self, output_csv_filename, download=False):
        '''
        Retrieves images from url and outputs images and meta data to desired output dir and filename respectively

        Usage:
        ==> csv_filename = 'test.csv'
        ==> spc.prep_for_retrieval(text_file, output_dir)
        ==> spc.retrieve(csv_filename)

        :param output_csv_filename: string type for where to output
        :return:
        '''
        next_page = 'empty'
        self.inurl = 'http://spc.ucsd.edu{!s}'
        self.imgurl = 'http://spc.ucsd.edu{!s}.jpg'
        self.output_dir = os.path.join (self.output_dir, '{!s}.jpg')

        with open(output_csv_filename, 'w') as csv_file:
            img_param = ['img_path', 'img', 'img_id', 'img_label', 'day']
            labelwriter = csv.DictWriter (csv_file, fieldnames=img_param)
            labelwriter.writeheader ()

            for i in self.data.url:
                print('Starting download {}'.format(i))
                url_to_open = i
                while(next_page):
                    json_data = json.load(urllib2.urlopen(url_to_open))
                    next_page = json_data['image_data']['next']

                    if next_page:
                        url_to_open = self.inurl.format(next_page[21::])
                    else:
                        pass
                    img_dicts = json_data['image_data']['results']
                    for ii in range(len(img_dicts)):
                        img_url = img_dicts[ii]['image_url']
                        img = img_url.split ('/')[6] + ".jpg"
                        img_id = img_dicts[ii]['image_id']
                        img_label = [str(i) for i in img_dicts[ii]['user_labels']]
                        img_timestamp = " ".join (img_dicts[ii]['image_timestamp'].split ()[:3])

                        labelwriter.writerow ({'img_path': str (img_url), 'img': str (img), 'img_id': str (img_id),
                                               "day": str (img_timestamp), 'img_label': str (img_label)})

                        if download:
                            self.download(img_url=img_url)
                next_page = 'empty'
        csv_file.close()

    def download(self, img_url):
        '''
        Downloads image to desired output directory
        :param img_url: string parsed from json_data during retrieval
        :return:
        '''
        srcpath = self.imgurl.format(img_url)
        destpath = self.output_dir.format(os.path.basename(img_url))
        urllib.urlretrieve(srcpath, destpath)


    def is_machine(self):


if __name__ == '__main__':
    def download_tst():
        testfile = '/data6/lekevin/phytoplankton/time_period.txt'
        output_dir = '/data6/lekevin/phytoplankton/rawdata/images/'
        output_csv = '/data6/lekevin/phytoplankton/rawdata/data.csv'
        spc = SPCServer()
        spc.prep_for_retrieval(textfile=testfile, output_dir=output_dir)
        spc.retrieve(output_csv_filename=output_csv, download=True)

    def upload_tst():
        spc = SPCServer()
        account_info = {'username': 'kevin', 'password': 'ceratium'}
        login_url = 'http://spc.ucsd.edu/data/admin/?next=/data/admin'
        spc.prep_for_upload(login_url=login_url, account_info=account_info)

        label = 'Prorocentrum'
        # df = pd.read_csv('/data6/lekevin/phytoplankton/records/will_model/version_2/test_predictions.csv')
        # images = df['image'][df['predictions'] == 1].tolist()
        images = glob.glob('/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/images_orig/good_proro/*')
        images = [os.path.basename(i).replace('jpg', 'tif') for i in images]
        spc.submit_dict['name'] = 'Prorocentrum_revisit'  # NAME OF LABELING INSTANCE
        spc.submit_dict['tag'] = 'Melissa'
        spc.submit_dict['label'] = label  # STRING DENOTING LABEL ASSOCIATED WITH IMAGES
        spc.submit_dict['image_ids'] = images  # LIST OF IMAGE IDS
        # spc.submit_dict['is_machine'] = False  # BOOL FOR MACHINE UPLOAD. IF FALSE, USES USER LOG-IN AS ANNOTATOR NAME
        # spc.submit_dict['machine_name'] = 'AlexNet_d3_proro'
        spc.upload ()

    #download_tst()
    upload_tst()




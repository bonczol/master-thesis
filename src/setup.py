import os
import configparser
import requests
import shutil
import re
import rarfile
from clint.textui import progress


def main():
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    
    ds_path = conf['DEFAULT']['root_input_dir']
    os.mkdir(ds_path)
    os.mkdir(conf['DEFAULT']['root_results_dir'])

    for section in conf.sections():
        os.makedirs(conf[section]['output_dir_wav'])
        os.makedirs(conf[section]['output_dir_label'])

        r = requests.get(conf[section]['url'], stream=True)
        fname = conf[section]['file_name']
        print(f'Downloading {fname}')

        with open(fname, 'wb') as fd:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024**2), expected_size=(total_length/1024**2) + 1): 
                fd.write(chunk)

        print(f'Unpacking {fname}')

        _, ext = os.path.splitext(fname)
        if ext == ".rar":
            with rarfile.RarFile(fname, "r") as f:
                f.extractall(ds_path)
        else:   
            shutil.unpack_archive(fname, ds_path)



if __name__ == "__main__":
    main()
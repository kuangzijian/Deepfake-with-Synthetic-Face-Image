""" process the transient attribute scence dataset """

import os
import warnings
import urllib
import urllib.request
import tarfile

url_dataset = "http://transattr.cs.brown.edu/files/aligned_images.tar"
url_datalabel = "http://transattr.cs.brown.edu/files/annotations.tar"

path_data = './data/raw/transient_attribute_scenes'
if not os.path.exists(path_data):
    os.mkdir(path_data)

path_file_dataset = './data/raw/transient_attribute_scenes/aligned_images.tar'
path_file_datalabel = './data/raw/transient_attribute_scenes/annotations.tar'


##
# assume you are running from the project base

if not os.path.exists(path_file_dataset):
    urllib.request.urlretrieve(url_dataset, path_file_dataset)

if not os.path.exists(path_file_datalabel):
    urllib.request.urlretrieve(url_datalabel, path_file_datalabel)

##
# untar datafile
with tarfile.open(path_file_dataset) as f:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f, path_data)

with tarfile.open(path_file_datalabel) as f:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f, path_data)



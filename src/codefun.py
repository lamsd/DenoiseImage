from gdown import download
import cofig as coff
from os.path import join
from os import walk, listdir
from  zipfile import ZipFile

def downloadfile(url,output):
    download(url, output, quiet=False)
    
def unzipdata(pathinput, pathout):
    with ZipFile(pathinput, 'r') as zip_ref:
        zip_ref.extractall(pathout)

def checkfiledata(pathname):
    for dirname, _ ,  filenames in walk(pathname):
        for filename in filenames:
            print(join(dirname, filename))

def sortdata(pathname):
    return sorted(listdir(pathname))

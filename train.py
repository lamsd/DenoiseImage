import cofig as coff
import src.codefun as codf
import src.trainning as cotr
import src.display as cods
from os import listdir
from os.path import join

def checkfiledowlona(pathfile):
    if pathfile in listdir("./"):
        return False
    return True

def sortfolder(pathtrain, pathtest, pathtrain_clear):
    train_img = codf.sortdata(pathtrain)
    train_cleaned_img = codf.sortdata(pathtrain_clear)
    test_img = codf.sortdata(pathtest)
    return train_img,  train_cleaned_img, test_img

def main():
    codf.unzipdata(coff.output_data, coff.pathfolderzip)
    codf.checkfiledata(coff.pathfolderzip)
    for li in listdir(coff.pathfolderzip):
        codf.unzipdata(join(coff.pathfolderzip, li), coff.pathworking)
    pathtrain = join(coff.pathworking, "train")
    pathtrain_clear = join(coff.pathworking, "train_cleaned")
    pathtest = join(coff.pathworking, "test")
    sortfolder(pathtrain, pathtest, pathtrain_clear)
    print("set up the system!")
    history, modelA = cotr.trainner(pathtrain, pathtest, pathtrain_clear )
    print("Save model for the system!!")
    modelA.save(coff.pathmodel)
    cods.dislaytrainning(history)

if __name__ == "__main__":
    if checkfiledowlona(coff.output_data):
        codf.downloadfile(coff.url_download, coff.output_data)
    main()

  
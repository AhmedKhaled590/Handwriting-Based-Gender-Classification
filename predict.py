from hinge_feature_extraction import *
from cold_feature_extraction import *
import argparse
import os
import time
from joblib import dump, load
import sys

# if __name__ == '__name__':

# def GetFeatures(input,output):
print(sys.argv[1])

# read test and out folders directories
testDir=sys.argv[1]
outDir=sys.argv[2]

# prepare feature extraction parameters
parser = argparse.ArgumentParser(description="")
# parser.add_argument("--input_folder", type=str,default=r"Gold_Hinge/InputImages")
# parser.add_argument("--output_folder", type=str,default=r"Gold_Hinge/FeaturesOutput")
parser.add_argument("--sharpness_factor", type=int, default=10)
parser.add_argument("--bordersize", type=int, default=3)
parser.add_argument("--show_images", type=bool, default=False)
parser.add_argument("--is_binary", type=bool, default=False)
opt = parser.parse_args()


# hinge_feature_vectors = []
# cold_feature_vectors = []
# labels = []
# label_names = []
# ecount = 0

cold = Cold(opt)
hinge = Hinge(opt)

imgs = os.listdir(testDir)
imgs.sort()

# print(os.path.join(outDir, "times.txt"))

# open times and results files
timeFile = open(os.path.join(outDir, "times.txt"),"w")
resultFile = open(os.path.join(outDir, "results.txt"),"w")

# dir = os.getcwd()+'/Gold_Hinge/TA-evaluation/'
dir = './'

# load data scalers
scalerHinge = load(dir+'scaler_hinge.joblib')
scalerCold = load(dir+'scaler_cold.joblib')
scalerBoth = load(dir+'scaler_both.joblib')

# load classifiers
clfHinge = load(dir+'svm_hinge.joblib')
clfCold = load(dir+'svm_cold.joblib')
clfBoth = load(dir+'svm_both.joblib')



for i in range(len(imgs)):
    # try:
    img = imgs[i]
    img_path = os.path.join(testDir, img)
    # print(img_path)

    startTime = time.time()

    # get cold and hinge features
    h_f = hinge.get_hinge_features(img_path).reshape(1,-1)
    c_f = cold.get_cold_features(img_path).reshape(1,-1)

    # print(c_f.shape)
    # print(h_f.shape)
    # print(np.concatenate((c_f,h_f),axis=1).shape)

    # get the voting of the classification
    clfCold.predict(scalerCold.transform(c_f))
    pred = clfHinge.predict(scalerHinge.transform(h_f))  + clfBoth.predict(scalerBoth.transform(np.concatenate((c_f,h_f),axis=1)))
    pred = 1 if pred>=2 else 0

    endTime = time.time() - startTime

    # print(max(.001,round(endTime,3)))


    # save the time
    if(i):timeFile.write('\n')
    timeFile.write(str(max(.001,round(endTime,3))))

    # save the prediction
    if(i):resultFile.write('\n')
    resultFile.write(str(pred))

    # hinge_feature_vectors.append(h_f)
    # cold_feature_vectors.append(c_f)

    # except Exception as inst:
    #     ecount += 1
    #     if ecount % 20 == 0:
    #         print(inst, f'error count: {ecount}')
    #     continue

# print(f'error count: {ecount}')

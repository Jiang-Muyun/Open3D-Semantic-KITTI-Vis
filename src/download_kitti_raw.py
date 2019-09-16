import os
import json
import sys
from tqdm import tqdm
import time
import argparse
import requests

kitti_list = [
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",4068],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",458643963],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",319776816],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",645940900],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip",1791264044],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip",939815328],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip",630044598],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip",1261710548],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip",1206523127],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip",469598656],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip",1115646800],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0019/2011_09_26_drive_0019_sync.zip",2097547106],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip",367093728],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0022/2011_09_26_drive_0022_sync.zip",3409516068],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip",1999129806],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip",849872514],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0028/2011_09_26_drive_0028_sync.zip",1907576425],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0029/2011_09_26_drive_0029_sync.zip",1684284933],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0032/2011_09_26_drive_0032_sync.zip",1451679572],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip",512019684],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip",3257339348],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0039/2011_09_26_drive_0039_sync.zip",1538110914],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0046/2011_09_26_drive_0046_sync.zip",486623395],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip",83650171],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0051/2011_09_26_drive_0051_sync.zip",1702327445],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0052/2011_09_26_drive_0052_sync.zip",285158280],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0056/2011_09_26_drive_0056_sync.zip",1214030003],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0057/2011_09_26_drive_0057_sync.zip",1321890349],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0059/2011_09_26_drive_0059_sync.zip",1517489971],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0060/2011_09_26_drive_0060_sync.zip",298725765],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0061/2011_09_26_drive_0061_sync.zip",3049963820],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0064/2011_09_26_drive_0064_sync.zip",2376539657],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0070/2011_09_26_drive_0070_sync.zip",1759098358],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip",439133579],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0084/2011_09_26_drive_0084_sync.zip",1574584300],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0086/2011_09_26_drive_0086_sync.zip",3055393176],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0087/2011_09_26_drive_0087_sync.zip",3138789859],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0091/2011_09_26_drive_0091_sync.zip",1419552280],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip",1720308957],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip",1097371915],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0096/2011_09_26_drive_0096_sync.zip",2016323567],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0101/2011_09_26_drive_0101_sync.zip",3617310729],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0104/2011_09_26_drive_0104_sync.zip",1310103586],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0106/2011_09_26_drive_0106_sync.zip",936875950],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip",372444404],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0117/2011_09_26_drive_0117_sync.zip",2786458483],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0119/2011_09_26_drive_0119_sync.zip",5710265],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_calib.zip",4073],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0001/2011_09_28_drive_0001_sync.zip",424412944],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0002/2011_09_28_drive_0002_sync.zip",1458258535],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0016/2011_09_28_drive_0016_sync.zip",756451122],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0021/2011_09_28_drive_0021_sync.zip",855844547],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0034/2011_09_28_drive_0034_sync.zip",193510705],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0035/2011_09_28_drive_0035_sync.zip",131723850],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip",369717988],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0038/2011_09_28_drive_0038_sync.zip",451849097],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0039/2011_09_28_drive_0039_sync.zip",1476221896],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0043/2011_09_28_drive_0043_sync.zip",604398155],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0045/2011_09_28_drive_0045_sync.zip",173725652],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0047/2011_09_28_drive_0047_sync.zip",123103730],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0053/2011_09_28_drive_0053_sync.zip",295337473],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0054/2011_09_28_drive_0054_sync.zip",195437062],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0057/2011_09_28_drive_0057_sync.zip",320437507],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0065/2011_09_28_drive_0065_sync.zip",167935773],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0066/2011_09_28_drive_0066_sync.zip",125814469],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0068/2011_09_28_drive_0068_sync.zip",290664620],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0070/2011_09_28_drive_0070_sync.zip",169295097],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0071/2011_09_28_drive_0071_sync.zip",186674086],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0075/2011_09_28_drive_0075_sync.zip",303143703],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0077/2011_09_28_drive_0077_sync.zip",182314464],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0078/2011_09_28_drive_0078_sync.zip",160697640],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0080/2011_09_28_drive_0080_sync.zip",172762287],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0082/2011_09_28_drive_0082_sync.zip",324118229],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0086/2011_09_28_drive_0086_sync.zip",130739109],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0087/2011_09_28_drive_0087_sync.zip",357182085],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0089/2011_09_28_drive_0089_sync.zip",165564431],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0090/2011_09_28_drive_0090_sync.zip",200362597],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0094/2011_09_28_drive_0094_sync.zip",372271335],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0095/2011_09_28_drive_0095_sync.zip",177817864],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0096/2011_09_28_drive_0096_sync.zip",195251631],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0098/2011_09_28_drive_0098_sync.zip",190678556],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0100/2011_09_28_drive_0100_sync.zip",323180083],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0102/2011_09_28_drive_0102_sync.zip",193048180],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0103/2011_09_28_drive_0103_sync.zip",159515450],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0104/2011_09_28_drive_0104_sync.zip",184636446],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0106/2011_09_28_drive_0106_sync.zip",314547727],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0108/2011_09_28_drive_0108_sync.zip",200917052],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0110/2011_09_28_drive_0110_sync.zip",264196469],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0113/2011_09_28_drive_0113_sync.zip",310511554],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0117/2011_09_28_drive_0117_sync.zip",150486958],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0119/2011_09_28_drive_0119_sync.zip",322630862],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0121/2011_09_28_drive_0121_sync.zip",192762200],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0122/2011_09_28_drive_0122_sync.zip",180266851],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0125/2011_09_28_drive_0125_sync.zip",247319927],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0126/2011_09_28_drive_0126_sync.zip",134052781],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0128/2011_09_28_drive_0128_sync.zip",121502858],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0132/2011_09_28_drive_0132_sync.zip",306195410],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0134/2011_09_28_drive_0134_sync.zip",230595320],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0135/2011_09_28_drive_0135_sync.zip",176161577],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0136/2011_09_28_drive_0136_sync.zip",130045777],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0138/2011_09_28_drive_0138_sync.zip",286710514],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0141/2011_09_28_drive_0141_sync.zip",297900856],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0143/2011_09_28_drive_0143_sync.zip",134208483],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0145/2011_09_28_drive_0145_sync.zip",151015865],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0146/2011_09_28_drive_0146_sync.zip",297735474],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0149/2011_09_28_drive_0149_sync.zip",192923310],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0153/2011_09_28_drive_0153_sync.zip",376787274],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0154/2011_09_28_drive_0154_sync.zip",180253120],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0155/2011_09_28_drive_0155_sync.zip",201334439],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0156/2011_09_28_drive_0156_sync.zip",125713935],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0160/2011_09_28_drive_0160_sync.zip",171702560],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0161/2011_09_28_drive_0161_sync.zip",154892601],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0162/2011_09_28_drive_0162_sync.zip",159121039],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0165/2011_09_28_drive_0165_sync.zip",347674884],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0166/2011_09_28_drive_0166_sync.zip",163378555],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0167/2011_09_28_drive_0167_sync.zip",226207751],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0168/2011_09_28_drive_0168_sync.zip",238715266],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0171/2011_09_28_drive_0171_sync.zip",117380254],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0174/2011_09_28_drive_0174_sync.zip",225884553],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0177/2011_09_28_drive_0177_sync.zip",326712111],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0179/2011_09_28_drive_0179_sync.zip",180155301],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0183/2011_09_28_drive_0183_sync.zip",163488111],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0184/2011_09_28_drive_0184_sync.zip",360343144],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0185/2011_09_28_drive_0185_sync.zip",334968556],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0186/2011_09_28_drive_0186_sync.zip",171726108],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0187/2011_09_28_drive_0187_sync.zip",230477498],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0191/2011_09_28_drive_0191_sync.zip",157634097],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0192/2011_09_28_drive_0192_sync.zip",351788149],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0195/2011_09_28_drive_0195_sync.zip",162817175],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0198/2011_09_28_drive_0198_sync.zip",263315557],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0199/2011_09_28_drive_0199_sync.zip",145873361],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0201/2011_09_28_drive_0201_sync.zip",345762778],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0204/2011_09_28_drive_0204_sync.zip",212190008],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0205/2011_09_28_drive_0205_sync.zip",145981018],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0208/2011_09_28_drive_0208_sync.zip",224367125],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0209/2011_09_28_drive_0209_sync.zip",358923876],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0214/2011_09_28_drive_0214_sync.zip",175470479],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0216/2011_09_28_drive_0216_sync.zip",246629515],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0220/2011_09_28_drive_0220_sync.zip",311130693],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0222/2011_09_28_drive_0222_sync.zip",224626212],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0225/2011_09_28_drive_0225_sync.zip",14354315],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_calib.zip",4071],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0004/2011_09_29_drive_0004_sync.zip",1406686016],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip",590851286],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_sync.zip",4337332333],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0108/2011_09_29_drive_0108_sync.zip",14140145],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip",4073],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip",1134226963],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip",11320688364],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0020/2011_09_30_drive_0020_sync.zip",4474549463],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip",4424930450],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0028/2011_09_30_drive_0028_sync.zip",21198755156],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0033/2011_09_30_drive_0033_sync.zip",6555121798],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0034/2011_09_30_drive_0034_sync.zip",5079656140],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0072/2011_09_30_drive_0072_sync.zip",5096853],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip",4075],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip",18349607637],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0034/2011_10_03_drive_0034_sync.zip",19782483045],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0042/2011_10_03_drive_0042_sync.zip",4239391956],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip",3103291675],
    ["https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0058/2011_10_03_drive_0058_sync.zip",60225505]
]

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)

def http_download(local_filename,url):
    bytes_downloaded = 0
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        t_start = time.time()
        with tqdm(total = int(r.headers['Content-Length']), smoothing=0.9) as pbar:
            with open(local_filename, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=102400): 
                    pbar.update(len(chunk))
                    bytes_downloaded += len(chunk)
                    speed = int(bytes_downloaded /(time.time() - t_start))
                    status = '  %s (%s/s)'%(sizeof_fmt(bytes_downloaded), sizeof_fmt(speed))
                    pbar.set_description(status)
                    if chunk:
                        fp.write(chunk)
        return True
    except Exception as err:
        print(err)
        return False

def download(dl_folder):
    for i,(url,length) in enumerate(kitti_list):
        fn = url.split('/')[-1]
        local_filename = os.path.join(dl_folder,fn)
        if os.path.exists(local_filename):
            file_size = os.path.getsize(local_filename)
            if file_size == length:
                print('Size Matched, skip ->', fn, sizeof_fmt(length))
            else:
                print('Size Mismatch',length, file_size)
                os.remove(local_filename)

        if not os.path.exists(local_filename):
            print('Downloading', local_filename)
            http_download(local_filename,url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('download_kitti.py')
    parser.add_argument('-d','--dst', type=str, required=True , help='folder to download kitti')
    args = parser.parse_args()
    download(args.dst)
    print('Done')
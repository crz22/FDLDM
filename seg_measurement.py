import os
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from seg_io import seg_dataset,cut_image,splice_image,save_image,read_image
from seg_network import UNet3D_RES

def save_measure(MEASURE,save_path):
    measure_name = []
    for key, _ in MEASURE.items():
        measure_name.append(key)
    with open(os.path.join(save_path, "measurement2.txt"), 'w', encoding='utf-8') as f:
        for iter in range(len(MEASURE['image_name'])):
            for name in measure_name:
                f.write(f'{name}: {MEASURE[name][iter]} ')
            f.write('\n')
        # break

def caculate_MEASURE(output,label,image_name):
    esp = 1e-5
    MEASURE['image_name'].append(image_name)
    TP = np.sum(output*label)
    FP = np.sum(output*(1-label))
    FN = np.sum((1-output)*label)

    precision = TP / (TP + FP + esp)
    recall = TP / (TP + FN + esp)
    F1 = 2 * precision * recall / (precision + recall + esp)
    IOU = TP / (TP + FP + FN + esp)
    MEASURE['F1score'].append(F1)
    MEASURE['recall'].append(recall)
    MEASURE['precision'].append(precision)
    MEASURE['iou'].append(IOU)
    print(image_name,"precision: ",precision,"recall: ",recall,"F1: ",F1,"IOU: ",IOU)
    return



if __name__ == '__main__':
    test_dataset_path = r'/home/crz/crz_short_cut/Neuron_Generation8/segment_test/result2'
    model_name = 'UNet3D_RES_FDGMACPO_LDM_TEST1_NG_Trans_ce10_dice'
    label_dir = r'/home/crz/crz_short_cut/Neuron_Generation8/dataset/BN_dataset3/label'
    # if 'NG' in model_name:
    #     label_dir = r'/home/crz/crz_short_cut/Neuron_Generation8/dataset/NG_dataset3/label'
    # # elif 'BN' in model_name:
    # else:
    #     label_dir = r'/home/crz/crz_short_cut/Neuron_Generation8/dataset/BN_dataset3/label'
    MEASURE = {}
    MEASURE['image_name'] = []
    MEASURE['F1score'] = []
    MEASURE['recall'] = []
    MEASURE['precision'] = []
    MEASURE['iou'] = []
    seg_list = os.listdir(os.path.join(test_dataset_path,model_name))
    for seg_name in seg_list:
        if seg_name[-4:] != '.tif':
            continue
        print(seg_name)
        seg = read_image(os.path.join(test_dataset_path, model_name,seg_name))
        label = read_image(os.path.join(label_dir, seg_name[4:]+'_label.tif_soma.tif'))
        seg = seg / 255.0
        label = label / 255.0
        caculate_MEASURE(seg,label,seg_name)
    save_measure(MEASURE,os.path.join(test_dataset_path,model_name))







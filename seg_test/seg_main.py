import os
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from seg_io import seg_dataset,cut_image,splice_image,save_image
from seg_network import UNet3D_RES

torch.cuda.set_device(3)

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        predict_ = predict[:,1,:,:,:]
        # print(predict_.shape)
        assert predict_.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pred = torch.sigmoid(predict_).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum()  # 利用预测值与标签相乘当作交集
        union = (pred + targ).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

class SEG_MODEL():
    def __init__(self,model_name, train_dataste_path, test_dataste_path):
        self.model_name = model_name

        self.result_dir = 'resultp/'+model_name+'/'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.checkpoint_dir = 'seg_checkpointp/'+model_name+'/'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.train_dataset = DataLoader(seg_dataset(dir_path=train_dataste_path), batch_size=64, shuffle=True, num_workers=3)
        self.test_dataset = DataLoader(seg_dataset(dir_path=test_dataste_path,mode='test'), batch_size=1, shuffle=False, num_workers=3)

        self.model = UNet3D_RES(in_channel=1,out_channel=2).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.lr_update = torch.optim.lr_scheduler.StepLR(self.opt, step_size=20, gamma=0.1)

        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device))
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.,10]).to(device))
        self.dice_criterion = DiceLoss()
        self.LOSS = {}
        self.LOSS['BCE'] = []

    def trainer(self,start_epochs = 0, end_epochs = 60,load_checkpoint = False):
        if load_checkpoint:
            print("load checkpoint")
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, "SEG_60.pkl"),map_location=lambda storage, loc: storage, weights_only=False)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=0.000001)
            self.lr_update = torch.optim.lr_scheduler.StepLR(self.opt, step_size=20, gamma=1)
            self.model.load_state_dict(checkpoint)
        self.model.train()
        iter = 0
        for epoch in range(start_epochs,end_epochs):
            print("epoch: ",epoch,"cur_lr: ",self.opt.param_groups[0]['lr'])
            for data in tqdm(self.train_dataset):
                image, label, image_name = data
                image = image.to(device)
                label = label.to(device)
                # label_one_hot = torch.cat([1-label,label],dim=1)
                # print(image.max(),image.min())
                output = self.model(image)
                # print(label_one_hot.shape,label_one_hot[0,:,0,0,0],output.shape,output[0,:,0,0,0])
                # loss = self.criterion(output, label_one_hot)
                label[label>0.5] = 1
                label_long = label.long().squeeze(1)
                # print("output: ",output.shape,output.max(),output.min())
                # print("label_long: ", label_long.shape,label_long.max(),label_long.min())
                # loss = self.criterion(output, label_long)
                loss = self.dice_criterion(output, label_long) + 5*self.criterion(output, label_long)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                iter += 1
                if iter % 100 == 0:
                    self.LOSS['BCE'].append(loss.item())
                    print("iter: ",iter,"loss: ",loss.item())
            self.lr_update.step()
        self.save_loss(load_checkpoint)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "SEG_" + str(end_epochs) + '.pkl'))

    def save_loss(self, load_checkpoint=False):
        LOSS_NAME = []
        for key, _ in self.LOSS.items():
            LOSS_NAME.append(key)
        if load_checkpoint:
            with open(os.path.join(self.checkpoint_dir, "loss.txt"), 'a', encoding='utf-8') as f:
                for iter in range(len(self.LOSS[LOSS_NAME[0]])):
                    for loss_name in LOSS_NAME:
                        f.write(f'{loss_name}: {self.LOSS[loss_name][iter]} ')
                    f.write('\n')
        else:
            with open(os.path.join(self.checkpoint_dir, "loss.txt"), 'w', encoding='utf-8') as f:
                for iter in range(len(self.LOSS[LOSS_NAME[0]])):
                    for loss_name in LOSS_NAME:
                        f.write(f'{loss_name}: {self.LOSS[loss_name][iter]} ')
                    f.write('\n')

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, "SEG_60.pkl"), map_location=lambda storage, loc: storage,weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def tester(self, n_samples = 64, load_checkpoint=False):
        if load_checkpoint:
            self.load_model()
        self.model.eval()
        self.MEASURE = {}
        self.MEASURE['image_name'] = []
        self.MEASURE['F1score'] = []
        self.MEASURE['recall'] = []
        self.MEASURE['precision'] = []
        for data in tqdm(self.test_dataset):
            image, label, image_name = data
            image_name = image_name[0]
            # label_blocks, block_num, max_num = cut_image(label, step=(32, 32, 32))
            image_blocks, block_num, max_num = cut_image(image, step=(32, 32, 32))
            output_blocks = []
            for i in range(0, block_num, n_samples):
                cur_image = image_blocks[i:min(i + n_samples, block_num)]
                cur_image = torch.cat(cur_image, dim=0).to(device)

                cur_output = self.model(cur_image)
                cur_output = torch.argmax(cur_output,dim=1,keepdim=True)
                cur_output = cur_output.unsqueeze(1).detach().cpu()
                for sample in cur_output:
                    output_blocks.append(sample)
            seg_image = splice_image(output_blocks, block_num, max_num, image_size=image.shape[2:5],step=(32, 32, 32))
            self.caculate_MEASURE(seg_image,label,image_name)
            print(image_name, seg_image.shape,seg_image.max())
            seg_image = seg_image.squeeze(0).squeeze(0).cpu().numpy()
            save_image(os.path.join(self.result_dir, "SEG_"+image_name), seg_image * 255.0)
        self.save_measure()
        # save measurement

    def save_measure(self):
        measure_name = []
        for key, _ in self.MEASURE.items():
            measure_name.append(key)
        with open(os.path.join(self.result_dir, "measurement.txt"), 'w', encoding='utf-8') as f:
            for iter in range(len(self.MEASURE['image_name'])):
                for name in measure_name:
                    f.write(f'{name}: {self.MEASURE[name][iter]} ')
                f.write('\n')
            # break

    def caculate_MEASURE(self,output,label,image_name):
        esp = 1e-5
        self.MEASURE['image_name'].append(image_name)
        TP = torch.sum(output*label)
        FP = torch.sum(output*(1-label))
        FN = torch.sum((1-output)*label)

        precision = TP / (TP + FP + esp)
        recall = TP / (TP + FN + esp)
        F1 = 2 * precision * recall / (precision + recall + esp)
        self.MEASURE['F1score'].append(F1)
        self.MEASURE['recall'].append(recall)
        self.MEASURE['precision'].append(precision)
        print(image_name,"precision: ",precision,"recall: ",recall,"F1: ",F1)
        return



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset_path = r'/home/crz/crz_short_cut/Neuron_Generation8/dataset/train_seg_p12'
    test_dataset_path = r'/home/crz/crz_short_cut/Neuron_Generation8/dataset/BN_dataset3'
    model_name = 'UNet3D_RES_seg_p12'
    MODEL = SEG_MODEL(model_name=model_name,
                      train_dataste_path=train_dataset_path,
                      test_dataste_path=test_dataset_path)

    MODEL.trainer(start_epochs=0,end_epochs=10,load_checkpoint=False)
    MODEL.tester(n_samples=64,load_checkpoint=False)







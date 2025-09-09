import sys
sys.path.append('../')
import os
import numpy as np
from tqdm import tqdm
import torch
from network import DFGM_Autoencoder,Unet_3D_CA,Discriminator,PerceptualEncoder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import label_transform,image_transform,make_beta_schedule,normalize_tensor
from io1 import mydataset, save_image,cut_image,splice_image
torch.cuda.set_device(3)

def generate_poisson(x, ref_image, X_Max = 255):
    # print(x.shape,ref_image.shape)
    x_ = F.interpolate(ref_image,x.shape[-3:])
    x_ = x_.repeat(1,x.shape[1],1,1,1)
    # print("x_: ",x_.shape)
    x_ = (x_+1)/2*X_Max
    x_ = torch.clamp(x_, 20, X_Max)
    # print('mean:',x_.max(),x_.min())
    noise = torch.poisson(x_)
    noise = (noise-x_)/10#.clip(-3,3)#/(x_+1e-5)
    # print('noise:', noise.shape, noise.max(), noise.min(), noise[:, 0, 0, 0, 0])
    return noise

class Latent_Diffusion_Model():
    def __init__(self, timesteps=1000, model_name = 'ALDM_TEST1'):
        print('Initializing Latent Diffusion Model')
        dir_path = '/home/crz/crz_short_cut/Neuron_Generation8/dataset/train_BNdataset'
        self.model_checkpoint = '/home/crz/crz_short_cut/Neuron_Generation8/My_model/checkpoint2/'+model_name
        if not os.path.exists(self.model_checkpoint):
            os.makedirs(self.model_checkpoint)
        self.result_dir = '/home/crz/crz_short_cut/Neuron_Generation8/My_model/result2/'+model_name
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.model_name = model_name

        self.dataset = DataLoader(dataset=mydataset(dir_path=dir_path), batch_size=16,shuffle=True, num_workers=3)
        # Diffusion_Model param setting
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)  # 线性噪声调度
        # self.betas = make_beta_schedule("cosine", timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        # torch.set_printoptions(precision=5, sci_mode=False)
        # print("param",self.betas,self.alphas,self.alpha_cumprod)

        self.AE = DFGM_Autoencoder(in_channels=1,out_channel=4).to(device)
        self.DN = Unet_3D_CA(in_channels=4,out_channel=4).to(device)
        self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=1e-4)
        self.DN_optimizer = torch.optim.Adam(self.DN.parameters(), lr=1e-3)
        self.AE_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.AE_optimizer, step_size=5000, gamma=0.1)
        self.DN_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.DN_optimizer, step_size=4000, gamma=0.1)

        self.DIS = Discriminator().to(device)
        self.DIS_optimizer = torch.optim.Adam(self.DIS.parameters(), lr=1e-4)
        self.DIS_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.DIS_optimizer, step_size=5000, gamma=0.1)

        self.loss_initial()
        # self.PE = PerceptualEncoder().to(device)

    def loss_initial(self):
        # set loss cash
        self.LOSS = {}
        self.LOSS['AE_dis'] = []
        self.LOSS['AE_adv'] = []
        self.LOSS['AE_rec'] = []
        self.LOSS['AE_cont'] = []
        self.LOSS['AE_idt'] = []
        self.LOSS['DN_ldm'] = []
        self.LOSS['AE_sty'] = []

    def train_Discriminator(self,real_sample,fake_sample):
        real_pred = self.DIS(real_sample)
        fake_pred = self.DIS(fake_sample.detach())

        loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
        loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        dis_loss = 0.5 * (loss_real + loss_fake)

        self.DIS_optimizer.zero_grad()
        dis_loss.backward()
        self.DIS_optimizer.step()
        return dis_loss.item()

    def train_autoencoder(self,epochs):
        self.AE.train()
        iters = 0
        rec_lamb = 10  #label,xl2i
        adv_lamb = 1
        idt_lamb = 10  #image,xi2i
        cont_lamb = 0.01
        sty_lamb = 10000
        for epoch in range(epochs):
            print('Train AE Epoch {}/{}'.format(epoch + 1, epochs),"cur_lr: ",self.AE_optimizer.param_groups[0]['lr'] )
            for batch_idx, data in tqdm(enumerate(self.dataset)):
                image,label,_,_ = data
                image = image.to(device)
                label = label.to(device)

                image = image_transform(image)
                label = label_transform(label)

                # print("raw: ",image.shape,image.max(),label.shape,label.max())
                xl2i, xi2i = self.AE(image, label)
                # print("RECON: ",recon.max(),label.max(),recon.min(),label.min())
                # train Discriminator
                dis_loss = self.train_Discriminator(image, xl2i)

                # train Autoencoder
                ### adv_loss
                fake_pred = self.DIS(xl2i)
                adv_loss = adv_lamb * F.mse_loss(fake_pred, torch.ones_like(fake_pred))

                ### rec_loss and cyc_loss
                # rec_loss = rec_lamb * (F.mse_loss(xi2i, image) + F.mse_loss(x_l2i2, xl2i))
                # cyc_loss = cyc_lamb * F.l1_loss(x_rel2i, x_rel2i)
                idt_loss = idt_lamb * F.mse_loss(xi2i, image)
                # idt_loss = idt_lamb * F.l1_loss(xi2i, image)
                rec_loss = rec_lamb * F.mse_loss(xl2i, label)

                ### cont_loss
                zcont_l2i = self.AE.get_zcont_img(xl2i)
                zcont_lab = self.AE.get_zcont_lab(label)

                _, xl2i_GM = self.AE.get_zsty_img(xl2i)
                _, img_GM = self.AE.get_zsty_img(image)
                _, xi2i_GM = self.AE.get_zsty_img(xi2i)

                cont_loss = cont_lamb * F.mse_loss(zcont_l2i, zcont_lab.detach())
                sty_loss = sty_lamb * (F.mse_loss(xl2i_GM, img_GM.detach()) + F.mse_loss(xi2i_GM, img_GM.detach()))

                # per_loss = per_lamb * self.PE(recon, image, label)
                # loss = rec_loss + adv_loss + per_loss
                loss = adv_loss + rec_loss + cont_loss + idt_loss + sty_loss
                self.AE_optimizer.zero_grad()
                loss.backward()
                # for name, param in self.AE.named_parameters():
                #     print(name,param.grad)
                self.AE_optimizer.step()
                iters += 1
                if iters % 100 == 0:
                    print(f"DIS iters [{iters}], Loss: {dis_loss:.4f}")
                    # print(f"AE iters [{iters}], Loss: {loss.item():.4f}, adv_loss: {adv_loss.item():.4f}, re_loss: {rec_loss.item():.4f}, per_loss: {per_loss.item():.4f}")
                    print(f"AE iters [{iters}], Loss: {loss.item():.4f}, adv_loss: {adv_loss.item():.4f}, re_loss: {rec_loss.item():.4f} ,"
                          f"cont_loss: {cont_loss.item():.4f}, idt_loss: {idt_loss.item():.4f}, sty_loss: {sty_loss.item():.4f}")
                    self.LOSS['AE_adv'].append(adv_loss.item())
                    self.LOSS['AE_rec'].append(rec_loss.item())
                    self.LOSS['AE_cont'].append(cont_loss.item())
                    self.LOSS['AE_idt'].append(idt_loss.item())
                    self.LOSS['AE_dis'].append(dis_loss)
                    self.LOSS['AE_sty'].append(sty_loss.item())
                self.AE_lr_scheduler.step()
                self.DIS_lr_scheduler.step()

        torch.save(self.AE.state_dict(), os.path.join(self.model_checkpoint, "AE_" + str(epochs) + '.pkl'))
        torch.save(self.DIS.state_dict(), os.path.join(self.model_checkpoint, "DIS_" + str(epochs) + '.pkl'))

    def q_sample(self, x0, t, ref_image=None, noise=None):
        if noise is None:
            # noise = torch.randn_like(x0)
            noise = generate_poisson(x0, ref_image).to(device)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1, 1)  # [B,C,IMGSIZE]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t]).view(-1, 1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def p_sample(self, x, t, sty_GM, ref_img):
        t = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        noise_pred = self.DN(x, t, sty_GM)
        # print("noise_pred: ", noise_pred.shape)
        alpha = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1, 1, 1)
        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)
        posterior_mean = (x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha)
        # print(alpha[0], sqrt_one_minus_alpha_cumprod[0], noise_pred.max())
        if t[0] > 0:
            # noise = torch.randn_like(x)
            noise = generate_poisson(x, ref_img).to(device)
            return posterior_mean + torch.sqrt(1 - alpha) * noise
        return posterior_mean

    def train_LDM(self,epochs,load_AEcheckpoint=False):
        if load_AEcheckpoint:
            checkpoint_path ='checkpoint2/FDGM_LDM_TEST6/AE_20.pkl'
            checkpoint_AE = torch.load(checkpoint_path,map_location=lambda storage, loc: storage, weights_only=False)
            self.AE.load_state_dict(checkpoint_AE)
            torch.save(self.AE.state_dict(), os.path.join(self.model_checkpoint, 'AE_20.pkl'))
        self.AE.eval()
        self.DN.train()
        iters = 0
        for epoch in range(epochs):
            print('Train LDM Epoch {}/{}'.format(epoch + 1, epochs),"cur_lr: ",self.DN_optimizer.param_groups[0]['lr'] )
            for batch_idx, data in tqdm(enumerate(self.dataset)):
                image,label,_,_ = data
                image = image.to(device)
                label = label.to(device)

                image = image_transform(image)
                label = label_transform(label)

                # print("raw: ",image.shape,image.max(),label.shape,label.max())
                t = torch.randint(0, self.timesteps, (image.size(0),), device=device)
                zl2i, sty_GM = self.AE.get_zl2i(image, label)
                z_noisy, noise = self.q_sample(zl2i, t, image)
                noise_pred = self.DN(z_noisy, t, sty_GM)
                # print("z: ",zl2i.max(),zl2i.min(),z_noisy.max(),z_noisy.min())
                # print("noise: ",noise.max(),noise.min())
                # print("noise_pred: ",noise_pred.max(),noise_pred.min())
                loss = F.mse_loss(noise_pred, noise)
                # loss = F.smooth_l1_loss(noise, noise_pred)
                # print("noise_pred: ", noise_pred.shape)
                self.DN_optimizer.zero_grad()
                loss.backward()
                self.DN_optimizer.step()
                iters += 1
                if iters % 100 == 0:
                    print(f"LDM iters [{iters}], Loss: {loss.item():.4f}")
                    self.LOSS['DN_ldm'].append(loss.item())
                self.DN_lr_scheduler.step()
        torch.save(self.DN.state_dict(), os.path.join(self.model_checkpoint, "DN_" + str(epochs) + '.pkl'))
        if load_AEcheckpoint:
            self.save_loss(only_DE = True)

    def trainer(self,AE_epochs=10,LDM_epochs=10):
        print("Training Autoencoder...")
        self.train_autoencoder(AE_epochs)
        print("Training Latent Diffusion Model...")
        self.train_LDM(LDM_epochs)
        self.save_loss()

    def save_loss(self,only_DE = False):
        # save AE loss
        AE_LOSS = []
        DN_LOSS = []
        for key, _ in self.LOSS.items():
            if 'AE' in key:
                AE_LOSS.append(key)
            elif 'DN' in key:
                DN_LOSS.append(key)
        with open(os.path.join(self.model_checkpoint,"loss.txt"), 'w', encoding='utf-8') as f:
            if not only_DE:
                for iter in range(len(self.LOSS[AE_LOSS[0]])):
                    for loss_name in AE_LOSS:
                        f.write(f'{loss_name}: {self.LOSS[loss_name][iter]} ')
                    f.write('\n')

            for iter in range(len(self.LOSS[DN_LOSS[0]])):
                for loss_name in DN_LOSS:
                    f.write(f'{loss_name}: {self.LOSS[loss_name][iter]}\n')

    def sample_ldm(self, labels, ref_images):
        self.AE.eval()
        self.DN.eval()
        latent_dim = 4
        with torch.no_grad():
            # x = torch.randn(labels.shape[0], latent_dim, 8, 8, 8).to(device)  # 从噪声开始
            x, sty_GM = self.AE.get_zl2i(ref_images, labels)
            x_min,x_max = torch.min(x), torch.max(x)
            print("LDM_IN: ",x_min,x_max)

            for t in reversed(range(self.timesteps)):
                x = self.p_sample(x.detach(), t, sty_GM, ref_images)

            print("LDM_OUT: ",x.max(),x.min())
            images = self.AE.decoder(x)
        images = images.unsqueeze(1)
        # images = (images - images.min()) / (images.max() - images.min())
        images = images.detach().cpu()#.numpy()
        # print("generated image: ",images.shape)
        # for i in range(n_samples):
        #     save_image(os.path.join(self.result_dir,"generate_img"+str(i)+".tif"), images[i]*255.0)
        return images

    def load_model(self):
        print("loading model...")
        checkpoint_AE = torch.load(os.path.join(self.model_checkpoint, "AE_20.pkl"), map_location=lambda storage, loc: storage,weights_only=False)
        self.AE.load_state_dict(checkpoint_AE)
        self.AE.eval()
        checkpoint_DN = torch.load(os.path.join(self.model_checkpoint, "DN_20.pkl"), map_location=lambda storage, loc: storage,weights_only=False)
        self.DN.load_state_dict(checkpoint_DN)
        self.DN.eval()

    def test(self, load_checkpoint=False):
        # test
        print("test ...")
        test_result_dir = '/home/crz/crz_short_cut/Neuron_Generation8/My_model/result2/' + model_name
        # test_dir = r"/home/crz/crz_short_cut/Neuron_Generation7/dataset/NG_dataset2/"
        test_dir = r'/home/crz/crz_short_cut/Neuron_Generation7/dataset/BN_dataset2/'
        test_dataset = DataLoader(dataset=mydataset(dir_path=test_dir, mode='test'), batch_size=1, shuffle=False,
                                  num_workers=3)
        if load_checkpoint:
            self.load_model()
        n_samples = 256
        for batch_idx, data in tqdm(enumerate(test_dataset)):
            image, label, image_name, label_name = data
            print(image_name, label_name)
            generate_name = "ref_" + image_name[0].split(".")[0] + "lab_" + label_name[0].split(".")[0] + '.tif'
            # label = label.to(device)
            image = F.interpolate(image, label.shape[2:5], mode='nearest')
            label_blocks, block_num, max_num = cut_image(label, step=(32, 32, 32))
            image_blocks, _, _ = cut_image(image, step=(32, 32, 32))

            label_sum = [label_blocks[i].sum() for i in range(block_num)]
            label_sum = torch.tensor(label_sum).float()
            label_sort = torch.argsort(label_sum)

            image_sum = [image_blocks[i].sum() for i in range(block_num)]
            image_sum = torch.tensor(image_sum).float()
            image_sort = torch.argsort(image_sum)

            generate_img_blocks = []
            index = []
            cur_batch = 0
            for i in range(0, block_num):
                cur_label0 = label_blocks[i]
                img_index = torch.where(image_sort == label_sort[i])[0]
                # print(img_index)
                # cur_image0 = image_blocks[img_index]
                cur_image0 = image_blocks[i]
                generate_img_blocks.append(-torch.ones_like(cur_label0))
                if cur_label0.max() == 0 and i != block_num-1:
                    continue
                index.append(i)
                cur_batch += 1
                if cur_batch == 1:
                    cur_label = cur_label0
                    cur_image = cur_image0
                else:
                    cur_label = torch.cat((cur_label,cur_label0), dim=0)
                    cur_image = torch.cat((cur_image,cur_image0), dim=0)
                # print(i,cur_label.shape)
                if cur_batch == n_samples or i == block_num-1:
                    cur_label = cur_label.to(device)
                    cur_image = cur_image.to(device)
                    # print(cur_label.shape)
                    cur_label = label_transform(cur_label)
                    cur_image = image_transform(cur_image)
                    # print(i,cur_label.shape,block_num)
                    # cur_label = cur_label.unsqueeze(1)
                    generate_img = MOEDL.sample_ldm(cur_label, cur_image)  # [batch,d,w,h]
                # generate_img = cur_label
                #     print(len(index),generate_img.shape)
                    for j in range(len(index)):
                        generate_img_blocks[index[j]] = generate_img[j]
                    index.clear()
                    cur_batch = 0
            fake_image = splice_image(generate_img_blocks, block_num, max_num, image_size=image.shape[2:5],step=(32, 32, 32))
            print(generate_name, fake_image.shape, fake_image.max(), fake_image.min())
            # fake_image = (fake_image - fake_image.min()) / (fake_image.max() - fake_image.min())
            fake_image = (fake_image / torch.abs(fake_image).max() + 1) / 2
            fake_image = fake_image.squeeze(0).squeeze(0).cpu().numpy()
            save_image(os.path.join(test_result_dir, generate_name), fake_image * 255.0)
            # break
    # def test(self,load_checkpoint=False):
    #     # test
    #     print("test ...")
    #     test_result_dir = '/home/crz/crz_short_cut/Neuron_Generation8/My_model/result/' + model_name
    #     # test_dir = r"/home/crz/crz_short_cut/Neuron_Generation7/dataset/NG_dataset2/"
    #     test_dir = r'/home/crz/crz_short_cut/Neuron_Generation7/dataset/BN_dataset2/'
    #     test_dataset = DataLoader(dataset=mydataset(dir_path=test_dir, mode='test'), batch_size=1, shuffle=False, num_workers=3)
    #     if load_checkpoint:
    #         self.load_model()
    #
    #     n_samples = 256
    #     for batch_idx, data in tqdm(enumerate(test_dataset)):
    #         image, label, image_name, label_name = data
    #         print(image_name, label_name)
    #         generate_name = "ref_"+image_name[0].split(".")[0] + "lab_" + label_name[0].split(".")[0] + '.tif'
    #         # label = label.to(device)
    #         image = F.interpolate(image, label.shape[2:5], mode='nearest')
    #         label_blocks, block_num, max_num = cut_image(label, step=(32, 32, 32))
    #         image_blocks, _, _ = cut_image(image, step=(32, 32, 32))
    #         generate_img_blocks = []
    #         for i in range(0, block_num, n_samples):
    #             cur_label = label_blocks[i:min(i + n_samples, block_num)]
    #             cur_image = image_blocks[i:min(i + n_samples, block_num)]
    #
    #             cur_label = torch.cat(cur_label, dim=0).to(device)
    #             cur_image = torch.cat(cur_image, dim=0).to(device)
    #
    #             cur_label = label_transform(cur_label)
    #             cur_image = image_transform(cur_image)
    #             # print(i,cur_label.shape,block_num)
    #             # cur_label = cur_label.unsqueeze(1)
    #             generate_img = MOEDL.sample_ldm(cur_label,cur_image)  # [batch,d,w,h]
    #             # generate_img = cur_label
    #             for sample in generate_img:
    #                 generate_img_blocks.append(sample)
    #         fake_image = splice_image(generate_img_blocks, block_num, max_num, image_size=image.shape[2:5],step=(32, 32, 32))
    #         print(generate_name, fake_image.shape, fake_image.max(), fake_image.min())
    #         fake_image = (fake_image - fake_image.min()) / (fake_image.max() - fake_image.min())
    #         fake_image = fake_image.squeeze(0).squeeze(0).cpu().numpy()
    #         save_image(os.path.join(test_result_dir, generate_name), fake_image * 255.0)
    #         # break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_name = "FDGMACPO_LDM_TEST11"
    MOEDL = Latent_Diffusion_Model(timesteps=100, model_name=model_name)
    # MOEDL.train_autoencoder(1)
    # MOEDL.train_LDM(20,load_AEcheckpoint=True)
    # MOEDL.trainer(20,20)
    MOEDL.test(load_checkpoint=True)




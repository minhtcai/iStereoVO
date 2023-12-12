import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from workflow import WorkFlow, TorchFlow
import torch.nn.functional as F
import wandb
wandb.login(key="75fbf7abfe79f7ac319d36b07d577306465d3a60")

from utils import tensor2img, depth_float32_rgba, visflow, visdepth
from CustomDataset import CustomDataset
from EurocDataset import EurocDataset
from torch.utils.data import Dataset, DataLoader
import kornia as K
from augmentation import StereoDataTransform, FlowDataTransform

from argumentsf import get_args
# from Network.FlowNet2 import FlowNet2 as FlowNet
from Network.StereoFlowNet import FlowNet
# from Network.PWC import PWCDCNet as FlowNet

# from scipy.io import savemat
np.set_printoptions(precision=4, threshold=10000)


class TrainStereoFlow(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainStereoFlow, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'stereoflow'

        self.stereoflownet = FlowNet()

        if self.args.load_flow_model:
            modelname = self.args.working_dir + '/models/' + args.flow_model
            self.load_model(self.stereoflownet, modelname)
            # loadPretrain(self.stereoflownet, modelname)

        if self.args.multi_gpu>1:
            self.stereoflownet = nn.DataParallel(self.stereoflownet)
            self.stereofunc = self.stereoflownet.module
        else:
            self.stereofunc = self.stereoflownet

        self.stereoflownet.cuda()

        self.LrDecrease = [int(self.args.train_step/2), int(self.args.train_step*3/4), int(self.args.train_step*7/8)]

        self.lr = self.args.lr

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.uncertainty = self.args.uncertainty
        self.stereo_normfactor = self.args.stereo_norm
        self.flow_normfactor = self.args.flow_norm

        base_path = '/project/learningvo/tartanair_v1_5'
        num_traj= 5
        test_num_traj= 5
        environments = [ # [Environment name, Number of Trajectories to Load]
                     ['abandonedfactory', num_traj],  
                     ['abandonedfactory_night', num_traj],
                     ['house',num_traj],
                     ['oldtown',num_traj],
                     ['house_dist0',num_traj],
                     ['seasidetown',num_traj],
                     ['amusement',num_traj],
                     ['japanesealley',num_traj],
                     ['seasonsforest',num_traj],
                     ['carwelding',num_traj],
                     ['neighborhood',num_traj],
                     ['seasonsforest_winter',num_traj],
                     ['endofworld',num_traj],
                     ['slaughter',num_traj],
                     ['gascola',num_traj],
                     ['ocean',num_traj],
                     ['soulcity',num_traj] 
                    ]
        test_environments = [
                     ['hongkongalley',test_num_traj],
                     ['office',test_num_traj],
                     ['westerndesert',test_num_traj],
                     ['hospital',test_num_traj],
                     ['office2',test_num_traj]
                    ]
        # stereo_train_dataset = CustomDataset(base_path, envs=environments, load_depth=True, load_flow=False,  load_left_img=True, load_right_img=True, load_next_img=False, left_camera=False)  
        # stereo_test_dataset = CustomDataset(base_path, envs=test_environments, load_depth=True, load_flow=False,  load_left_img=True, load_right_img=True, load_next_img=False, left_camera=False)  
        # self.stereotransform =  StereoDataTransform ((448, 512), data_augment=False, resize_factor=2.0, rand_hsv=0.2, random_rotate_rightimg=5.0, stereo_norm_factor=0.02)
    
        base_path = '/project/learningvo/euroc'
        trajectories = ['MH_01_easy_mav0_StereoRectified',
                        'MH_02_easy_mav0_StereoRectified',
                        'MH_03_medium_mav0_StereoRectified',
                        'MH_04_difficult_mav0_StereoRectified',
                        'MH_05_difficult_mav0_StereoRectified',
                        # 'V1_01_easy_mav0_StereoRectified',
                        # 'V1_02_medium_mav0_StereoRectified',
                        # 'V1_03_difficult_mav0_StereoRectified',
                        # 'V2_01_easy_mav0_StereoRectified',
                        # 'V2_02_medium_mav0_StereoRectified',
                        # 'V2_03_difficult_mav0_StereoRectified'
                    ]
        flow_test_dataset = EurocDataset(base_path, trajectories=trajectories, load_depth=False, load_disparity=False, load_flow=True, 
                                load_left_img=True, load_right_img=False, load_next_img=True, left_camera=True)
        # flow_train_dataset = CustomDataset(base_path, envs=environments, load_depth=False, load_disparity=False, load_flow=True,  load_left_img=True, load_right_img=False, load_next_img=True, left_camera=True)  
        # flow_test_dataset = CustomDataset(base_path, envs=test_environments, load_depth=False, load_disparity=False, load_flow=True,  load_left_img=True, load_right_img=False, load_next_img=True, left_camera=True)  
        self.flowtransform =  FlowDataTransform((448, 640), data_augment=False, resize_factor=2.0, rand_hsv=0.2, flow_norm_factor=0.05)

        if not args.test:
            # self.stereo_trainDataloader = DataLoader(stereo_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
            # self.strainDataiter = iter(self.stereo_trainDataloader)
            self.flow_trainDataloader = DataLoader(flow_train_dataset, batch_size=64, shuffle=True, num_workers=8)
            self.ftrainDataiter = iter(self.flow_trainDataloader)
        # self.stereo_testDataloader = DataLoader(stereo_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        # self.stestDataiter = iter(self.stereo_testDataloader)
        self.flow_testDataloader = DataLoader(flow_test_dataset, batch_size=1, shuffle=False, num_workers=8)
        self.ftestDataiter = iter(self.flow_testDataloader)

        self.criterion = nn.L1Loss()

        self.stereoOptimizer = optim.Adam(self.stereoflownet.parameters(),lr = self.lr)


    def initialize(self):
        super(TrainStereoFlow, self).initialize()

        self.AV['loss'].avgWidth = 1000
        # self.add_accumulated_value('t_sloss')
        self.add_accumulated_value('floss', 1000)
        self.add_accumulated_value('t_floss')
        if self.uncertainty:
            # self.add_accumulated_value('wounc_sloss', 1000)
            # self.add_accumulated_value('t_sloss_wounc')
            self.add_accumulated_value('wounc_floss', 1000)
            self.add_accumulated_value('t_floss_wounc')
            # self.append_plotter("stereo_loss", ['wounc_sloss', 't_sloss_wounc'], [True, False])
            # self.append_plotter("stereo_loss_unc", ['loss', 't_sloss'], [True, False])
            self.append_plotter("flow_loss", ['wounc_floss', 't_floss_wounc'], [True, False])
            self.append_plotter("flow_loss_unc", ['floss', 't_floss'], [True, False])
        else:
            # self.append_plotter("stereo_loss", ['loss', 't_sloss'], [True, False])
            self.append_plotter("flow_loss", ['floss', 't_floss'], [True, False])

        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        self.count = 0
        self.test_count = 0
        self.save_count = 0

        super(TrainStereoFlow, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.stereoflownet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward_stereo(self, sample, transform, mask=False): 
        leftTensor = sample['left_img'].squeeze(1).cuda()
        rightTensor = sample['right_img'].squeeze(1).cuda()
        targetdepth = sample['depth'].cuda()
        targetdisp = 80.0/(targetdepth + 10e-6)
        # print('left',torch.min(leftTensor), torch.max(leftTensor))
        # print('r',torch.min(rightTensor), torch.max(rightTensor))
        # print('d',torch.min(targetdisp), torch.max(targetdisp))
        # print(leftTensor.shape, rightTensor.shape, targetdisp.shape) #torch.Size([100, 3, 480, 640]) torch.Size([100, 3, 480, 640]) torch.Size([100, 480, 640])

        with torch.no_grad():
            leftTensor, rightTensor, targetdisp = transform(leftTensor, rightTensor, targetdisp)
            # print(leftTensor.shape, rightTensor.shape, targetdisp.shape) #torch.Size([10, 3, 448, 512]) torch.Size([10, 3, 448, 512]) torch.Size([10, 448, 512])
            # print('aleft',torch.min(leftTensor), torch.max(leftTensor))
            # print('ar',torch.min(rightTensor), torch.max(rightTensor))
            # print('ad',torch.min(targetdisp), torch.max(targetdisp))
            intensor=torch.cat((leftTensor, rightTensor), dim=1)
        output, output_unc= self.stereoflownet(intensor)

        if self.args.no_gt: # run test w/o GT file
            return 0, output/self.stereo_normfactor

        valid_mask = targetdisp>0 
        if valid_mask.sum() == torch.prod(torch.tensor(valid_mask.shape)):
            valid_mask = None
        # print(output.shape, targetdisp.shape)
        loss, loss_nounc = self.stereofunc.calc_loss(output, targetdisp, self.criterion, mask=valid_mask, unc=output_unc)

        loss_nounc = loss_nounc/self.stereo_normfactor if loss_nounc else None
        return loss/self.stereo_normfactor, loss_nounc, (output/self.stereo_normfactor, output_unc), (leftTensor, rightTensor, targetdisp) 

    def forward_flow(self, sample, transform, use_mask=False): 
        inputTensor1 = sample['left_img']
        # print(inputTensor1.shape) #[10, 3, 480, 640]
        inputTensor2 = sample['next_img']
        inputTensor = torch.cat((inputTensor1, inputTensor2), dim=1)
        inputTensor = inputTensor.cuda()
        targetflow = sample['flow'].squeeze(1).cuda()
        # print(inputTensor.shape, targetflow.shape) #[10, 6, 448, 640], [10, 2, 448, 640]

        with torch.no_grad():
            inputTensor, targetflow = transform(inputTensor, targetflow)
            # print(inputTensor.shape, targetflow.shape) #[10, 6, 448, 640] , ([10, 2, 448, 640]
        # img0Tensor = inputTensor[:,:3,:,:]
        # img1Tensor = inputTensor[:,3:,:,:]
        output, output_unc  = self.stereoflownet(inputTensor)
        # print(output.shape) # [10, 2, 448, 640]

        if use_mask and not self.args.uncertainty: # turn off mask if train with uncertainty
            mask = sample['fmask'].squeeze(1)
        else:
            mask = None
        # print(targetflow.shape)
        loss, loss_nounc = self.stereofunc.calc_loss(output, targetflow, self.criterion, mask=mask, unc=output_unc)

        loss_nounc = loss_nounc/self.flow_normfactor if loss_nounc else None
        return loss/self.flow_normfactor, loss_nounc, output, output_unc, inputTensor, targetflow

    def train(self):
        super(TrainStereoFlow, self).train()

        self.count = self.count + 1

        self.stereoflownet.train()

        # train stereo 
        # starttime1 = time.time()
        # try:
        #     sample = next(self.strainDataiter)
        # except StopIteration:
        #     print('New epoch..')
        #     self.strainDataiter = iter(self.stereo_trainDataloader)
        #     sample = next(self.strainDataiter)
        # loadtime1 = time.time()
        # self.stereoOptimizer.zero_grad()
        # loss_stereo, loss_stereo_nounc, _, _ = self.forward_stereo(sample, self.stereotransform)
        # loss_stereo.backward()
        # self.stereoOptimizer.step()
        # nntime1 = time.time()

        # self.AV['loss'].push_back(loss_stereo.item(), self.count)
        # if self.uncertainty:
        #     self.AV['wounc_sloss'].push_back(loss_stereo_nounc.item(), self.count)

        # train flow
        starttime2 = time.time()
        try:
            sample = next(self.ftrainDataiter)
        except StopIteration:
            print('New epoch..')
            self.ftrainDataiter = iter(self.flow_trainDataloader)
            sample = next(self.ftrainDataiter)
        loadtime2 = time.time()
        self.stereoOptimizer.zero_grad()
        loss_flow, loss_flow_nounc, _, _, _, _ = self.forward_flow(sample, self.flowtransform)
        loss_flow.backward()
        self.stereoOptimizer.step()
        nntime2 = time.time()

        self.AV['floss'].push_back(loss_flow.item(), self.count)
        if self.args.uncertainty:
            self.AV['wounc_floss'].push_back(loss_flow_nounc.item(), self.count)

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                self.lr = self.lr*0.3
                for param_group in self.stereoOptimizer.param_groups: # ed_optimizer is defined in derived class
                    param_group['lr'] = self.lr

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f - time load/bp (%.2f / %.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr, loadtime2-starttime2, nntime2-loadtime2))

        # if self.count % self.args.plot_interval == 0: 
        #     self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
            # for k in range(self.args.test_num):
            #     self.test(save_img=True, save_surfix_flow='test_flow_'+str(k), save_surfix_stereo='test_stereo_'+str(k))
        wandb.log({"train_loss":loss_flow.item(), "train_loss_nounc":loss_flow_nounc.item()})

    def visoutput_stereo(self, output, leftimg, rightimg, targetdisp, save_surfix='', unc=None):
        leftimg = tensor2img(leftimg[0], mean=self.mean, std=self.std)
        rightimg = tensor2img(rightimg[0], mean=self.mean, std=self.std)
        # import ipdb;ipdb.set_trace()
        outputnp = output[0].data.cpu().squeeze().numpy() # h x w
        resvis = visdepth(outputnp)
        targetdisp = targetdisp[0].squeeze().numpy()/self.stereo_normfactor
        dispvis = visdepth(targetdisp)

        diff = np.abs(outputnp - targetdisp)
        # import ipdb;ipdb.set_trace()

        diff[targetdisp==0] = 0 
        totalValid = np.sum(targetdisp>0)

        diffvis = visdepth(diff, scale=10)
        err3 = float(np.sum(diff>3))/totalValid *100
        lossnum = np.sum(diff)/totalValid

        img1 = np.concatenate((leftimg, diffvis),axis=0)
        img2 = np.concatenate((dispvis, resvis),axis=0)
        img = np.concatenate((img1,img2),axis=1)
        
        if self.args.uncertainty and unc is not None:
            uncnp = unc[0].data.cpu().squeeze().numpy()
            uncvis = np.clip((uncnp+2) * 100,0,255).astype(np.uint8)
            uncvis = np.tile(uncvis[:,:,np.newaxis], (1,1,3))
            img3 = np.concatenate((rightimg, uncvis),axis=0)
            img = np.concatenate((img,img3),axis=1)

        pts = np.array([[0,0],[200,0],[200,20],[0,20]],np.int32)
        cv2.fillConvexPoly(img,pts,(70,30,10))
        cv2.putText(img,'error = {:s}, err3 = {:s}'.format(str(lossnum)[0:5], str(err3)[0:5]),(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)
        cv2.imwrite(self.args.working_dir+'/testimg/'+self.args.exp_prefix+str(self.count)+'_'+save_surfix+'.jpg',img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    def visoutput_flow(self, output, img0Tensor, img1Tensor, targetflow, save_surfix='', output_unc=None):
        img1 = img0Tensor[0].cpu() #sample['img0'][0]
        img2 = img1Tensor[0].cpu() #sample['img0'][0]

        # print(targetflow.shape)
        flow = targetflow[0,:,:,:].cpu()
        img1 = tensor2img(img1, self.mean, self.std)
        img2 = tensor2img(img2, self.mean, self.std)
        flow = flow.numpy().transpose(1,2,0) / self.flow_normfactor #
        flowvis = visflow(flow)

        if type(output) is tuple: # in the case of pwc
            outputnp = F.interpolate(output[0][0:1], scale_factor=4, mode='bilinear', align_corners=True) # full size
            outputnp = outputnp.data.cpu().squeeze().numpy().transpose(1,2,0) / self.flow_normfactor
            resvis = self.pwc2vis(output, output_unc)
        else:
            outputnp = output[0].data.cpu().squeeze().numpy().transpose(1,2,0) / self.flow_normfactor # h x w
            resvis = visflow(outputnp)

        diff = np.abs(outputnp - flow)
        diff = np.mean(diff, axis=2)
        diffvis = visdepth(diff, scale=10)
        lossnum = np.mean(diff)

        disp1 = np.concatenate((img1, img2),axis=0)
        disp2 = np.concatenate((flowvis, resvis),axis=0)
        img = np.concatenate((disp1,disp2),axis=1)

        if self.args.uncertainty and output_unc is not None:
            uncnp = output_unc[0].data.cpu().squeeze().numpy()
            uncvis = np.clip((uncnp+1) * 100,0,255).astype(np.uint8)
            uncvis = np.tile(uncvis[:,:,np.newaxis], (1,1,3))
            pts = np.array([[0,0],[200,0],[200,20],[0,20]],np.int32)
            cv2.fillConvexPoly(uncvis,pts,(150,30,10))
            cv2.putText(uncvis,'unc mean = {:s}, std = {:s}'.format(str(np.mean(uncnp))[0:5], str(np.std(uncnp))[0:5]), 
                        (5,15), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)
            img3 = np.concatenate((img2, uncvis),axis=0)
            img = np.concatenate((img,img3),axis=1)

        pts = np.array([[0,0],[200,0],[200,20],[0,20]],np.int32)
        cv2.fillConvexPoly(img,pts,(70,30,10))
        cv2.putText(img,'error = {:s}'.format(str(lossnum)[0:5]),(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)
        # if save_surfix != '':
        cv2.imwrite(self.args.working_dir+'/testimg/'+self.args.exp_prefix+str(self.test_count)+'_'+save_surfix+'.jpg',img)
        # else:
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)

    def save_disp(self, output_np, savepathname):
        disp_rgba = depth_float32_rgba(output_np)
        cv2.imwrite(savepathname, disp_rgba)
        self.logger.info("Save file: {}".format(savepathname))

    def test(self, save_img=False, save_surfix_stereo='', save_surfix_flow=''):
        super(TrainStereoFlow, self).test()
        self.test_count += 1
        self.stereoflownet.eval()

        # stereo_loss_nounc, stereo_err3, finish_stereo = self.test_stereo(save_img=save_img, save_surfix=save_surfix_stereo)
        flow_loss_nounc, finish_flow = self.test_flow(save_img=save_img, save_surfix=save_surfix_flow)

        # return (stereo_loss_nounc, stereo_err3), flow_loss_nounc, (finish_stereo or finish_flow)
        return (flow_loss_nounc, finish_flow)

    def test_stereo(self, save_img=False, save_surfix=''):
        try:
            sample = next(self.stestDataiter)
        except StopIteration:
            self.stestDataiter = iter(self.stereo_testDataloader)
            sample = next(self.stestDataiter)
        with torch.no_grad():
            loss, loss_nounc,  (output, output_unc),(leftTensor, rightTensor, targetdisp) = self.forward_stereo(sample, self.stereotransform)#self.forward(sample, mask)
            outputnp = output.data.cpu().numpy()
        # print(outputnp.shape)
        lossnum = loss.item()
        self.AV['t_sloss'].push_back(lossnum, self.count)
        if self.uncertainty:
            self.AV['t_sloss_wounc'].push_back(loss_nounc.item(), self.count)
            lossnum = loss_nounc.item()
        leftTensor, rightTensor, targetdisp = leftTensor.cpu(), rightTensor.cpu(), targetdisp.cpu()
        targetdispnp = targetdisp.numpy() / self.stereo_normfactor
        # print(targetdispnp.shape)
        # targetdispnp = np.expand_dims(targetdispnp, axis=1)
        diff = np.abs(outputnp - targetdispnp)
        # print(diff.shape)
        diff[targetdispnp==0] = 0 
        totalValid = np.sum(targetdispnp>0)

        err3 = float(np.sum(diff>3))/totalValid *100

        loss_nounc = lossnum if loss_nounc is None else loss_nounc.item()
        self.logger.info("  TEST-Stereo %s #%d - loss %.4f, loss_nounc %.4f, err3 %.2f"  % (self.args.exp_prefix[:-1], 
            self.test_count, lossnum, loss_nounc, err3))

        if save_img:
            # visualize
            self.visoutput_stereo(output, leftTensor, rightTensor, targetdisp, save_surfix, unc=output_unc)

        finish = self.test_count*outputnp.shape[0]>= len(self.stereo_testDataloader) # only test one epoch when --test

        wandb.log({"test_loss":lossnum})

        return lossnum, err3, finish

    def test_flow(self, save_img=False, save_surfix=''):
        try:
            sample = next(self.ftestDataiter)
        except StopIteration:
            self.ftestDataiter = iter(self.flow_testDataloader)
            sample = next(self.ftestDataiter)
        with torch.no_grad():
            loss, loss_flow, output, output_unc, inputTensor, targetflow = self.forward_flow(sample, self.flowtransform)
        
        lossnum = loss.item()
        self.AV['t_floss'].push_back(lossnum, self.count)
        # print(inputTensor.shape)
        if self.args.uncertainty:
            self.AV['t_floss_wounc'].push_back(loss_flow.item(), self.count)
            lossnum = loss_flow.item()

        self.logger.info("  TEST-Flow %s #%d - loss %.4f"  % (self.args.exp_prefix[:-1], 
            self.test_count, lossnum))
        if save_img:
            self.visoutput_flow(output, inputTensor[:,:3,:,:], inputTensor[:,3:,:,:], targetflow, save_surfix, output_unc)

        # finish = self.test_count*output.shape[0]>= len(self.flow_testDataloader) # only test one epoch when --test
        # wandb.log({"test_loss":loss.item(), "test_loss_nounc": loss_flow.item()})

        return lossnum, finish

    def finalize(self):
        super(TrainStereoFlow, self).finalize()
        if self.count < self.args.train_step and not self.args.test:
            self.dumpfiles()

        if not self.args.test:
            self.flow_trainDataloader.stop_cachers()
        self.flow_testDataloader.stop_cachers()

        if not self.args.test:
            self.stereo_trainDataloader.stop_cachers()
        self.stereo_testDataloader.stop_cachers()

if __name__ == '__main__':
    args = get_args()
    CONFIG= {"lr": args.lr,
            #  "lr_scaling": args.lr_scale,
            #  "lrbatch": args.nbatch,
             "batch_size":args.batch_size,
            #  "transforms": args.transform,
            #  "network":args.network,
            #  "model_name": args.model_name,
            #  "num_workers":args.worker_num,
            #  "multi_gpu":args.multi_gpu
             }

    if args.use_int_plotter:
        plottertype = 'Int'
    else:
        plottertype = 'Visdom'
    try:
        # Instantiate an object for MyWF.
        trainStereoFlow = TrainStereoFlow(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)
        trainStereoFlow.initialize()
        if not args.test:
            run=wandb.init(project=f"flowonly", config=CONFIG)
            while True:
                trainStereoFlow.train()
                if (trainStereoFlow.count >= args.train_step):
                    break
        else: # testing 
        
            finish = False
            while True:
                saveimg = True # if trainStereoFlow.test_count<10 else False
                flowloss, finish = trainStereoFlow.test(save_img = saveimg)
                

        trainStereoFlow.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainStereoFlow.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")



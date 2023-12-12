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
from torch.utils.data import Dataset, DataLoader
import kornia as K
from augmentation import StereoDataTransform, FlowDataTransform, StereoVOTransform

from argumentsfull import get_args
from Network.posenet import VOFlowRes
from Network.StereoFlowNet import StereoFlowNet, StereoNet, FlowNet
from Network.StereoVONet import StereoVONet
# from Network.StereoNet7 import StereoNet7
# from Network.PSM import stackhourglass as StereoNet

np.set_printoptions(precision=4, threshold=10000, suppress=True)

def make_intrinsics_layer(intrinsics, intrinsics_scale=(1.0, 1.0) ):
    '''
    intrinsics: N x 6 tensor
    intrinsics_scale: N x 2 tensor
    '''
    w, h, fx, fy, ox, oy = intrinsics
    ww, hh = torch.meshgrid(torch.arange(w), torch.arange(h))
    ww = (ww.float() - ox + 0.5 )/fx
    hh = (hh.float() - oy + 0.5 )/fy
    intrinsicLayer = torch.stack((ww, hh))
    if intrinsics_scale[0] != 1 or intrinsics_scale[1] != 1:
        intrinsicLayer = K.geometry.rescale(intrinsicLayer, intrinsics_scale, align_corners=True)
    return intrinsicLayer

class TrainStereoVONet(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainStereoVONet, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'stereovoflow'

        self.intrinsics= (480.0, 640.0, 320.0, 320.0, 320.0, 240.0)
        self.intrinsics_scale =(1.0, 1.0)

        unc = 2 if args.uncertainty else 0
        self.vonet = StereoVONet(network=2, intrinsic=self.args.intrinsic_layer, 
                            down_scale=args.downscale_flow, uncertainty = unc, config=args.resvo_config, 
                            fixflow=args.fix_flow, fixstereo=args.fix_stereo, autoDistTarget=args.auto_dist_target)

        # load stereo
        if args.load_stereo_model:
            modelname0 = self.args.working_dir + '/models/' + args.stereo_model
            self.load_model(self.vonet.stereoNet, modelname0)

        # load flow
        if args.load_flow_model:
            modelname1 = self.args.working_dir + '/models/' + args.flow_model
            if args.flow_model.endswith('tar'): # load pwc net
                data = torch.load(modelname1)
                self.vonet.flowNet.load_state_dict(data)
                print('load pwc network...')
            else:
                self.load_model(self.vonet.flowNet, modelname1)

        # load pose
        if args.load_pose_model:
            modelname2 = self.args.working_dir + '/models/' + args.pose_model
            self.load_model(self.vonet.flowPoseNet, modelname2)

        # load the whole model
        if self.args.load_model:
            modelname = self.args.working_dir + '/models/' + self.args.model_name
            self.load_model(self.vonet, modelname)
        
        if self.args.multi_gpu>1:
            self.vonet = nn.DataParallel(self.vonet)
            self.vonetfunc = self.vonet.module
        else:
            self.vonetfunc = self.vonet
        self.vonet.cuda()

        self.pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013] # hard code, use when save motionfile when testing

        self.LrDecrease = [int(self.args.train_step/2), int(self.args.train_step*3/4), int(self.args.train_step*7/8)]
        self.lr = self.args.lr
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.uncertainty = self.args.uncertainty
        self.stereo_norm = self.args.stereo_norm
        self.flow_norm = self.args.flow_norm

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
        stereo_train_dataset = CustomDataset(base_path, envs=environments, load_depth=True, load_flow=True,  load_left_img=True, load_right_img=True, load_next_img=True, left_camera=True)  
        stereo_test_dataset = CustomDataset(base_path, envs=test_environments, load_depth=True, load_flow=True,  load_left_img=True, load_right_img=True, load_next_img=True, left_camera=True)  
        self.stereotransform =  StereoVOTransform ((448, 640), rand_hsv=0.3, resize_factor=2.0, flow_norm_factor=0.05, stereo_norm_factor=0.02)

        if not args.test:
            self.stereo_trainDataloader = DataLoader(stereo_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
            self.strainDataiter = iter(self.stereo_trainDataloader)
        self.stereo_testDataloader = DataLoader(stereo_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.stestDataiter = iter(self.stereo_testDataloader)
        print(len(self.stereo_trainDataloader))
        print(len(self.stereo_testDataloader))

        self.criterion = nn.L1Loss()

        # self.stereoOptimizer = optim.Adam(self.voflownet.parameters(),lr = self.lr)
        # self.flowOptimizer = optim.Adam(self.flownet.parameters(),lr = self.lr)
        self.voflowOptimizer = optim.Adam(self.vonet.parameters(), lr = self.lr)

    def initialize(self):
        super(TrainStereoVONet, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('flow', 100)
        self.add_accumulated_value('stereo', 100)
        self.add_accumulated_value('pose', 100)
        self.add_accumulated_value('vo_flow', 100)
        self.add_accumulated_value('vo_stereo', 100)

        self.add_accumulated_value('test', 1)
        self.add_accumulated_value('t_flow', 1)
        self.add_accumulated_value('t_stereo', 1)
        self.add_accumulated_value('t_pose', 1)

        self.add_accumulated_value('t_trans', 1)
        self.add_accumulated_value('t_rot', 1)
        self.add_accumulated_value('trans', 100)
        self.add_accumulated_value('rot', 100)
        self.append_plotter("loss", ['loss', 'test'], [True, False])
        self.append_plotter("loss_flow", ['flow', 'vo_flow', 't_flow'], [True, True, False])
        self.append_plotter("loss_pose", ['pose', 't_pose'], [True, False])
        self.append_plotter("loss_stereo", ['stereo', 'vo_stereo', 't_stereo'], [True, True, False])
        self.append_plotter("trans_rot", ['trans', 'rot', 't_trans', 't_rot'], [True, True, False, False])


        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        # if not self.args.test:
        #     self.logger.info("Training datasets:")
        #     # with open(self.args.train_spec_file, 'r') as f:
        #     #     lines = f.readlines()
        #     # self.logger.info(''.join(lines))

        # self.logger.info("Testing datasets:")
        # with open(self.args.test_spec_file, 'r') as f:
        #     lines = f.readlines()
        # self.logger.info(''.join(lines))
        self.count = 0
        self.test_count = 0
        self.epoch = 0

        super(TrainStereoVONet, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.vonet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward_vo(self, sample, transform): 
        currTensor = sample['left_img'].unsqueeze(1).cuda()
        nextTensor = sample['next_img'].unsqueeze(1).cuda()
        leftTensor = torch.cat((currTensor, nextTensor), dim=1)
        rightTensor = sample['right_img'].unsqueeze(1).cuda()
        flowTensor = sample['flow'].cuda()
        depthTensor = sample['depth'].cuda()
        dispTensor = 80.0/(depthTensor + 10e-6)
        # intrinsics = sample['intrinsics'].cuda()
        intrinsics_layer = make_intrinsics_layer(self.intrinsics, self.intrinsics_scale)
        intrinsics_layer = intrinsics_layer.repeat(flowTensor.shape[0],1,1,1).cuda()

        # print(currTensor.shape, leftTensor.shape, rightTensor.shape, flowTensor.shape, dispTensor.shape, intrinsics_layer.shape)
        # torch.Size([64, 1, 3, 480, 640]) torch.Size([64, 2, 3, 480, 640]) torch.Size([64, 1, 3, 480, 640]) torch.Size([64, 2, 480, 640]) torch.Size([64, 480, 640]) torch.Size([64, 2, 480, 640])
        leftTensor, rightTensor, flowTensor, dispTensor, intrinsics, scaleTensor = transform(leftTensor, rightTensor, flowTensor, dispTensor, intrinsics_layer)

        img0_flow   = leftTensor[:,0]
        img1_flow   = leftTensor[:,1]
        # import ipdb;ipdb.set_trace()
        # batch = leftTensor.shape[0]
        # blxfx = (sample['blxfx'].squeeze(1) * scaleTensor[:,1]).view((batch, 1, 1, 1)).cuda() # batch x 1 x 1 x 1

        # if random.random()>self.args.vo_gt_flow: 
        flow_output, stereo_output, pose_output = self.vonet(img0_flow, img1_flow, leftTensor[:,0], rightTensor[:,0], intrinsics, 
                                                                scale_disp=self.args.scale_disp,
                                                                blxfx = None)
        # else:
        #     flow_output, stereo_output, pose_output = self.vonet(img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsic, 
        #                                                          scale_w=scale_w, gt_flow=flow, gt_disp=disp, 
        #                                                          scale_disp=self.args.scale_disp,
        #                                                          blxfx = blxfx)
        pose_output_np = pose_output.data.cpu().detach().numpy()

        if self.args.no_gt: 
            return 0., 0., 0., 0.,0., pose_output_np

        # calculate flow loss
        flowloss, _ = self.vonetfunc.get_flow_loss(flow_output[0], flowTensor, self.criterion, mask=None) 
        stereoloss, _ = self.vonetfunc.get_stereo_loss(stereo_output[0], dispTensor, self.criterion) 
        flowloss = flowloss / self.flow_norm
        stereoloss = stereoloss / self.stereo_norm
        # calculate vo loss
        motion = sample['motion'].squeeze(1)
        motion = motion / torch.tensor(self.pose_norm,dtype=torch.float32)
        # import ipdb;ipdb.set_trace()

        lossPose = self.criterion(pose_output, motion.cuda())
        diff = torch.abs(pose_output.data.cpu().detach() - motion)
        trans_loss = diff[:,:3].mean().item()
        rot_loss = diff[:,3:].mean().item()
        return flowloss, stereoloss, lossPose, trans_loss, rot_loss, pose_output_np
    
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
            # print("mask is none")
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

        loss, loss_nounc = self.stereofunc.calc_loss(output, targetflow, self.criterion, mask=mask, unc=output_unc)

        loss_nounc = loss_nounc/self.flow_normfactor if loss_nounc else None
        return loss/self.flow_normfactor, loss_nounc, (output, output_unc), (inputTensor, targetflow)

    def train(self):
        super(TrainStereoVONet, self).train()

        self.count = self.count + 1

        self.vonet.train()

        # train stereo 
        try:
            sample = next(self.strainDataiter)
        except StopIteration:
            print('New epoch..')
            self.strainDataiter = iter(self.stereo_trainDataloader)
            sample = next(self.strainDataiter)

        self.voflowOptimizer.zero_grad()
        flowloss, stereoloss, poseloss, trans_loss, rot_loss, _ = self.forward_vo(sample, self.stereotransform)
        loss = poseloss
        if not self.args.fix_flow:
            loss = loss + flowloss * self.args.lambda_flow 
        if not self.args.fix_stereo:
            loss = loss + stereoloss * self.args.lambda_stereo #             

        loss.backward()
        self.voflowOptimizer.step()

        # import ipdb;ipdb.set_trace()
        self.AV['loss'].push_back(loss.item(), self.count)
        self.AV['vo_flow'].push_back(flowloss.item(), self.count)
        self.AV['vo_stereo'].push_back(stereoloss.item(), self.count)
        self.AV['pose'].push_back(poseloss.item(), self.count)
        self.AV['trans'].push_back(trans_loss, self.count)
        self.AV['rot'].push_back(rot_loss, self.count)

        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                    self.lr = self.lr*0.4
                    for param_group in self.voflowOptimizer.param_groups: 
                        param_group['lr'] = self.lr

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr:%.6f "  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr))

        # if self.count % self.args.plot_interval == 0: 
        #     self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
        wandb.log({"train_loss":loss.item(), "flow": flowloss.item(), 'stereo': stereoloss.item(),
                   'pose':poseloss.item(), 'trans': trans_loss, 'rot': rot_loss})

    
    def test(self):
        super(TrainStereoVONet, self).test()
        self.test_count += 1

        self.vonet.eval()
        try:
            sample = next(self.stestDataiter)
        except StopIteration:
            self.stestDataiter = iter(self.stereo_testDataloader)
            sample = next(self.stestDataiter)

        with torch.no_grad():
            flowloss, stereoloss, poseloss, trans_loss, rot_loss, motion = self.forward_vo(sample, self.stereotransform)

        finish = self.test_count*motion.shape[0]>= len(self.stereo_testDataloader)
        motion_unnorm = motion.squeeze() * self.pose_norm

        if self.args.no_gt:
            if self.test_count % self.args.print_interval == 0:
                self.logger.info("  TEST %s #%d - output : %s"  % (self.args.exp_prefix[:-1], 
                    self.test_count, motion_unnorm))
            return 0, 0, 0, 0, 0, 0, motion_unnorm, finish

        loss = poseloss
        if not self.args.fix_flow:
            loss = loss + flowloss * self.args.lambda_flow 
        if not self.args.fix_stereo:
            loss = loss + stereoloss * self.args.lambda_stereo #             

        lossnum = loss.item()
        self.AV['test'].push_back(lossnum, self.count)
        self.AV['t_flow'].push_back(flowloss.item(), self.count)
        self.AV['t_stereo'].push_back(stereoloss.item(), self.count)
        self.AV['t_pose'].push_back(poseloss.item(), self.count)
        self.AV['t_trans'].push_back(trans_loss, self.count)
        self.AV['t_rot'].push_back(rot_loss, self.count)

        self.logger.info("  TEST %s #%d - (loss, flow, stereo, pose, rot, trans) %.4f  %.4f  %.4f  %.4f  %.4f  %.4f"  % (self.args.exp_prefix[:-1], 
            self.test_count, loss.item(), flowloss.item(), stereoloss.item(), poseloss.item(), rot_loss, trans_loss))
        
        wandb.log({"test_loss":lossnum, "t_flow": flowloss.item(), 't_stereo': stereoloss.item(),
                   't_pose':poseloss.item(), 't_trans': trans_loss, 't_rot': rot_loss})

        return lossnum, flowloss.item(), stereoloss.item(), poseloss.item(), trans_loss, rot_loss, motion_unnorm, finish

    def finalize(self):
        super(TrainStereoVONet, self).finalize()
        if self.count < self.args.train_step and not self.args.test and not self.args.test_traj:
            self.dumpfiles()

        if self.args.test and not self.args.no_gt:
            self.logger.info('The average loss values: (t-trans, t-rot, t-flow, t-pose)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['test'].last_avg(100), 
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100),
                self.AV['t_flow'].last_avg(100),
                self.AV['t_pose'].last_avg(100)))

        else:
            self.logger.info('The average loss values: (loss, trans, rot, test, t_trans, t_rot)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans'].last_avg(100),
                self.AV['rot'].last_avg(100),
                self.AV['test'].last_avg(100),
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100)))

        # if not self.args.test:
        #     if self.args.train_vo:
        #         self.trainDataloader.stop_cachers()
        #     if self.args.train_flow:
        #         self.trainFlowDataloader.stop_cachers()
        #     if self.args.train_stereo:
        #         self.trainStereoDataloader.stop_cachers()
        # self.testDataloader.stop_cachers()

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
        trainVOFlow = TrainStereoVONet(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)
        trainVOFlow.initialize()

        if args.test:
            errorlist = []
            motionlist = []
            finish = False
            while not finish:
                error0, error1, error2, error3, error4, error5, motion, finish = trainVOFlow.test()
                errorlist.append([error0, error1, error2, error3, error4, error5])
                motionlist.append(motion)
                if ( trainVOFlow.test_count == args.test_num ):
                    break
            errorlist = np.array(errorlist)
            print("Test reaches the maximum test number (%d)." % (args.test_num))
            print("Loss statistics: loss/flow/stereo/pose/trans/rot: (%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f)" % (errorlist[:,0].mean(),
                            errorlist[:,1].mean(), errorlist[:,2].mean(), errorlist[:,3].mean(), errorlist[:,4].mean(), errorlist[:,5].mean()))

            if args.test_traj:
                # save motion file
                outputdir_prefix = args.test_output_dir+'/'+args.model_name.split('vonet')[0]+args.val_file.split('/')[-1].split('.txt')[0] # trajtest/xx_xx_euroc_xx
                motionfilename = outputdir_prefix +'_output_motion.txt'
                motions = np.array(motionlist)
                np.savetxt(motionfilename, motions)
                # visualize the file 
                # import ipdb;ipdb.set_trace()
                from error_analysis import evaluate_trajectory
                from evaluator.transformation import motion_ses2pose_quats, pose_quats2motion_ses
                from Datasets.utils import per_frame_scale_alignment
                gtposefile = args.gt_pose_file
                gtposes = np.loadtxt(gtposefile)
                gtmotions = pose_quats2motion_ses(gtposes)
                # estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
                estposes = motion_ses2pose_quats(motions)
                evaluate_trajectory(gtposes, estposes, trajtype=args.test_data_type, outfilename=outputdir_prefix, scale=False, medir_dir=args.test_output_dir)
        else: # Training
            run=wandb.init(project=f"svo_full", config=CONFIG)
            while True:
                trainVOFlow.train()
                if (trainVOFlow.count >= args.train_step):
                    break

        trainVOFlow.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainVOFlow.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")

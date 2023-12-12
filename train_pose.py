import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
import kornia as K
from augmentation import StereoFlowVODataTransform
import wandb

import time
wandb.login(key="75fbf7abfe79f7ac319d36b07d577306465d3a60")
# from scipy.io import savemat
np.set_printoptions(precision=4, threshold=10000, suppress=True)


def dataset_intrinsics(dataset='tartanair', calibfile=None):
    if dataset == 'kitti':
        # focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
        with open(calibfile, 'r') as f:
            lines = f.readlines()
        cam_intrinsics = lines[2].strip().split(' ')[1:]
        focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])
    elif dataset == 'euroc':
        # focalx, focaly, centerx, centery = 355.6358642578, 417.1617736816, 362.2718811035, 249.6590118408
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000

    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0

    elif dataset == 'realsense':
        focalx, focaly, centerx, centery = 379.2695007324, 379.4417419434, 317.5869721176, 239.5056236642
    else:
        return None
    return focalx, focaly, centerx, centery
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

class TrainVOStereoFlow(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainVOStereoFlow, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'stereovoflow'

        from Network.posenet import VOFlowRes
        unc_add_layer = 2 if args.uncertainty else 0
        self.voflownet = VOFlowRes(self.args.intrinsic_layer, down_scale=args.downscale_flow, config=args.resvo_config, 
                                    stereo = True, autoDistTarget=args.auto_dist_target, uncertainty=unc_add_layer)

        if self.args.load_model:
            modelname = self.args.working_dir + '/models/' + self.args.model_name
            self.load_model(self.voflownet, modelname)

        


        if self.args.multi_gpu>1:
            self.voflownet = nn.DataParallel(self.voflownet)

        self.voflownet.cuda()

        self.LrDecrease = [int(self.args.train_step/2), int(self.args.train_step*3/4), int(self.args.train_step*7/8)]
        self.lr = self.args.lr

        self.intrinsics= (480.0, 640.0, 320.0, 320.0, 320.0, 240.0)
        self.intrinsics_scale =(1.0, 1.0)
        self.pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013] # hard code, use when save motionfile when testing
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
        train_dataset = CustomDataset(base_path, envs=environments, load_depth=True, load_flow=True, left_camera=False)  
        test_dataset = CustomDataset(base_path, envs=test_environments, load_depth=True, load_flow=True, left_camera=False)  
        self.transform =  StereoFlowVODataTransform((120, 160), resize_factor=1.0, flow_norm_factor=0.05, inverse_depth_factor=4.0, uncertainty=False)  
    
        if not args.test:
            self.trainDataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8)
            self.trainDataiter = iter(self.trainDataloader)
        self.testDataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=8)
        self.testDataiter = iter(self.testDataloader)

        self.criterion = nn.L1Loss()
        self.voflowOptimizer = optim.Adam(self.voflownet.parameters(), lr = self.lr)


    def initialize(self):
        super(TrainVOStereoFlow, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('test_loss', 1)
        self.add_accumulated_value('test_trans_loss', 1)
        self.add_accumulated_value('test_rot_loss', 1)
        self.add_accumulated_value('trans_loss', 100)
        self.add_accumulated_value('rot_loss', 100)
        self.append_plotter("loss", ['loss', 'test_loss'], [True, False])
        self.append_plotter("trans_rot", ['trans_loss', 'rot_loss', 'test_trans_loss', 'test_rot_loss'], [True, True, False, False])

        if self.args.test_traj: # additional plot for testing
            self.add_accumulated_value('trans_loss_norm', 100)
            self.add_accumulated_value('rot_loss_norm', 100)
            self.append_plotter("loss_norm", ['trans_loss_norm', 'rot_loss_norm'], [True, True])


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

        super(TrainVOStereoFlow, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.voflownet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward(self, sample): 
        depth = sample['depth'].cuda()
        # print('depth',torch.min(depth), torch.max(depth))
        flow = sample['flow'].cuda()
        # print('flow',torch.min(flow), torch.max(flow))
        # intrinsics = sample['intrinsics'].cuda()
        intrinsics_layer = make_intrinsics_layer(self.intrinsics, self.intrinsics_scale)
        # print('intrinsics_layer',torch.min(intrinsics_layer), torch.max(intrinsics_layer))
        # print(depth.shape)
        intrinsics_layer = intrinsics_layer.repeat(flow.shape[0],1,1,1).cuda()
        # print(intrinsics_layer.shape)
        # flow, depth, intrinsics = transform(flow, depth, intrinsics)

        # 
        # print('bef',flow.shape, depth.shape, intrinsics_layer.shape)
        flow, depth, intrinsics_layer = self.transform (flow, depth, intrinsics_layer)
        # print('intrinsics_layer2',torch.min(intrinsics_layer), torch.max(intrinsics_layer))
        # print('flow',torch.min(flow), torch.max(flow))
        # print('depth',torch.min(depth), torch.max(depth))
        # print('aft',flow.shape, depth.shape, intrinsics_layer.shape)
        inputTensor = torch.cat((flow, depth, intrinsics_layer ), dim=1)
        # print('in',inputTensor.shape)
        output = self.voflownet(inputTensor, scale_disp=self.args.scale_disp)
        outputnp = output.data.cpu().detach().numpy()

        if self.args.no_gt: 
            return 0., 0., 0., outputnp

        motion = sample['motion'].squeeze(1)
        motion = motion / torch.tensor(self.pose_norm,dtype=torch.float32)

        loss, trans_loss, rot_loss = self.linear_norm_trans_loss(output, motion.cuda())

        # loss = self.criterion(output, motion.cuda())
        # diff = torch.abs(output.data.cpu().detach() - motion)
        # trans_loss = diff[:,:3].mean().item()
        # rot_loss = diff[:,3:].mean().item()

        return loss, trans_loss, rot_loss, outputnp

    def linear_norm_trans_loss(self, output, motion):
        output_trans = output[:, :3]
        output_rot = output[:, 3:]

        trans_norm = torch.linalg.norm(output_trans, dim=1).view(-1, 1)
        output_norm = output_trans/trans_norm

        motion_trans_norm = torch.linalg.norm(motion[:, :3], dim=1).view(-1, 1)
        motion_norm = motion[:, :3] / motion_trans_norm

        trans_loss = self.criterion(output_norm, motion_norm)
        rot_loss = self.criterion(output_rot, motion[:, 3:])

        loss = (rot_loss + trans_loss)/2.0

        return loss, trans_loss.item() , rot_loss.item()


    def train(self):
        super(TrainVOStereoFlow, self).train()

        self.count = self.count + 1
        self.voflownet.train()

        starttime = time.time()

        try:
            sample = next(self.trainDataiter)
        except StopIteration:
            print('New epoch..')
            self.trainDataiter = iter(self.trainDataloader)
            sample = next(self.trainDataiter)

        loadtime = time.time()

        self.voflowOptimizer.zero_grad()

        loss, trans_loss, rot_loss, outputnp = self.forward(sample)

        loss.backward()
        self.voflowOptimizer.step()

        nntime = time.time()

        # import ipdb;ipdb.set_trace()
        self.AV['loss'].push_back(loss.item(), self.count)
        self.AV['trans_loss'].push_back(trans_loss, self.count)
        self.AV['rot_loss'].push_back(rot_loss, self.count)

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                self.lr = self.lr*0.4
                for param_group in self.voflowOptimizer.param_groups: # ed_optimizer is defined in derived class
                    param_group['lr'] = self.lr

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            # calculate ave-trans
            if self.args.no_gt:
                ave_trans = np.linalg.norm(outputnp[:, :3], axis=1)
            else:
                ave_trans = np.linalg.norm(sample['motion'][:, :3], axis=1)
                
            self.logger.info("%s #%d - %s lr%%: %.4f load/bp time (%.3f, %.3f) ave-trans: (%.2f, %.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr*100, loadtime-starttime, nntime-loadtime, ave_trans.mean(), ave_trans.std()))

        # if self.count % self.args.plot_interval == 0: 
        #     self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
            # for k in range(self.args.test_num):
            #     self.test(save_img=True, save_surfix='test_'+str(k))
        wandb.log({"train_loss":loss.item(),'train_trans_loss': trans_loss, 'train_rot_loss': rot_loss})

    def test(self):
        super(TrainVOStereoFlow, self).test()
        self.test_count += 1

        try:
            sample = next(self.testDataiter)
        except StopIteration:
            self.testDataiter = iter(self.testDataloader)
            sample = next(self.testDataiter)
        
        self.voflownet.eval()

        with torch.no_grad():
            loss, trans_loss, rot_loss, motion = self.forward(sample)

        finish = self.test_count*motion.shape[0]>= len(self.testDataloader)
        motion_unnorm = motion.squeeze() * self.pose_norm

        if self.args.no_gt:
            if self.test_count % self.args.print_interval == 0:
                self.logger.info("  TEST %s #%d - output : %s"  % (self.args.exp_prefix[:-1], 
                    self.test_count, motion_unnorm))
            return 0, 0, 0, motion_unnorm, finish

        lossnum = loss.item()
        self.AV['test_loss'].push_back(lossnum, self.count)
        self.AV['test_trans_loss'].push_back(trans_loss, self.count)
        self.AV['test_rot_loss'].push_back(rot_loss, self.count)

        wandb.log({"test_loss":lossnum,'test_trans_loss': trans_loss, 'test_rot_loss': rot_loss })

        if self.test_count % self.args.print_interval == 0:
            self.logger.info("  TEST %s #%d - loss/trans/rot: %.4f  %.4f  %.4f"  % (self.args.exp_prefix[:-1], 
                self.test_count, lossnum, trans_loss, rot_loss))

        return lossnum, trans_loss, rot_loss, motion_unnorm, finish


    def finalize(self):
        super(TrainVOStereoFlow, self).finalize()
        if self.count < self.args.train_step and not self.args.test and not self.args.test_traj:
            self.dumpfiles()

        if self.args.test_traj and not self.args.no_gt:
            self.logger.info('The average loss values:')
            self.logger.info('loss/trans/rot/norm_trans/norm_rot: %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans_loss'].last_avg(100),
                self.AV['rot_loss'].last_avg(100),
                self.AV['trans_loss_norm'].last_avg(100),
                self.AV['rot_loss_norm'].last_avg(100)))

        elif not self.args.test:
            self.logger.info('The average loss values:')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans_loss'].last_avg(100),
                self.AV['rot_loss'].last_avg(100),
                self.AV['test_loss'].last_avg(100),
                self.AV['test_trans_loss'].last_avg(100),
                self.AV['test_rot_loss'].last_avg(100)))

        if not self.args.test:
            self.trainDataloader.stop_cachers()
        self.testDataloader.stop_cachers()

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
        trainVOStereoFlow = TrainVOStereoFlow(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)
        trainVOStereoFlow.initialize()

        if args.test:
            errorlist = []
            motionlist = []
            finish = False
            while not finish:
                error0, error1, error2, motion, finish = trainVOStereoFlow.test()
                errorlist.append([error0, error1, error2])
                motionlist.append(motion)
                if ( trainVOStereoFlow.test_count == args.test_num ):
                    break
            print("Test reaches the maximum test number (%d)." % (args.test_num))
            errorlist = np.array(errorlist)
            print("Loss statistics: loss/trans/rot: (%.4f \t %.4f \t %.4f)" % (errorlist[:,0].mean(),errorlist[:,1].mean(), errorlist[:,2].mean()))

            if args.test_traj:
                # save motion file
                outputdir_prefix = args.test_output_dir+'/'+args.model_name.split('stereovoflow')[0].split('vonet')[0]+args.val_file.split('/')[-1].split('.txt')[0] # trajtest/xx_xx_euroc_xx
                motionfilename = outputdir_prefix +'_output_motion.txt'
                motions = np.array(motionlist)
                np.savetxt(motionfilename, motions)
                # visualize the file 
                # import ipdb;ipdb.set_trace()
                from error_analysis import evaluate_trajectory
                from evaluator.transformation import motion_ses2pose_quats
                gtposefile = args.gt_pose_file
                gtposes = np.loadtxt(gtposefile)
                estposes = motion_ses2pose_quats(motions)
                evaluate_trajectory(gtposes, estposes, trajtype=args.test_data_type, outfilename=outputdir_prefix, scale=False, medir_dir=args.test_output_dir)

        else: # Training
            run=wandb.init(project=f"stereo", config=CONFIG)
            while True:
                trainVOStereoFlow.train()
                if (trainVOStereoFlow.count >= args.train_step):
                    break

        trainVOStereoFlow.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainVOStereoFlow.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")



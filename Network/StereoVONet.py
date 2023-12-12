
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .StereoFlowNet import StereoFlowNet, StereoNet, FlowNet
from .posenet import VOFlowRes


class StereoVONet(nn.Module):
    def __init__(self, network=0, intrinsic=True, down_scale=True, uncertainty = 0,
                        flownet_norm = 0.05, posenet_flownorm = 0.05, stereonet_norm = 0.02, inverse_depth_norm = 4.0, 
                        config=0, fixflow=True, fixstereo=True, autoDistTarget=0.):
        '''
        flownet_norm: norm factor used in flownet output
        posenet_flownorm: norm factor used in the flow input of the posenet
        stereonet_norm: norm factor used in the stereonet output
        inverse_depth_norm: norm factor used in the depth input of the posenet
        autoDistTarget: 0.  : no auto scale
                        > 0.: scale the distance wrt the mean value 
        '''
        super(StereoVONet, self).__init__()

        self.flowNet   = FlowNet()
        self.stereoNet = StereoNet()
        self.flowPoseNet = VOFlowRes(intrinsic, down_scale=down_scale, config=config, 
                                    stereo = True, autoDistTarget=autoDistTarget, uncertainty=uncertainty)

        self.network = network
        self.intrinsic = intrinsic
        # flow_input = flow_output / flownet_norm * posenet_flownorm
        self.flowNormFactor = posenet_flownorm / flownet_norm
        # depth_input = inverse_depth_norm / (blxfx / (disp_output / stereonet_norm))
        #             = disp_output * inverse_depth_norm / stereonet_norm / blxfx 
        self.inverseDepthFactor = inverse_depth_norm / stereonet_norm
        self.down_scale = down_scale
        self.uncertainty = uncertainty

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

        if fixstereo:
            for param in self.stereoNet.parameters():
                param.requires_grad = False

        self.autoDistTarget = autoDistTarget

    def forward(self, x0_flow, x0n_flow, x0_stereo, x1_stereo, intrin=None, 
                only_flow=False, only_stereo=False, # gt_flow=None, gt_disp=None, 
                scale_disp=1.0, blxfx=None):
        '''
        x0_flow, x0n_flow, x0_stereo: B x 3 x h x w
        intrin: B x 2 x h x w

        flow_out: pwcnet: 5 scale predictions up to 1/4 size
                  flownet: 1/1 size prediction
        stereo_out: psmnet: 3 scale predictions up to 1/1 size
                    stereonet: 1/1 size prediction 
        scale_w: the x-direction scale factor in data augmentation
        scale_depth: scale input depth and output motion to shift the data distribution
        blxfx: batch x 1 x 1 x 1, focal_x * baseline according to the scale_w in RCR
        '''
        # import ipdb;ipdb.set_trace()
        if only_flow:
            return self.flowNet(torch.cat([x0_flow, x0n_flow], dim=1))

        if only_stereo:
            return self.stereoNet(torch.cat([x0_stereo, x1_stereo], dim=1))

        downscale_size = 256
        downscale_size1 = 256
        downscale_size2 = 256

        # TODO: organize the code, move the following into the network implementation
        # hard code the size because the networks do not support smaller size
        if self.network>=2 and self.down_scale: # decrease the input size to accelerate the training
            input_w, input_h = x0_flow.shape[-1], x0_flow.shape[-2]
            x0_flow = F.interpolate(x0_flow, size=(downscale_size, downscale_size), mode='bilinear', align_corners=True)
            x0n_flow = F.interpolate(x0n_flow, size=(downscale_size, downscale_size), mode='bilinear', align_corners=True)
            x0_stereo = F.interpolate(x0_stereo, size=(downscale_size, downscale_size), mode='bilinear', align_corners=True)
            x1_stereo = F.interpolate(x1_stereo, size=(downscale_size, downscale_size), mode='bilinear', align_corners=True)

        flow_out, flow_out_unc = self.flowNet(torch.cat((x0_flow, x0n_flow),dim=1))
        stereo_out, stereo_out_unc = self.stereoNet(torch.cat((x0_stereo, x1_stereo),dim=1))

        if self.network>=2 and self.down_scale: # decrease the input size to accelerate the training
            flow_out = F.interpolate(flow_out, size=(int(input_h/4), int(input_w/4)), mode='bilinear', align_corners=True)
            flow_out[:,0,:,:] = flow_out[:,0,:,:] * (float(input_w)/downscale_size)
            flow_out[:,1,:,:] = flow_out[:,1,:,:] * (float(input_h)/downscale_size)
            flow = flow_out
            stereo_out = F.interpolate(stereo_out, size=(int(input_h/4), int(input_w/4)), mode='bilinear', align_corners=True)
            stereo_out[:,0,:,:] = stereo_out[:,0,:,:] * (float(input_w)/downscale_size)
            stereo = stereo_out


        flow_input    = flow * self.flowNormFactor
        depth_input   = stereo * (self.inverseDepthFactor / 80.0) #blxfx

        if self.uncertainty > 0:
            if self.down_scale:
                flow_out_unc = F.interpolate(flow_out_unc, size=(int(input_h/4), int(input_w/4)), mode='bilinear', align_corners=True)
                stereo_out_unc = F.interpolate(stereo_out_unc, size=(int(input_h/4), int(input_w/4)), mode='bilinear', align_corners=True)
            flow_out_unc = 0.5 * torch.tanh(-flow_out_unc-2)+0.5
            stereo_out_unc = 0.5 * torch.tanh(-stereo_out_unc-2)+0.5
            flow_input = torch.cat((flow_input, flow_out_unc), dim=1)
            depth_input = torch.cat((depth_input, stereo_out_unc), dim=1)

        if self.intrinsic:
            if self.down_scale:
                intrin = F.interpolate(intrin, scale_factor=0.25, mode='bilinear', align_corners=True)
            inputTensor = torch.cat( ( flow_input, depth_input, intrin ), dim=1 )
        else:
            inputTensor = torch.cat( ( flow_input, depth_input ), dim=1 )
        
        pose = self.flowPoseNet( inputTensor, scale_disp=scale_disp )
        # scale the translation back
        # if self.autoDistTarget == 0:
        #     pose[:, :3] = pose[:, :3] * scale_disp
        # else:
        #     pose[:, :3] = pose[:, :3] * scale_disp.view(scale_disp.shape+(1,))

        return (flow_out, flow_out_unc), (stereo_out, stereo_out_unc), pose

    def get_flow_loss(self, netoutput, target, criterion, mask=None, unc=None):
        if self.down_scale:
            target = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss, loss_nounc = self.flowNet.calc_loss(netoutput, target, criterion, mask = mask, unc = unc)
        return loss, loss_nounc

    def get_stereo_loss(self, netoutput, target, criterion, mask=None, unc=None):
        if self.down_scale:
            target = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss, loss_nounc = self.stereoNet.calc_loss(netoutput, target, criterion, mask = mask, unc = unc)
        return loss, loss_nounc

if __name__ == '__main__':
    
    voflownet = StereoVONet(network=2) # 
    voflownet.cuda()
    voflownet.eval()
    print (voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    x, y = np.ogrid[:448, :640]
    # print (x, y, (x+y))
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print (img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    intrin = imgInput[:,:2,:,:].copy()

    imgTensor = torch.from_numpy(imgInput)
    intrinTensor = torch.from_numpy(intrin)
    print (imgTensor.shape)
    stime = time.time()
    for k in range(100):
        flow, stereo, pose = voflownet(imgTensor.cuda(), imgTensor.cuda(), imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda(), blxfx=80.)
        # print (flow[0][0].data.shape, pose.data.shape)
        print (pose.data.cpu().numpy())
        print (time.time()-stime)
    print ((time.time()-stime)/100)

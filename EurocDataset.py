# depth = (f*b)/ (disparity + 10e-6) #f*b=80.0 for tartanair

import numpy as np
import cv2
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as trf
import torch.nn.functional as F
import matplotlib.pyplot as plt

def make_intrinsics_layer(intrinsics):
    w, h, fx, fy, ox, oy = intrinsics
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww, hh)).transpose(1,2,0)
    return intrinsicLayer

class EurocDataset(Dataset):
    ''' Loads samples from any combination of trajectories specified. 
    '''
    def __init__(self, base_path, trajectories, load_depth=True, load_disparity=True, load_flow=True, 
                 load_left_img=True, load_right_img=True, load_next_img=True,  
                 left_camera=True, img_scale=1.0, flow_depth_scale=1.0):
        self.base_path = base_path
        self.load_depth = load_depth
        self.load_disparity = load_disparity
        self.load_flow = load_flow
        self.load_left_img = load_left_img
        self.load_right_img = load_right_img
        self.load_next_img = load_next_img
        self.camera_type = 'left' if left_camera else 'right'
        self.img_scale = img_scale
        self.flow_depth_scale = flow_depth_scale
        
        self.trajs = trajectories
        self.data = {}
        self.num_samples = 0
        self.idx_map = []
        for p_idx, p in enumerate(self.trajs):
            p_path = os.path.join(self.base_path, p)
            p_data = {}
            
            # Load the images:
            if self.load_left_img or (self.load_next_img and self.camera_type == 'left'):
                left_img_files = sorted([os.path.join(p_path, f'cam0', 'data2', f) 
                                         for f in os.listdir(os.path.join(p_path, f'cam0', 'data2')) 
                                         if f.endswith(f'.png')])
                p_data['left_img_files'] = left_img_files
            if self.load_right_img or (self.load_next_img and self.camera_type == 'right'):
                right_img_files = sorted([os.path.join(p_path, f'cam1', 'data2', f) 
                                          for f in os.listdir(os.path.join(p_path, f'cam1', 'data2')) 
                                          if f.endswith(f'.png')])
                p_data['right_img_files'] = right_img_files
            
            # Load the motion data (except the last)
            motion_file_path = os.path.join(p_path, f'motion.txt')
            motion_data = np.loadtxt(motion_file_path)
            motion_data = motion_data[:-1]
            p_data['motion_data'] = motion_data
            self.idx_map += [[p_idx, data_idx] for data_idx in range(len(motion_data))]
            self.num_samples += len(motion_data)

            # Disparity and flow file paths (except the last)
            if self.load_depth or self.load_disparity:
                disp_files = sorted([os.path.join(p_path, f'cam0', f'disp_hsm', f) 
                                     for f in os.listdir(os.path.join(p_path, f'cam0', f'disp_hsm')) 
                                     if f.endswith(f'.png')])
                # import pdb;pdb.set_trace()
                disp_files = disp_files[:-1]
                # print(len(depth_files))
                p_data['disp_files'] = disp_files
            
            if self.load_flow:
                flow_files = sorted([os.path.join(p_path, 'cam0', 'flow', f) 
                                     for f in os.listdir(os.path.join(p_path, 'cam0', 'flow')) 
                                     if f.endswith('_flow.png')])
                # mask_files = sorted([os.path.join(p_path, 'flow', f) for f in os.listdir(os.path.join(p_path, 'flow')) if f.endswith('_mask.npy')])
                # print(len(flow_files))
                p_data['flow_files'] = flow_files
            
            self.data[p] = p_data                             

    def __len__(self):
        # The length of the dataset is determined by the number of motion samples
        return self.num_samples

    def visualize_depth(self, idx, maxthresh = 50):
        # Load
        p_idx, data_idx = self.idx_map[idx]
        p = self.trajs[p_idx]
        
        disp_path = self.data[p]['disp_files'][data_idx]
           
        # image_object = png.Reader(filename=disp_path)
        # image_direct = image_object.asDirect()
        # image_data = list(image_direct[2])
        # (w, h) = image_direct[3]['size']
        # channel = len(image_data[0]) / w
        # disp = np.zeros((h, w, channel), dtype=np.uint16)
        # for i in range(len(image_data)):
        #     for j in range(channel):
        #         disp[i, :, j] = image_data[i][j::channel]
        # disp = disp[:, :, 0] / 256 
        
        
        disp_rgba = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        assert disp_rgba is not None, "Error loading disparity {}".format(disp_path)
        
        disp = disp_rgba.view("<f4")
        
        depth = 458*0.11 / (disp + 10e-6)
        
        # Visualize
        depth = depth.reshape(depth.shape[:-1]) # (H, W)
        depthvis = np.clip(depth,0,None)
        depthvis = depthvis/maxthresh*255
        depthvis = depthvis.astype(np.uint8)
        depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))
        
        return depthvis
        
    def _calculate_angle_distance_from_du_dv(self, du, dv, flagDegree=False):
        a = np.arctan2( dv, du )

        angleShift = np.pi

        if ( True == flagDegree ):
            a = a / np.pi * 180
            angleShift = 180
            # print("Convert angle from radian to degree as demanded by the input file.")

        d = np.sqrt( du * du + dv * dv )

        return a, d, angleShift
    
    def visualize_flow(self, idx, maxF=500.0, n=8, mask=True, hueMax=179, angShift=0.0):
        # Load
        p_idx, data_idx = self.idx_map[idx]
        p = self.trajs[p_idx]
        
        flow_path = self.data[p]['flow_files'][data_idx]
        
        flow16 = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(flow_path)
        
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0
        
        # mask8 = flow16[:,:,2].astype(np.uint8)
        
        # Visualize
        ang, mag, _ = self._calculate_angle_distance_from_du_dv( flow32[:, :, 0], flow32[:, :, 1], flagDegree=False )

        # Use Hue, Saturation, Value colour model
        hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

        am = ang < 0
        ang[am] = ang[am] + np.pi * 2

        hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
        hsv[ :, :, 1 ] = mag / maxF * n
        hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

        hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
        hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
        hsv = hsv.astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # if mask:
        #     mask8 = mask8 > 0
        #     rgb[mask8] = np.array([0, 0 ,0], dtype=np.uint8)
        
        return rgb
        
    def visualize(self, idx, depth_maxthresh = 50, flow_maxF=500.0, flow_n=8, 
                  flow_mask=True, flow_hueMax=179, flow_angShift=0.0):
        # Get depth and flow visualizations
        depthvis = self.visualize_depth(idx, maxthresh=depth_maxthresh)
        rgb = self.visualize_flow(idx, maxF=flow_maxF, n=flow_n, mask=flow_mask, 
                                  hueMax=flow_hueMax, angShift=flow_angShift)
        # Get image
        p_idx, data_idx = self.idx_map[idx]
        p = self.trajs[p_idx]
        left_img_path = self.data[p]['left_img_files'][data_idx]
        img = cv2.imread(left_img_path)
        
        # Plot
        fig, axs = plt.subplots(1,3,clear=True, figsize=(15,5))
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[1].imshow(depthvis)
        axs[1].set_title('Depth')
        axs[2].imshow(rgb)
        axs[2].set_title('Flow')
        fig.savefig("/data1/datasets/msathena/workspace/stereo/Euroc_vis/full_%06d_%02d_%06d" 
                    % (idx,p_idx,data_idx))
        fig.clear()
        plt.close(fig)
    
    def load_img(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, "Error loading image {}".format(img_path)
        
        h, w = img.shape[:2]
        img = torch.from_numpy(img.transpose(2,0,1))
        return img
        # return trf.resize(img, (int(h*self.img_scale), int(w*self.img_scale)), 
        #                   antialias=True)
        # Batch size, 3 channels, H, W
    
    def get_by_traj(self, p, data_idx):
        data = {}
        
        # Load images
        if self.load_left_img:
            left_img_path = self.data[p]['left_img_files'][data_idx]
            data['left_img'] = self.load_img(left_img_path)
        if self.load_right_img:
            right_img_path = self.data[p]['right_img_files'][data_idx]
            data['right_img'] = self.load_img(right_img_path)
        if self.load_next_img:
            next_img_path = self.data[p][f'{self.camera_type}_img_files'][data_idx+1]
            data['next_img'] = self.load_img(next_img_path)
        
        # Load motion data and convert to float
        motion = self.data[p]['motion_data'][data_idx].astype(np.float32)
        data['motion'] = torch.from_numpy(motion)
        # Batch size, [tx, ty, tz, r...]

        # Load and convert depth data to float
        if self.load_depth or self.load_disparity:
            disp_path = self.data[p]['disp_files'][data_idx]
            
            disp_rgba = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
            assert disp_rgba is not None, "Error loading disparity {}".format(disp_path)
            
            disp = disp_rgba.view("<f4")
            
            # h, w = disp.shape[:2]
            disp = torch.from_numpy(disp.transpose(2,0,1))
            # data['disp'] = trf.resize(disp, (int(h*self.flow_disp_scale), int(w*self.flow_disp_scale)), 
            #                            antialias=True)
            # data['disp'] = trf.interpolate(disp, (int(h*self.flow_disp_scale), int(w*self.flow_disp_scale)), mode= 'linear',
            #                            align_corners=True)
            if self.load_depth:
                data['depth'] = 458*0.11 / (disp + 10e-6)
                # From https://github.com/raulmur/ORB_SLAM2/issues/32 (focal length checked with dataset)
            if self.load_disparity:
                data['disparity'] = disp
            # Batch size, 1 channel, H, W

        # Load and convert optical flow data to float
        if self.load_flow:
            flow_path = self.data[p]['flow_files'][data_idx]
            
            flow16 = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
            assert flow16 is not None, "Error loading flow {}".format(flow_path)
            
            flow32 = flow16[:,:,:2].astype(np.float32)
            flow32 = (flow32 - 32768) / 64.0
            
            # mask8 = flow16[:,:,2].astype(np.uint8)
            # mask8 = mask8 > 0
            # flow32[mask8] = np.array([0, 0], dtype=np.float32)
            
            # h, w = flow32.shape[:2]
            flow32 = torch.from_numpy(flow32.transpose(2,0,1))
            # data['flow'] = trf.resize(flow32, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), 
            #                           antialias=True)
            # data['flow'] = trf.interpolate(flow32, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), 
            #                           align_corners=True, antialias=True)
            data['flow'] = flow32
            # Batch size, 2 channels, H, W

        return data
    
    def __getitem__(self, idx):
        p_idx, data_idx = self.idx_map[idx]
        p = self.trajs[p_idx]
        return self.get_by_traj(p, data_idx)
        

if __name__ == "__main__":
    # Example usage
    base_path = '/project/learningvo/euroc'
    trajectories = ['MH_01_easy_mav0_StereoRectified',
                    # 'MH_02_easy_mav0_StereoRectified',
                    # 'MH_03_medium_mav0_StereoRectified',
                    # 'MH_04_difficult_mav0_StereoRectified',
                    # 'MH_05_difficult_mav0_StereoRectified',
                    # 'V1_01_easy_mav0_StereoRectified',
                    # 'V1_02_medium_mav0_StereoRectified',
                    # 'V1_03_difficult_mav0_StereoRectified',
                    # 'V2_01_easy_mav0_StereoRectified',
                    # 'V2_02_medium_mav0_StereoRectified',
                    # 'V2_03_difficult_mav0_StereoRectified'
                   ]
    dataset = EurocDataset(base_path, trajectories=trajectories, load_depth=True, load_disparity=True, load_flow=True, 
                            load_left_img=True, load_right_img=True, load_next_img=True, left_camera=True)
    print(dataset[0]['left_img'].shape, dataset[0]['right_img'].shape, dataset[0]['next_img'].shape, 
          dataset[0]['motion'].shape, dataset[0]['depth'].shape, dataset[0]['disparity'].shape, dataset[0]['flow'].shape)
    for i in np.random.choice(len(dataset), 10, replace=False):
        dataset.visualize(i)
    dataset.visualize(182)
    
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Check a sample from the dataset
    for sample in data_loader:
        if 'left_img' in sample:
            print("Sample Left Image Shape:", sample['left_img'].shape)
        if 'right_img' in sample:
            print("Sample Right Image Shape:", sample['right_img'].shape)
        if 'next_img' in sample:
            print("Sample Next Image Shape:", sample['next_img'].shape)
        if 'motion' in sample:
            print("Sample Motion Data Shape:", sample['motion'].shape)
        if 'depth' in sample:
            print("Sample Depth Data Shape:", sample['depth'].shape)
        if 'depth' in sample:
            print("Sample Disparity Data Shape:", sample['disparity'].shape)
        if 'flow' in sample:
            print("Sample Flow Data Shape:", sample['flow'].shape)
        break  # Just show the first batch
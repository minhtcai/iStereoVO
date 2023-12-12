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

class CustomDataset(Dataset):
    ''' Loads samples from any combination of environments specified. 
        Loads motion (tx, ty, tz, rotation[3]), depth, and the flow mask.
        Allows accessing individual samples by their environment, trajectory index, and sample index.
    '''
    def __init__(self, base_path, envs, load_depth=True, load_disparity=True, load_flow=True, 
                 load_left_img=True, load_right_img=True, load_next_img=True,  
                 left_camera=True, data_fast=False, img_scale=1.0, flow_depth_scale=1.0):
        self.base_path = base_path
        self.load_depth = load_depth
        self.load_disparity = load_disparity
        self.load_flow = load_flow
        self.load_left_img = load_left_img
        self.load_right_img = load_right_img
        self.load_next_img = load_next_img
        self.camera_type = 'left' if left_camera else 'right'
        self.data_fast = 'Data_fast' if data_fast else 'Data'
        self.img_scale = img_scale
        self.flow_depth_scale = flow_depth_scale
        
        self.envs = envs
        self.data = {}
        self.num_samples = 0
        self.idx_map = []
        for env_idx, env_tup in enumerate(envs):
            env, num_trajectories = env_tup
            env_path = os.path.join(self.base_path, env, self.data_fast)
            p_paths = sorted([os.path.join(env_path, P) for P in os.listdir(env_path)])
            p_paths = p_paths[:num_trajectories]
            self.data[env] = []
            for p_idx, p_path in enumerate(p_paths):
                p_data = {}
                
                # Load the images:
                if self.load_left_img or (self.load_next_img and self.camera_type == 'left'):
                    left_img_files = sorted([os.path.join(p_path, f'image_left', f) for f in os.listdir(os.path.join(p_path, f'image_left')) if f.endswith(f'left.png')])
                    p_data['left_img_files'] = left_img_files
                if self.load_right_img or (self.load_next_img and self.camera_type == 'right'):
                    right_img_files = sorted([os.path.join(p_path, f'image_right', f) for f in os.listdir(os.path.join(p_path, f'image_right')) if f.endswith(f'right.png')])
                    p_data['right_img_files'] = right_img_files
                
                # Load the motion data (except the last)
                motion_file_path = os.path.join(p_path, 'motion_%s.npy' % self.camera_type)
                motion_data = np.load(motion_file_path)
                motion_data = motion_data[:-1]
                p_data['motion_data'] = motion_data
                self.idx_map += [[env_idx, p_idx, data_idx] for data_idx in range(len(motion_data))]
                self.num_samples += len(motion_data)

                # Depth and flow file paths (except the last)
                if self.load_depth or self.load_disparity:
                    depth_files = sorted([os.path.join(p_path, 'depth_%s' % self.camera_type, f) 
                                          for f in os.listdir(os.path.join(p_path, f'depth_%s' % self.camera_type))
                                          if f.endswith('%s_depth.png' % self.camera_type)])
                    # import pdb;pdb.set_trace()
                    depth_files = depth_files[:-1]
                    # print(len(depth_files))
                    p_data['depth_files'] = depth_files
                    

                if self.load_flow:
                    flow_files = sorted([os.path.join(p_path, 'flow', f) for f in os.listdir(os.path.join(p_path, 'flow')) if f.endswith('_flow.png')])
                    # mask_files = sorted([os.path.join(p_path, 'flow', f) for f in os.listdir(os.path.join(p_path, 'flow')) if f.endswith('_mask.npy')])
                    # print(len(flow_files))
                    p_data['flow_files'] = flow_files
                
                self.data[env].append(p_data)                

    def __len__(self):
        # The length of the dataset is determined by the number of motion samples
        return self.num_samples

    def visualize_depth(self, idx, maxthresh = 50):
        # Load
        env_idx, p_idx, data_idx = self.idx_map[idx]
        env = self.envs[env_idx][0]
        
        depth_path = self.data[env][p_idx]['depth_files'][data_idx]
            
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, "Error loading depth {}".format(depth_path)
        
        depth = depth_rgba.view("<f4")
        
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
        env_idx, p_idx, data_idx = self.idx_map[idx]
        env = self.envs[env_idx][0]
        
        flow_path = self.data[env][p_idx]['flow_files'][data_idx]
        
        flow16 = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(flow_path)
        
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0
        
        mask8 = flow16[:,:,2].astype(np.uint8)
        
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

        if mask:
            mask8 = mask8 > 0
            rgb[mask8] = np.array([0, 0 ,0], dtype=np.uint8)
        
        return rgb
        
    def visualize(self, idx, depth_maxthresh = 50, flow_maxF=500.0, flow_n=8, 
                  flow_mask=True, flow_hueMax=179, flow_angShift=0.0):
        # Get depth and flow visualizations
        depthvis = self.visualize_depth(idx, maxthresh=depth_maxthresh)
        rgb = self.visualize_flow(idx, maxF=flow_maxF, n=flow_n, mask=flow_mask, 
                                  hueMax=flow_hueMax, angShift=flow_angShift)
        # Get image
        env_idx, p_idx, data_idx = self.idx_map[idx]
        env = self.envs[env_idx][0]
        left_img_path = self.data[env][p_idx]['left_img_files'][data_idx]
        img = plt.imread(left_img_path)
        
        # Plot
        fig, axs = plt.subplots(1,3,clear=True, figsize=(12,7))
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[1].imshow(depthvis)
        axs[1].set_title('Depth')
        axs[2].imshow(rgb)
        axs[2].set_title('Flow')
        fig.savefig("/data1/datasets/msathena/workspace/stereo/visualizations/full_%06d_%s_%03d_%06d" 
                    % (idx,env,p_idx,data_idx))
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
    
    def get_by_env(self, env, p_idx, data_idx):
        data = {}
        
        # Load images
        if self.load_left_img:
            left_img_path = self.data[env][p_idx]['left_img_files'][data_idx]
            data['left_img'] = self.load_img(left_img_path)
        if self.load_right_img:
            right_img_path = self.data[env][p_idx]['right_img_files'][data_idx]
            data['right_img'] = self.load_img(right_img_path)
        if self.load_next_img:
            next_img_path = self.data[env][p_idx][f'{self.camera_type}_img_files'][data_idx+1]
            data['next_img'] = self.load_img(next_img_path)
        
        # Load motion data and convert to float
        motion = self.data[env][p_idx]['motion_data'][data_idx].astype(np.float32)
        data['motion'] = torch.from_numpy(motion)
        # Batch size, [tx, ty, tz, r...]

        # Load and convert depth data to float
        if self.load_depth or self.load_disparity:
            depth_path = self.data[env][p_idx]['depth_files'][data_idx]
            
            depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            assert depth_rgba is not None, "Error loading depth {}".format(depth_path)
            
            depth = depth_rgba.view("<f4")
            
            h, w = depth.shape[:2]
            depth = torch.from_numpy(depth.transpose(2,0,1))
            # data['depth'] = trf.resize(depth, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), 
            #                            antialias=True)
            # data['depth'] = trf.interpolate(depth, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), mode= 'linear',
            #                            align_corners=True)
            if self.load_depth:
                data['depth'] = depth
            if self.load_disparity:
                data['disparity'] = 80.0 / (depth + 10e-6)
            # Batch size, 1 channel, H, W

        # Load and convert optical flow data to float
        if self.load_flow:
            flow_path = self.data[env][p_idx]['flow_files'][data_idx]
            
            flow16 = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
            assert flow16 is not None, "Error loading flow {}".format(flow_path)
            
            flow32 = flow16[:,:,:2].astype(np.float32)
            flow32 = (flow32 - 32768) / 64.0
            
            mask8 = flow16[:,:,2].astype(np.uint8)
            # print(np.unique(mask8))
            mask8 = mask8 > 0
            flow32[mask8] = np.array([0, 0], dtype=np.float32)
            
            h, w = flow32.shape[:2] 
            flow32 = torch.from_numpy(flow32.transpose(2,0,1))
            # data['flow'] = trf.resize(flow32, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), 
            #                           antialias=True)
            # data['flow'] = trf.interpolate(flow32, (int(h*self.flow_depth_scale), int(w*self.flow_depth_scale)), 
            #                           align_corners=True, antialias=True)
            data['flow'] = flow32
            # Batch size, 2 channels, H, W

        return data
    
    def __getitem__(self, idx):
        env_idx, p_idx, data_idx = self.idx_map[idx]
        env = self.envs[env_idx][0]
        return self.get_by_env(env, p_idx, data_idx)
        

if __name__ == "__main__":
    # Example usage
    base_path = '/project/learningvo/tartanair_v1_5'
    environments = [ # [Environment name, Number of Trajectories to Load]
                    #  ['abandonedfactory', 2],
                    #  ['abandonedfactory_night', 1],
                    #  ['house',-1]
                    #  ['oldtown',-1],
                    #  ['house_dist0',-1],
                    #  ['seasidetown',-1],
                    #  ['amusement',-1],
                    #  ['japanesealley',-1],
                    #  ['seasonsforest',-1],
                    #  ['carwelding',-1],
                    #  ['neighborhood',-1],
                    #  ['seasonsforest_winter',-1],
                     ['endofworld',-1],
                    #  ['occ',-1],
                    #  ['slaughter',-1],
                    #  ['gascola',-1],
                    #  ['ocean',-1],
                    #  ['soulcity',-1],
                    #  ['hongkongalley',-1],
                    #  ['office',-1],
                    #  ['westerndesert',-1],
                    #  ['hospital',-1],
                    #  ['office2',-1],
                    ]
    dataset = CustomDataset(base_path, envs=environments, load_depth=True, load_disparity=True, load_flow=True, 
                            load_left_img=True, load_right_img=True, load_next_img=True, left_camera=True)
    print(dataset[0]['left_img'].shape, dataset[0]['right_img'].shape, dataset[0]['next_img'].shape, 
          dataset[0]['motion'].shape, dataset[0]['depth'].shape, dataset[0]['disparity'].shape, dataset[0]['flow'].shape)
    for i in np.random.choice(len(dataset), 10, replace=False):
        dataset.visualize(i)
    
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
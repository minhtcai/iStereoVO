import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--working-dir', default='./',
                        help='working directory')

    parser.add_argument('--exp-prefix', default='stereonly_1_',
                        help='exp prefix')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help='decay of learning rate.')

    parser.add_argument('--normalize-output', type=float, default=1,
                        help='normalize the output (default: 1)')

    parser.add_argument('--stereo-norm', type=float, default=0.02,
                        help='normalize the output (default: 0.02)')

    parser.add_argument('--flow-norm', type=float, default=0.05,
                        help='normalize the output (default: 0.05)')

    parser.add_argument('--pose-trans-norm', type=float, default=0.13,
                        help='normalize the translation (default: 0.13)')

    parser.add_argument('--pose-rot-norm', type=float, default=0.013,
                        help='normalize the rotation (default: 0.013)')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 1)')

    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='test batch size (default: 1)')

    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')

    parser.add_argument('--train-step', type=int, default=40000,
                        help='number of interactions in total (default: 1000000)')

    parser.add_argument('--snapshot', type=int, default=1000,
                        help='snapshot (default: 100000)')

    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')

    parser.add_argument('--image-height', type=int, default=480,
                        help='image height (default: 320)')

    parser.add_argument('--image-scale', type=float, default=1.0,
                        help='image is divided by this scale  (default: 1.0)')

    parser.add_argument('--hsv-rand', type=float, default=0.0,
                        help='augment rand-hsv by adding different hsv to a set of images (default: 0.0)')

    parser.add_argument('--rand-blur', type=float, default=0.0,
                        help='randomly load blur image for training (default: 0.0)')

    parser.add_argument('--warmup', type=int, default=0,
                        help='warmup first few iterations (default: 50000)')

    parser.add_argument('--train-data-type', default='tartan',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--train-data-balence', default='1',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--test-data-type', default='tartan',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load pretrained model (default: False)')

    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')

    parser.add_argument('--test', action='store_true', default=False,
                        help='test (default: False)')

    parser.add_argument('--test-num', type=int, default=10,
                        help='test (default: 10)')

    parser.add_argument('--test-save-image', action='store_true', default=False,
                        help='test output image to ./testimg (default: False)')

    parser.add_argument('--test-save-disp', action='store_true', default=False,
                        help='test output disparity to test-output-dir (default: False)')

    parser.add_argument('--test-interval', type=int, default=10,
                        help='The test interval.')

    parser.add_argument('--use-int-plotter', action='store_true', default=False,
                        help='Enable cluster mode.')

    parser.add_argument('--print-interval', type=int, default=1,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--plot-interval', type=int, default=100,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--network', type=int, default=1,
                        help='network structure')

    parser.add_argument('--act-fun',default='relu',
                        help='activation function')

    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='batch normalization (default: False)')

    parser.add_argument('--no-data-augment', action='store_true', default=False,
                        help='no data augmentation (default: False)')

    parser.add_argument('--multi-gpu', type=int, default=2,
                        help='multiple gpus numbers (default: False)')

    parser.add_argument('--platform', default='local',
                        help='deal with different data root directory in dataloader, could be one of local, cluster, azure (default: "local")')

    # VO-Flow
    parser.add_argument('--norm-trans-loss', action='store_true', default=False,
                        help='use cosine simularity to calculate the trans loss (default: False)')

    parser.add_argument('--linear-norm-trans-loss', action='store_true', default=False,
                        help='use normalized L1 loss to calculate the trans loss (default: False)')

    parser.add_argument('--two-heads', action='store_true', default=False,
                        help='(deprecated) separate translation and rotation heads (default: False)')

    parser.add_argument('--two-heads2', action='store_true', default=False,
                        help='(deprecated) separate translation and rotation heads (default: False)')

    parser.add_argument('--resvo-config', type=int, default=0,
                        help='configuration for resvo (default: 0)')

    parser.add_argument('--downscale-flow', action='store_true', default=True,
                        help='when resvo, use 1/4 flow size, which is the size output by pwc')

    parser.add_argument('--azure', action='store_true', default=False,
                        help='(deprecate - platform) training on azure (default: False)')

    parser.add_argument('--load-from-e2e', action='store_true', default=False,
                        help='load pose model from end2end flow-pose model')

    parser.add_argument('--test-traj-dir', default='',
                        help='test trajectory folder for flowvo (default: "")')

    parser.add_argument('--test-output-dir', default='./',
                        help='output dir of the posefile and media files for flowvo (default: "")')

    parser.add_argument('--traj-pose-file', default='',
                        help='test trajectory gt pose file (default: "")')
   
    parser.add_argument('--test-worker-num', type=int, default=1,
                        help='data loader worker number for testing set (default: 10)')

    parser.add_argument('--intrinsic-layer', action='store_true', default=True,
                        help='add two layers as intrinsic input')

    parser.add_argument('--random-crop', type=int, default=0,
                        help='crop and resize the flow w/ intrinsic layers')

    parser.add_argument('--random-crop-center', action='store_true', default=False,
                        help='(Deprecated) random crop at the center of the image')

    parser.add_argument('--random-intrinsic', type=float, default=0,
                        help='(Deprecated) similar with random-crop but cover contineous intrinsic values')

    parser.add_argument('--random-resize-factor', type=float, default=0,
                        help='Randomly crop and resize the image, in replace of the random-intrinsic parameter')

    parser.add_argument('--random-rotate-rightimg', type=float, default=0.0,
                        help='for stereo matching, randomly rotate the right image by a small angle')

    parser.add_argument('--fix-ratio', action='store_true', default=False,
                        help='(Deprecated) fix resize ratio')

    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')

    parser.add_argument('--intrinsic-kitti', action='store_true', default=False,
                        help='(Deprecated) transform tartan image to kitti to match the intrinsic value (default: False)')

    parser.add_argument('--load-flow-model', action='store_true', default=False,
                        help='In end2end training, load pretrained flow model')

    parser.add_argument('--flow-model', default='pwc_net.pth.tar',
                        help='In end2end training, the name of pretrained flow model')

    parser.add_argument('--load-pose-model', action='store_true', default=False,
                        help='In end2end training, load pretrained pose model')

    parser.add_argument('--pose-model', default='',
                        help='In end2end training, the name of pretrained pose model')

    parser.add_argument('--small2', action='store_true', default=False,
                        help='For flow, smaller PSM feature extractor (default: False)')

    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')

    parser.add_argument('--realsense', action='store_true', default=False,
                        help='realsense test (default: False)')

    parser.add_argument('--test-traj', action='store_true', default=False,
                        help='test trajectory from --test-traj-dir (default: False)')

    parser.add_argument('--save-trajs', action='store_true', default=False,
                        help='save trajectories as animation files (default: False)')

    parser.add_argument('--scale-w', type=float, default=1.0,
                        help='scale_w for the kitti transform')

    parser.add_argument('--no-gt', action='store_true', default=False,
                        help='test wo/ gt motion/disp/flow (default: False)')

    parser.add_argument('--gt-pose-file', default='',
                        help='trajectory GT pose file using for visualization when testing )')

    parser.add_argument('--flow-thresh', type=float, default=10.0,
                        help='in end2end training, skip the sample if the flow error is bigger than thresh (default: 10.0)')

    parser.add_argument('--train-flow', action='store_true', default=False,
                        help='step 1, only train flow (default: False)')

    parser.add_argument('--train-vo', action='store_true', default=False,
                        help='step 2, no extra flow and train vo e2e (default: False)')

    parser.add_argument('--fix-flow', action='store_true', default=False,
                        help='step 2, when train-vo=Ture, fix flownet and train vo (default: False)')

    parser.add_argument('--lambda-flow', type=float, default=1.0,
                        help='lambda vo in the loss function (default: 1.0)')

    parser.add_argument('--flow-file', default='',
                        help='txt file specify the training data (default: "")')

    parser.add_argument('--flow-data-balence', default='1',
                        help='specify the data balence scale e.g. 1,5,10 ')

    parser.add_argument('--flow-data-type', default='tartan',
                        help='sintel / flying / tartan')

    parser.add_argument('--lr-flow', type=float, default=1e-4,
                        help='learning rate for flow in e2e training (default: 1e-4)')

    parser.add_argument('--vo-gt-flow', type=float, default=0.0,
                        help='when e2e, use GT flow instead of e2e with this probability')

    parser.add_argument('--train-spec-file', default='',
                        help='a yaml file specify the parameters of data-cacher and data-loader (default: "")')

    parser.add_argument('--test-spec-file', default='',
                        help='a yaml file specify the parameters of data-cacher and data-loader (default: "")')

    parser.add_argument('--train-spec-file2', default='',
                        help='a yaml file specify the parameters of data-cacher and data-loader (default: "")')

    parser.add_argument('--test-spec-file2', default='',
                        help='a yaml file specify the parameters of data-cacher and data-loader (default: "")')

    parser.add_argument('--transform-spec-file', default='',
                        help='a yaml file specify the parameters of data augmentation (default: "")')

    parser.add_argument('--flow-from-depth', action='store_true', default=False,
                        help='calculate flow from depth and motion (default: False)')

    parser.add_argument('--flow-from-disp', action='store_true', default=False,
                        help='calculate flow from disparity and motion (default: False)')

    # stereo-vo-e2e
    parser.add_argument('--fix-stereo', action='store_true', default=False,
                        help='when train-stereo=Ture, fix stereonet and train vo (default: False)')

    parser.add_argument('--load-stereo-model', action='store_true', default=True,
                        help='Flow load pretrained stereo model (default: False)')

    parser.add_argument('--stereo-model', default='stereonly_1_stereoflow_20000.pkl',
                        help='In end2end training, the name of pretrained stereo model')

    parser.add_argument('--train-stereo', action='store_true', default=False,
                        help='train stereo (default: False)')

    parser.add_argument('--lambda-stereo', type=float, default=1.0,
                        help='lambda stereo in the loss function (default: 1.0)')

    parser.add_argument('--stereo-file', default='',
                        help='txt file specify the training data (default: "")')

    parser.add_argument('--stereo-data-balence', default='1',
                        help='specify the data balence scale e.g. 1,5,10 ')

    parser.add_argument('--stereo-data-type', default='tartan',
                        help='sintel / flying / tartan')

    parser.add_argument('--lr-stereo', type=float, default=1e-4,
                        help='learning rate for stereo in e2e training (default: 1e-4)')

    # deprecated
    # parser.add_argument('--stereo-baseline-x-focal', type=float, default=80.0,
    #                     help='baseline x focal length (default: 80.0 for tartanair)')

    parser.add_argument('--scale-disp', type=float, default=1.0,
                        help='scale disp input, and scale back motion output, 2.0: scale things closer (default: 1.0 no scale)')

    parser.add_argument('--random-scale-disp-motion', action='store_true', default=False,
                        help='scale disp input, and scale back motion output as data augmentation')

    parser.add_argument('--random-static', type=float, default=0.0,
                        help='randomly turn a sample to static motion')

    parser.add_argument('--auto-dist-target', type=float, default=0.0,
                        help='scale the depth and motion frame by frame')

    # Flow mask.
    parser.add_argument('--use-train-flow-mask', action='store_true', default=False, 
        help="Set this flag to enable the mask for optical flow data during training. If no mask file is available, an exception will be raised. ")

    parser.add_argument('--use-test-flow-mask', action='store_true', default=False, 
        help="Set this flag to enable the mask for optical flow data during testing. If no mask file is available, an exception will be raised. ")

    parser.add_argument('--compressed', action='store_true', default=False, 
        help="Load the data that has been compressed. ")

    parser.add_argument('--uncertainty', action='store_true', default=False,
                    help='output uncertainty estimation for optical flow and stereo (default: False)')

    parser.add_argument('--cacher-max-repeat', type=int, default=3,
                        help='Input length of the IMU sequence (default: 3)')


    # IMU net
    parser.add_argument('--imu-input-len', type=int, default=100,
                        help='Input length of the IMU sequence (default: 100)')

    parser.add_argument('--imu-noise', type=float, default=0.0,
        help="Add noise to IMU data. ")

    parser.add_argument('--imu-stride', type=int, default=1,
                        help='Stride of the dataloader (default: 1)')

    parser.add_argument('--frame-skip', type=int, default=0,
                        help='skip frames (default: 0)')

    parser.add_argument('--six-dof', action='store_true', default=False,
                    help='use six-dof representation for angle/orientation (default: False)')

    parser.add_argument('--imu-init', action='store_true', default=False,
                    help='use init vel and angle in the imu integration network (default: False)')

    parser.add_argument('--supervise-vo', type=float, default=0.0,
                    help='use up-to-scale loss for vo network (default: False)')

    parser.add_argument('--supervise-imu', type=float, default=0.0,
                    help='use motion loss for imu network (default: False)')

    parser.add_argument('--load-imu-model', action='store_true', default=False,
                        help='In vio training, load pretrained imu model')

    parser.add_argument('--imu-model', default='',
                        help='In vio training, the name of pretrained imu model')

    parser.add_argument('--no-imu-norm', action='store_true', default=False,
                        help='Do not do normalization for imu data')

    parser.add_argument('--network-scale', type=int, default=1,
                        help='network size factor (default: 1)')

    parser.add_argument('--gravity-regress-frame', type=int, default=-1,
                        help='gt frame number in vio-init, regress all frames if set to -2 (default: -1)')

    # jointly train the flow and steteo network

    parser.add_argument('--train-data-type2', default='tartan',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--train-data-balence2', default='1',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--test-data-type2', default='tartan',
                        help='sceneflow / kitti / tartan')

    args = parser.parse_args()

    return args

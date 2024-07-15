import argparse

parser = argparse.ArgumentParser(description='U2Fusion')
parser.add_argument('--debug', action='store_true', help='Enables debug mode')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--validation', action='store_true', help='set this option to validate after training')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
# parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--train_test', action='store_true', help='test after training each epoch')

# Data specifications
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--lr', type=int, default=1e-5, help='learning rate')
parser.add_argument('--ep', type=int, default=None, help='test epoch')
parser.add_argument('--log_dir', type=str, default='./train_log', help='output event file path')
parser.add_argument('--dir_train', type=str, default='dataset/train_data/', help='training dataset directory')
parser.add_argument('--dir_val', type=str, default='dataset/val_data/', help='validation dataset directory')
parser.add_argument('--dir_test', type=str, default='pre_processing_test/', help='test dataset directory')
parser.add_argument('--model_path', type=str, default='model/', help='trained model directory')
parser.add_argument('--model', type=str, default='model.pth', help='model name')
parser.add_argument('--ext', type=str, default='.jpg', help='extension of image files')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='input patch size')
parser.add_argument('--save_dir', type=str, default='results/noisy_result/', help='test results directory')


# Unet Data specifications
parser.add_argument('--unet_model_path', type=str, default='model_denoise/', help='trained model directory')
parser.add_argument('--unet_model', type=str, default='model.pth', help='model name')
parser.add_argument('--unet_dir_train', type=str, default='dataset/denoise_train_data/', help='UNet training dataset directory')
parser.add_argument('--unet_dir_test', type=str, default='dataset/denoise_test_data/', help='UNet test dataset directory')
parser.add_argument('--unet_log_dir', type=str, default='./unet_train_log/', help='output event file path')
parser.add_argument('--denoised_save_dir', type=str, default='results/final_result/', help='test results directory')
parser.add_argument('--learingrate', type=int, default=3e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--patchsize', type=int, default=256, help='input UNet patch size')
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--Lambda3", type=float, default=1.0)
parser.add_argument("--Lambda4", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument("--noisetype", type=str, default="poisson40")


# Model specifications
parser.add_argument('--in_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--out_channels', type=int, default=3, help='number of output channels')
parser.add_argument('--num_features', type=int, default=44, help='number of features')
parser.add_argument('--growth', type=int, default=44, help='channel growth in dense blocks')
parser.add_argument('--num_layers', type=int, default=5, help='number of dense layers')
parser.add_argument('--act_type', type=str, default='prelu', help='type of activation function')
parser.add_argument('--eval', action='store_true', help='evaluate the test results')

args = parser.parse_args()

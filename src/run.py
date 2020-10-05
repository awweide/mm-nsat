import gan as gan
import argparse

def parse_args():
    desc = 'GAN cost function comparison'
    parser = argparse.ArgumentParser(description=desc)
    #General
    parser.add_argument('--epochs',type=int,default=1,help='Number of epochs to train model')
    parser.add_argument('--gpus',type=str,default='3',help='Commaseparated indices of GPUs to use (0-ind), NOT number of GPUs')
    parser.add_argument('--dataset',type=str,default='mnist',help='Dataset to use (mnist,cifar,mnist1k,cats128,cats256,ffhq32,...,ffhq1024)')
    parser.add_argument('--runs_n',type=int,default=10,help='Number of independent runs')
    parser.add_argument('--output_folder',type=str,default='debug',help='Folder to save outputs, index for each run is appended')
    parser.add_argument('--batch_size',type=int,default=128,help='Batch size')

    #Evaluation
    parser.add_argument('--fid_n',type=int,default=50000,help='Samples used to calculate FID')
    parser.add_argument('--fid_act',type=int,default=1,help='Cache activations rather than images: much more memory efficient at high resolutions')
    parser.add_argument('--eval_n',type=int,default=10,help='Number of evaluation steps (time intensive)')
    parser.add_argument('--eval_skip',type=int,default=0,help='Skip FID and CDD during evals (time saver)')

    #Optimizers
    parser.add_argument('--opt',type=str,default='adam',help='Optimizer for gradient descent (adam, sgd)')
    parser.add_argument('--g_lr',type=float,default=1e-4,help='Generator learning rate')
    parser.add_argument('--g_beta1',type=float,default=0.5,help='Generator beta1 parameter for Adam optimizer')
    parser.add_argument('--g_beta2',type=float,default=0.999,help='Generator beta2 parameter for Adam optimizer')
    parser.add_argument('--g_adameps',type=float,default=1e-8,help='Generator eps-hat parameter for Adam optimizer')
    parser.add_argument('--g_adamreset',type=int,default=0,help='Generator frequency of resetting Adam optimizer (0 disables)')
    parser.add_argument('--d_lr',type=float,default=1e-4,help='Discriminator learning rate')
    parser.add_argument('--d_beta1',type=float,default=0.5,help='Discriminator beta1 parameter for Adam optimizer')
    parser.add_argument('--d_beta2',type=float,default=0.999,help='Discriminator beta2 parameter for Adam optimizer')
    parser.add_argument('--d_adameps',type=float,default=1e-8,help='Discriminator eps-hat parameter for Adam optimizer')
    
    #Costs
    parser.add_argument('--d_cost',type=str,default='ns',help='Cost function for training (see costs.py)')
    parser.add_argument('--g_cost',type=str,default='ns',help='Cost function for training (see costs.py)')
    parser.add_argument('--g_renorm',type=str,default='none',help='Renorm gradient magnitude (none, frac, nsat, unit)')
    #These general purpose parameters are reused for different purposes by different costs, controlling either strength of gradient penalty or linear interpolation weighting and so on ...
    parser.add_argument('--g_cost_parameter',type=float,default=0,help='Cost function parameter')
    parser.add_argument('--d_cost_parameter',type=float,default=0,help='Cost function parameter')

    #Networks - note that many networks ignore or override arguments!
    parser.add_argument('--d_net',type=str,default='sn',help='Discriminator network (see nets.py)')
    parser.add_argument('--g_net',type=str,default='sn',help='Generator network (see nets.py)')
    parser.add_argument('--m_dim',type=int,default=64,help='Model complexity factor')
    parser.add_argument('--z_dim',type=int,default=128,help='Latent space (z) dimension')
    parser.add_argument('--g_layers',type=int,default=4,help='Total layers in dense generator')
    parser.add_argument('--d_layers',type=int,default=4,help='Total layers in dense discriminator')

    parser.add_argument('--g_bn',type=int,default=0,help='Batch norm in G')
    parser.add_argument('--d_bn',type=int,default=0,help='Batch norm in D')
    parser.add_argument('--g_res',type=int,default=0,help='Residual connections in G')
    parser.add_argument('--d_res',type=int,default=0,help='Residual connections in D')
    parser.add_argument('--g_selu',type=int,default=0,help='Use self-normalizing-network architecture for G')
    parser.add_argument('--d_selu',type=int,default=0,help='Use self-normalizing-network architecture for D')
    parser.add_argument('--g_sn',type=int,default=0,help='Use spectral normalization for G')
    parser.add_argument('--d_sn',type=int,default=0,help='Use spectral normalization for D')
    parser.add_argument('--g_sa',type=int,default=0,help='Use self attention for G')
    parser.add_argument('--d_sa',type=int,default=0,help='Use self attention for D')

    #FFHQ NCHW <->NHWC handling - important to avoid excessive and expensive tf.transpose calls
    parser.add_argument('--data_nchw',type=int,default=0,help='Networks channel first')
    parser.add_argument('--net_nchw',type=int,default=0,help='Networks channel first')

    return parser.parse_args()

def main():
    a = parse_args()
    of = a.output_folder
    for i in xrange(a.runs_n):
        vars(a)['output_folder'] = 'out/' + of + '_' + str(i)
        gan.main(a)

if __name__ == '__main__': main()

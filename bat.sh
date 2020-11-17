##Real datasets: python src/run.py
##See example configs below or src/run.py for options. Default args only for debugging.
##Examples assume 4 GPUs exposed to TensorFlow

##Example 3-layer fully connected MNIST runs (metrics disabled: re-enabling requires Inception and MNIST classifier networks: see line 16)
##Outputs saved in out/<output_folder>_<run#>
#python src/run.py --gpus='0' --output_folder=mnist_fc3_mm --g_cost_parameter=0.00 --g_renorm=none --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 &
#python src/run.py --gpus='1' --output_folder=mnist_fc3_ns --g_cost_parameter=0.00 --g_renorm=none --g_cost=ns --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 &
#python src/run.py --gpus='2' --output_folder=mnist_fc3_mmunit --g_cost_parameter=0.00 --g_renorm=unit --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 &
#python src/run.py --gpus='3' --output_folder=mnist_fc3_mmnsat --g_cost_parameter=1.00 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 &

##For explicitly gradient rescaled version of mmnsat:
#python src/run.py --gpus='3' --output_folder=mnist_fc3_mmnsat --g_cost_parameter=0.00 --g_renorm=frac --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 &

##Uncomment following line to perform all the following setup steps in succession:
#python src/download.py && python src/mnist.py && python src/cifar.py

##CIFAR-10 dataset expected at data/cifar-10/data_batch_1 etc
##Available at https://www.cs.toronto.edu/~kriz/cifar.html
##Trained inception network for FID expected at 'data/fid_inception_model/frozen_inception_v1_2015_12_05.tar.gz'
##Available at http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz
##Classifiers must be trained for the D_JS(CD) metric
##Train classifiers - saves to correct destination when run from root folder
##Training the CIFAR-10 classifier takes some time
#python src/mnist.py
#python src/cifar.py

##Example Conv-4-sn CIFAR runs with metrics for linear combinations
#python src/run.py --gpus='0' --dataset=cifar --output_folder=cifar_conv4sn_nsmmnsat0.00 --g_cost_parameter=0.00 --g_cost=ns_mmnsat --d_cost=ns --d_sn=1 --g_net=conv32sn --d_net=conv32sn --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='1' --dataset=cifar --output_folder=cifar_conv4sn_nsmmnsat0.33 --g_cost_parameter=0.33 --g_cost=ns_mmnsat --d_cost=ns --d_sn=1 --g_net=conv32sn --d_net=conv32sn --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='2' --dataset=cifar --output_folder=cifar_conv4sn_nsmmnsat0.67 --g_cost_parameter=0.67 --g_cost=ns_mmnsat --d_cost=ns --d_sn=1 --g_net=conv32sn --d_net=conv32sn --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='3' --dataset=cifar --output_folder=cifar_conv4sn_nsmmnsat1.00 --g_cost_parameter=1.00 --g_cost=ns_mmnsat --d_cost=ns --d_sn=1 --g_net=conv32sn --d_net=conv32sn --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &

##Convenient summary of (FID,ClassDistributions) from above experiment
#python src/mread_js_fid.py out/cifar_conv4sn_cvpr_nsmmnsat0.00_ ## or 0.33, 0.67, 1.00


##Setup for CAT and FFHQ datasets not included
##CAT expects pre-processed .jpg files, FFHQ expects multi-resolution tfrecords files, see src/data.py

##Example conv-9 FFHQ512 MM-nsat
#python src/run.py --gpus='0' --output_folder=ffhq512_conv_e1000_mmnsat --g_cost_parameter=1.00 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=convn --d_net=convn --dataset=ffhq512 --batch_size=64 --epochs=1000 --eval_n=20 --fid_n=50000 --m_dim=32 --runs_n=3 --data_nchw=1 --net_nchw=1 &
#python src/run.py --gpus='1' --output_folder=ffhq512_conv_e1000_ns --g_cost_parameter=0.00 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=convn --d_net=convn --dataset=ffhq512 --batch_size=64 --epochs=1000 --eval_n=20 --fid_n=50000 --m_dim=32 --runs_n=3 --data_nchw=1 --net_nchw=1 &

###Example DCG-bnsn CAT128 - fid_n=5000 due to dataset size
#python src/run.py --gpus='0' --output_folder=cat128_dcgbnsn_e1000_ns --g_cost_parameter=0.00 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=dcg --d_net=dcg --d_sn=1 --d_sa=0 --g_sa=0 --d_bn=1 --g_bn=1 --dataset=cats128 --batch_size=64 --epochs=1000 --fid_n=5000 &
#python src/run.py --gpus='1' --output_folder=cat128_dcgbnsn_e1000_ls --g_cost_parameter=0.00 --g_renorm=none --g_cost=sig_ls --d_cost=sig_ls --g_net=dcg --d_net=dcg --d_sn=1 --d_sa=0 --g_sa=0 --d_bn=1 --g_bn=1 --dataset=cats128 --batch_size=64 --epochs=1000 --fid_n=5000 &
#python src/run.py --gpus='2' --output_folder=cat128_dcgbnsn_e1000_hinge --g_cost_parameter=0.00 --g_renorm=none --g_cost=hinge --d_cost=hinge --g_net=dcg --d_net=dcg --d_sn=1 --d_sa=0 --g_sa=0 --d_bn=1 --g_bn=1 --dataset=cats128 --batch_size=64 --epochs=1000 --fid_n=5000 &
#python src/run.py --gpus='3' --output_folder=cat128_dcgbnsn_e1000_mmf --g_cost_parameter=1.00 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=dcg --d_net=dcg --d_sn=1 --d_sa=0 --g_sa=0 --d_bn=1 --g_bn=1 --dataset=cats128 --batch_size=64 --epochs=1000 --fid_n=5000 &

###Example CIFAR-10 Miyato SN-GAN replication
#python src/run.py --gpus='0,1' --output_folder=cifar_rep_e1000_ns --g_cost_parameter=0.00 --d_cost_parameter=0.0 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=rep --d_net=rep --d_sn=1 --g_bn=1 --dataset=cifar --batch_size=256 --g_lr=0.0002 --d_lr=0.0002 --fid_n=50000 --epochs=1000 --eval_n=20 --runs_n=10 --eval_skip=0 &
#python src/run.py --gpus='2,3' --output_folder=cifar_rep_e1000_mmnsat --g_cost_parameter=1.00 --d_cost_parameter=0.0 --g_renorm=none --g_cost=ns_mmnsat --d_cost=ns --g_net=rep --d_net=rep --d_sn=1 --g_bn=1 --dataset=cifar --batch_size=256 --g_lr=0.0002 --d_lr=0.0002 --fid_n=50000 --epochs=1000 --eval_n=20 --runs_n=10 --eval_skip=0 &


##TOY PROBLEMS: python src/toy.py
##Parameter 1: 0: (1-a)NS + (0+a)MM, 1: (1-a)NS + (0+a)MM-nsat
##Parameter 2: Value of weighting factor a
##Parameter 3: Toy data (ring, spiral)
##Parameter 4: Output to out/<this folder>

##Examples for replicating paper results
##Mode frequencies are only printed to console
##Plots of distributions are saved to output folder: plots for one run included in repository under out/repo_toy_<...>
#python src/toy.py 0 0.0 ring toy_ring_ns
#python src/toy.py 0 1.0 ring toy_ring_mm
#python src/toy.py 0 0.0 spiral toy_spiral_ns
#python src/toy.py 0 1.0 spiral toy_spiral_mm

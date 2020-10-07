Prototype code for replicating core results from arXiv preprint https://arxiv.org/abs/2010.02035. Does not include StyleGAN experiments or setup instructions for the CAT and FFHQ datasets.

Main code contained in src/gan.py . Helper file src/run.py for feeding configuration for a run.

Toy data code in src/toy.py .

The CIFAR-10 dataset is expected in data/cifar-10 and not included in the repository. The MNIST dataset is automatically downloaded when needed.

Evaluation of metrics requires additional setup: training classifiers, using src/mnist.py and src/cifar.py, as well as including the pre-trained Inception network.

Bash script bat.sh can be edited as raw text and includes additional details, explanations and example configurations.

import os
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

def get_data(data_root="datasets/", batch_size_train=256, batch_size_test=512, **kwargs):
    # Create the folder where the data will be downloaded
    # exist_ok avoid exception if the folder already exists
    os.makedirs(data_root, exist_ok=True)

    # Next, we prepare a preprocessing pipeline which will be applied before feeding our data into the model
    # namely, ToTensor() transforms an image in a tensor and squishes its values between 0 and 1
    # Normalize(), instead, normalizes it w.r.t. the given mean and std. Since MNIST is grayscale,
    # we have only 1 color channel, hence, mean and std are considered as singleton tuples. If we had RGB
    # images, we should've written someting like Normalize((mean channel 1, mean channel 2, mean channel 3), ...)
    transforms = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])


    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = MNIST(data_root, train=True, transform=transforms,  download=True)
    testset = MNIST(data_root, train=False, transform=transforms,  download=True)

    #trainset.targets = one_hot(trainset.targets)
    #testset.targets = one_hot(testset.targets)

    # We feed our datasets into DataLoaders, which automatically manage the split into batches for SGD
    # shuffle indicates whether the data needs to be shuffled before the creation of batches
    # it's an overhead, but is necessary for a clean training, so we don't use it for the test set
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    testloader = DataLoader(trainset, batch_size=batch_size_test, shuffle=False)

    return trainloader, testloader


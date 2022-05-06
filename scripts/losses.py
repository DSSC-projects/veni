import torch
def multiclass_crossentropy(y, y_hat, classes = 10):
    return torch.sum(- y * torch.log(y_hat))

def mse(y,y_hat):
    return torch.sum((y- y_hat)**2)/y_hat.shape[0]
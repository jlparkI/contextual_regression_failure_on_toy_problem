import torch
from torch.nn import functional as F
import numpy as np

from sklearn.metrics import r2_score



class SNetUnit(torch.nn.Module):

    def __init__(self, in_features, out_features,
            kernel_size, stride, padding):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_features, 2 * out_features,
                kernel_size, stride, padding)
        self.bnorm = torch.nn.BatchNorm1d(in_features, affine=False)
        self.n_out = out_features

    def forward(self, x):
        x1 = self.conv(self.bnorm(x))
        return x1[:,:self.n_out,:] * F.sigmoid(x1[:,self.n_out:,:])




class SimpleNet(torch.nn.Module):

    #The model does fine without any dropout...so we never implemented it.
    def __init__(self, dropout = 0.0, rseed = 123):
        super().__init__()
        torch.manual_seed(rseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.f1 = torch.nn.Conv1d(3, 32, 3, 1, 1, 1)
   
        self.f2 = SNetUnit(32, 32, 3, 1, 1)
        self.f3 = SNetUnit(32, 32, 3, 1, 1)
        self.f4 = SNetUnit(32, 32, 3, 1, 1)
        self.f5 = SNetUnit(32, 32, 3, 1, 1)
        self.f6 = SNetUnit(32, 32, 3, 1, 1)

        self.o_layer = torch.nn.Linear(32,3)

        self.bnorm = torch.nn.BatchNorm1d(32)



    def forward(self, x, extract_rep = False):
        x1 = torch.transpose(x,1,2)
        x1 = F.elu(self.f1(x1))
        xo = self.f2(x1)
        xo = self.f3(xo)
        xo = self.f4(xo)
        xo = self.f5(xo)
        xo = self.f6(xo)
        xo = self.o_layer(torch.transpose(xo, 1, 2))
        if extract_rep:
            return xo
        return (x * xo).sum(dim=2).sum(dim=1).squeeze(-1)


def train_helper(model, trainx, trainy, valx, valy,
                n_epochs=100, batch_size=200, lr=5e-5):
    torch.manual_seed(123)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.cuda()

    train_r2, val_r2 = [], []

    for epoch in range(n_epochs):
        for i in range(0, trainx.shape[0], batch_size):
            x_mini = trainx[i:i+batch_size,...].float().cuda()
            y_mini = trainy[i:i+batch_size].float().cuda()
            y_pred = model(x_mini)
            loss = loss_fn(y_pred, y_mini)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch: {epoch}")
        train_preds, val_preds = [], []

        model.eval()
        with torch.no_grad():
            for i in range(0, trainx.shape[0], batch_size):
                train_preds.append(model(trainx[i:i+batch_size,...].float().cuda()).cpu().numpy())
            for i in range(0, valx.shape[0], batch_size):
                val_preds.append(model(valx[i:i+batch_size,...].float().cuda()).cpu().numpy())

        train_preds = np.concatenate(train_preds)
        val_preds = np.concatenate(val_preds)
        train_r2.append(r2_score(trainy.numpy(), train_preds))
        val_r2.append(r2_score(valy.numpy(), val_preds))
        model.train()
    
    model.eval()
    return train_r2, val_r2


def predict_helper(model, xin, batch_size = 500):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, xin.shape[0], batch_size):
            x_mini = xin[i:i+batch_size,...].float().cuda()
            all_preds.append(model(x_mini).cpu().numpy())

    return np.concatenate(all_preds)

def rep_helper(model, xin, batch_size=500):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, xin.shape[0], batch_size):
            x_mini = xin[i:i+batch_size,...].float().cuda()
            all_preds.append(model(x_mini, True).cpu().numpy())

    return np.vstack(all_preds)

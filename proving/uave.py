import torch
from linear_attention_transformer import ICLer
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(model,test_x,test_y,steps,batch_size,opt,criterion,N=20,test_every=10,save=None,sl=None):
    best_test=-1
    best_test_l=0
    for epoch in range(steps):
        train_loss=0
        W=generate_task_reg(d=dim,B=batch_size)
        X,Y=sample_bin(W,N,sigma=1)
        c=Y.size(-1)
        bloss=0
        for i in range(N):
            x=X[:,:i+1,]
            y=torch.cat([Y[:,:i,],torch.zeros(batch_size,1,c).cuda()],1)
            z=torch.cat((x,y),-1)
            pred=model(z)
            
            loss=criterion(pred[:,-1,],Y[:,i,])
            bloss+=loss/N
        opt.zero_grad()
        bloss.backward()
        opt.step()
        train_loss+=bloss.detach().item()
        #print('epoch:',epoch,'train_loss:',train_loss)
        if epoch %test_every== 0:
            test_loss=[]
            with torch.no_grad():
                X,Y=test_x,test_y
                for i in range(N):
                    x=X[:,:i+1,]
                    y=torch.cat([Y[:,:i,],torch.zeros(Y.size(0),1,c).cuda()],1)
                    z=torch.cat((x,y),-1)
                    pred=model(z)[:,:,-c:]
                    ac=acc(pred[:,-1],Y[:,i,])
                    test_loss.append(ac.cpu().item())
                t=np.mean(test_loss)
                if t>=best_test:
                    best_test=t
                    best_test_l=test_loss
                    if save!=None:
                        torch.save(model.state_dict(), save)
                    if sl!=None:
                        np.save(sl,test_loss)
                print('epoch:',epoch,'train_loss:',train_loss,'average test acc:',t,'  best average acc:',best_test,' best acc',best_test_l)
    return best_test_l


def mse(p,y):
    return torch.mean((p-y)*(p-y))

def loss_func(p,y):
    return F.cross_entropy(p,y)

def acc(p,y):
    correct=torch.argmax(p,-1)==torch.argmax(y,-1)
    return (correct.sum()/p.size(0))

def generate_task_reg(d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,d).normal_(mu,sigma)
    return W.cuda()

def sample_euc(W,N=1,sigma=1):
    B,d=W.size()
    W=W.unsqueeze(1).repeat(1,N,1)#bnd
    X=torch.normal(W,sigma).cuda()#bnd
    #Y=torch.sum(W+X,-1,True)
    #t=(W+X).view(-1,1,d)
    #Y=torch.bmm(t,t.transpose(-1,-2))
    #Y=Y.view(B,-1,1)
    Y=(W-X)*(W-X)
    Y=(torch.sum(Y,-1,True))

    return X,Y

def sample_sig(W,N=1,sigma=1):
    B,d=W.size()
    W=W.unsqueeze(1).repeat(1,N,1)#bnd
    X=torch.normal(W,sigma).cuda()#bnd
    #Y=torch.sum(W+X,-1,True)
    #t=(W+X).view(-1,1,d)
    #Y=torch.bmm(t,t.transpose(-1,-2))
    #Y=Y.view(B,-1,1)
    Y=(W-X)
    Y=torch.sigmoid(torch.sum(Y,-1,True))

    return X,Y

def sample_bin(W,N=1,sigma=1):
    B,d=W.size()
    W=W.unsqueeze(1).repeat(1,N,1)#bnd
    X=torch.normal(W,sigma).cuda()#bnd
    #Y=torch.sum(W+X,-1,True)
    #t=(W+X).view(-1,1,d)
    #Y=torch.bmm(t,t.transpose(-1,-2))
    #Y=Y.view(B,-1,1)

    d=(W-X).sum(-1)#bn
    n=(d<0).float().unsqueeze(-1)
    p=torch.ones_like(n)-n
    Y=torch.cat([n,p],-1).cuda()
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X,Y

dim=2
N=50
#train_num=2**15
C=2
loss=[]

model = ICLer(
    dim = dim+2*C,
    heads = 1,
    depth = 8,
    max_seq_len = 8192,
    label_dim=2*C,
    ff_list=[1,1,1,1,1,1,1,1],
    readout='linear',
).cuda()

#train_set=generate_task_reg(c=2,d=dim,B=train_num)
test_set=generate_task_reg(d=dim,B=1024)
tx,ty=sample_bin(test_set,N=N)

opt=torch.optim.Adam(model.parameters(),lr=1e-3)
l=train(model,tx,ty,100000,128,opt,loss_func,N=N,test_every=100,save='uave_dim2.pt',sl='uave_acc.npy')

def test(test_x,test_y):
    X,Y=test_x,test_y
    test_loss=[]
    for i in range(N):
        x=X[:,:i+1,]
        y=torch.cat([Y[:,:i,],torch.zeros(Y.size(0),1,c).cuda()],1)
        z=torch.cat((x,y),-1)
        pred=model(z)
        ac=acc(pred[:,-1],Y[:,i,])
        test_loss.append(ac.cpu().item())
    t=np.mean(test_loss)
    if t>=best_test:
        best_test=t
        best_test_l=test_loss
        if save!=None:
            torch.save(model.state_dict(), save)
    print('epoch:',epoch,'train_loss:',train_loss,'average test acc:',t,'  best average acc:',best_test,' best acc',best_test_l)




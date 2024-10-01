import torch
from linear_attention_transformer import ICLer
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def train(model,test_x,test_y,steps,batch_size,opt,criterion,N=20,test_every=10,save=None,sl=None,C=2):
    best_test=-1
    best_test_l=0
    for epoch in range(steps):
        train_loss=0
        W=generate_task_proto(c=C,d=dim,B=batch_size)
        X,Y=sample_proto(W,N,sigma=1)
        c=Y.size(-1)
        bloss=0
        for i in range(2,N):
            x=X[:,:i+1,]
            y=torch.cat([Y[:,:i,],torch.zeros(batch_size,1,c).cuda()],1)
            z=torch.cat((x,y),-1)
            pred=model(z)
            #print(pred.size())
            
            loss=criterion(pred[:,-1,],Y[:,i,])
            bloss+=loss/(N-2)
        opt.zero_grad()
        bloss.backward()
        opt.step()
        train_loss+=bloss.detach().item()
        #print('epoch:',epoch,'train_loss:',train_loss)
        if epoch %test_every== 0:
            test_loss=[]
            with torch.no_grad():
                X,Y=test_x,test_y
                for i in range(2,N):
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


def loss_func(p,y):
    return F.cross_entropy(p,y)

def acc(p,y):
    correct=torch.argmax(p,-1)==torch.argmax(y,-1)
    return (correct.sum()/p.size(0))

def generate_task_proto(c=2,d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,c,d).normal_(mu,sigma)
    return W.cuda()

'''def sample_proto(W,N=1,mu=0,sigma=1):
    B,c,d=W.size()
    q=int(N/c)
    t=W.unsqueeze(1).repeat(1,q,1,1)#bqcd
    X=torch.normal(t,sigma).cuda()#bqcd
    X=X.view(B,-1,d).unsqueeze(-2)#bn1d
    X=X[:,torch.randperm(N),:]
    W=W.unsqueeze(1)#b1cd
    Y=torch.zeros(B,N,c).cuda()
    dist=(W-X)*(W-X)#bncd
    dist=torch.sum(dist,-1)
    id=torch.argmin(dist,-1,True)
    #Y[id]=1
    id0=torch.arange(Y.size(0)).unsqueeze(1).repeat(1,Y.size(1)).view(-1)
    id1=torch.arange(Y.size(1)).unsqueeze(0).repeat(Y.size(0),1).view(-1)
    id=id.view(-1)
    Y[id0,id1,id]=1
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X.squeeze(-2),Y'''
def sample_proto(W,N=1,mu=0,sigma=1):
    '''B,c,d=W.size()
    q=int(N/c)
    t=W.unsqueeze(1).repeat(1,q,1,1)#bqcd
    X=torch.normal(t,sigma).cuda()#bqcd
    Y=torch.zeros(B,q,c,c).cuda()
    Y[:,:,0,0]=1
    Y[:,:,1,1]=1
    
    Y=Y.view(B,-1,c)
    X=X.view(B,-1,d)
    shuffle=torch.normal(0.0,1.0,(1,))
    if shuffle<0:
        idx=torch.arange(N,0,-1)-1
        X=X[:,idx]
        Y=Y[:,idx]
    idx=torch.randperm(N-2)+2
    X=torch.cat([X[:,0:2],X[:,idx]],1)
    Y=torch.cat([Y[:,0:2],Y[:,idx]],1)
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X.squeeze(-2),Y'''
    B,c,d=W.size()
    q=int(N/c)
    N=c*q
    t=W.unsqueeze(1).repeat(1,q,1,1)#bqcd
    X=torch.normal(t,sigma).cuda()#bqcd
    X=X.view(B,-1,d).unsqueeze(-2)#bn1d
    X=X[:,torch.randperm(N),:]
    W=W.unsqueeze(1)#b1cd
    Y=torch.zeros(B,N,c).cuda()
    dist=(W-X)*(W-X)#bncd
    dist=torch.sum(dist,-1)
    id=torch.argmin(dist,-1,True)
    #Y[id]=1
    id0=torch.arange(Y.size(0)).unsqueeze(1).repeat(1,Y.size(1)).view(-1)
    id1=torch.arange(Y.size(1)).unsqueeze(0).repeat(Y.size(0),1).view(-1)
    id=id.view(-1)
    Y[id0,id1,id]=1
    X0=X
    Y0=Y

    t=W
    X=torch.normal(t,sigma).cuda()#bqcd
    Y=torch.zeros(B,1,c,c).cuda()
    Y[:,:,0,0]=1
    Y[:,:,1,1]=1
    
    Y=Y.view(B,-1,c)
    X=X.view(B,-1,d)
    shuffle=torch.normal(0.0,1.0,(1,))
    if shuffle<0:
        X=torch.cat([X[:,1:2],X[:,0:1]],1)
        Y=torch.cat([Y[:,1:2],Y[:,0:1]],1)
    X0=X0.squeeze(-2)
    #X=torch.cat([X,X0[:,2:]],1)
    #Y=torch.cat([Y,Y0[:,2:]],1)
    Y=torch.cat((Y0,torch.zeros_like(Y0)),-1)

    return X0.squeeze(-2),Y

C=2
dim=2
N=50
#train_num=2**15
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
test_set=generate_task_proto(c=C,d=dim,B=2048)
tx,ty=sample_proto(test_set,N=N)

opt=torch.optim.Adam(model.parameters(),lr=3e-3)
l=train(model,tx,ty,1000000,128,opt,loss_func,N=N,test_every=100,save='uproto_dim2.pt',sl='uproto_acc.npy',C=C)

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
        if sl!=None:
            np.save(sl,test_loss)
    print('epoch:',epoch,'train_loss:',train_loss,'average test acc:',t,'  best average acc:',best_test,' best acc',best_test_l)




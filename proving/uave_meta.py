import torch
from linear_attention_transformer import ICLer
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class MatchNet(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim):
        super(MatchNet, self).__init__()
        self.ff=nn.Sequential(nn.Linear(inp_dim,hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim,out_dim))
    
    def forward(self,xs,ys,xq):
        xs=self.ff(xs)#bnd
        xq=self.ff(xq)
        sims=torch.bmm(xq,xs.transpose(-1,-2))#bqn
        pred=torch.bmm(sims,ys)#bqn*bnc=bqc
        pred=torch.cat([pred[:,:,:C],-9999*torch.ones_like(pred[:,:,:C])],-1)
        pred=torch.softmax(pred,-1)
        return pred
    
def extract_class_indices(ys) -> torch.Tensor:
    labels=torch.argmax(ys,-1)#n
    class_mask = torch.eq(labels, 0).float().unsqueeze(-1)
    return class_mask,1-class_mask

class ProtoNet(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim):
        super(ProtoNet, self).__init__()
        self.ff=nn.Sequential(nn.Linear(inp_dim,hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim,out_dim))
    
    def forward(self,xs,ys,xq):
        xs=self.ff(xs)#bnd
        #for b in range(int(xs.size(0))):
        id0,id1=extract_class_indices(ys)#bn1
        if id0.size(1)==0:
            p0=torch.zeros((xs.size(0),xs.size(2),1)).cuda()
        else:
            d0=torch.sum(id0,1,True)
            d0[d0==0]=1
            p0=torch.bmm(xs.transpose(-1,-2),id0)/d0#bd1
        if id1.size(1)==0:
            p1=torch.zeros((xs.size(0),xs.size(2),1)).cuda()
        else:
            d1=torch.sum(id1,1,True)
            d1[d1==0]=1
            p1=torch.bmm(xs.transpose(-1,-2),id1)/d1
        #p0=torch.nan_to_num(p0, nan=0.0)
        #p1=torch.nan_to_num(p1, nan=0.0)
        ps=torch.cat([p0,p1],-1)#bdc
        ps=ps.transpose(-1,-2).unsqueeze(1)#b1cd
        xq=xq.unsqueeze(2)#bq1d
        #print(ps.sum(),xq.sum())
        d=(ps-xq)*(ps-xq)#bqcd
        pred=torch.sum(d,-1)
        #print(pred.sum())
        pred=torch.cat([pred,9999*torch.ones_like(pred)],-1)
        pred=torch.softmax(-pred,-1)
        return pred
    
class Amortization(nn.Module):
    def __init__(self, inp_dim1,hidden_dim1,out_dim1,inp_dim2,hidden_dim2,out_dim2):
        super(Amortization, self).__init__()
        self.context_dim=out_dim1
        self.ff1=nn.Sequential(nn.Linear(inp_dim1,hidden_dim1),nn.LeakyReLU(),nn.Linear(hidden_dim1,out_dim1))
        self.ff2=nn.Sequential(nn.Linear(out_dim1,out_dim1),nn.LeakyReLU(),nn.Linear(out_dim1,out_dim1))
        self.ff_o=nn.Sequential(nn.Linear(inp_dim2,hidden_dim2),nn.LeakyReLU(),nn.Linear(hidden_dim2,out_dim2))
    
    def forward(self,xs,ys,xq):
        if xs.size(1)==0:
            context=torch.zeros((xq.size(0),xq.size(1),self.context_dim)).cuda()
        else:
            xs=torch.cat([xs,ys],-1)
            xs=self.ff1(xs)#bnd
            context=torch.mean(xs,1,True).repeat(1,xq.size(1),1)#bqd
            context=self.ff2(context)
        x=torch.cat([context,xq],-1)
        #x=xq+context
        pred=self.ff_o(x)
        return pred
    
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
            xs=X[:,:i,]
            ys=Y[:,:i,]
            xq=X[:,i:i+1,]
            yq=Y[:,i:i+1,]
            pred=model(xs,ys,xq)

            loss=criterion(pred.squeeze(),yq.squeeze())
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
                    xs=X[:,:i,]
                    ys=Y[:,:i,]
                    xq=X[:,i:i+1,]
                    yq=Y[:,i:i+1,]
                    pred=model(xs,ys,xq)

                    loss=criterion(pred.squeeze(),yq.squeeze())
                    ac=acc(pred,yq)
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
    c=2
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

#model = MatchNet(dim,dim,dim).cuda()
model= ProtoNet(dim,dim,dim).cuda()
#model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()

#train_set=generate_task_reg(c=2,d=dim,B=train_num)

test_set=generate_task_reg(d=dim,B=2048)
tx,ty=sample_bin(test_set,N=N)

opt=torch.optim.Adam(model.parameters(),lr=1e-3)
l=train(model,tx,ty,100000,128,opt,loss_func,N=N,test_every=100,save='uave_meta_proto_dim2.pt',sl='uave_meta_acc_dim2.npy')

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




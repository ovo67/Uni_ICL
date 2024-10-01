import torch
from linear_attention_transformer import ICLer
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '4'



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
    
class MatchNet(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim):
        super(MatchNet, self).__init__()
        self.ff=nn.Sequential(nn.Linear(inp_dim,hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim,out_dim))
    
    def forward(self,xs,ys,xq):
        xs=self.ff(xs)#bnd
        xq=self.ff(xq)
        xs=xs/xs.norm(2,-1,True)
        xq=xq/xq.norm(2,-1,True)
        sims=torch.bmm(xq,xs.transpose(-1,-2))#bqn
        pred=torch.bmm(sims,ys)#bqn*bnc=bqc
        pred=torch.cat([pred[:,:,:2],-99999*torch.ones_like(pred[:,:,:2])],-1)
        pred=torch.softmax(pred,-1)
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
    batch_size=int(batch_size/3)*3
    best_test=-1
    best_test_l=0
    for epoch in range(steps):
        train_loss=0
        X,Y=get_batch_mix(d=dim,B=batch_size,N=N)
        c=Y.size(-1)
        bloss=0
        for i in range(2,N):
            if Type=='icl':
                x=X[:,:i+1,]
                y=torch.cat([Y[:,:i,],torch.zeros(batch_size,1,c).cuda()],1)
                z=torch.cat((x,y),-1)
                pred=model(z)
                
                loss=criterion(pred[:,-1,],Y[:,i,])
            else:
                xs=X[:,:i,]
                ys=Y[:,:i,]
                xq=X[:,i:i+1,]
                yq=Y[:,i:i+1,]
                pred=model(xs,ys,xq)

                loss=criterion(pred.squeeze(),yq.squeeze())
                
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
                    if Type=='icl':
                        x=X[:,:i+1,]
                        y=torch.cat([Y[:,:i,],torch.zeros(Y.size(0),1,c).cuda()],1)
                        z=torch.cat((x,y),-1)
                        pred=model(z)[:,:,-c:]
                        ac=acc(pred[:,-1],Y[:,i,])
                    else:
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

def loss_func(p,y):
    return F.cross_entropy(p,y)

def acc(p,y):
    correct=torch.argmax(p,-1)==torch.argmax(y,-1)
    return (correct.sum()/p.size(0))

def generate_task_proto(c=2,d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,c,d).normal_(mu,sigma)
    return W.cuda()

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
    X=torch.cat([X,X0[:,2:]],1)
    Y=torch.cat([Y,Y0[:,2:]],1)
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X.squeeze(-2),Y

def sample_proto_normal(W,N=1,mu=0,sigma=1):
    B,c,d=W.size()
    q=int(N/c)
    t=torch.zeros(B,N,d)
    X=torch.normal(t,sigma).cuda()#bqcd
    X=X.view(B,-1,d).unsqueeze(-2)#bn1d
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
    return X.squeeze(-2),Y
def generate_task_sim0(c=2,d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,c,d).normal_(mu,sigma)
    W=W/W.norm(2,-1,True)
    return W.cuda()

def sample_sim0(W,N=1,mu=0,sigma=1):
    '''B,c,d=W.size()
    q=int(N/c)
    t=W.unsqueeze(1).repeat(1,q,1,1)#bqcd
    X=torch.normal(t,sigma).cuda()#bqcd
    X=X/X.norm(2,-1,True)
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
    X= torch.FloatTensor(B,N, d).normal_(mu,sigma).cuda()
    X0=X/X.norm(2,-1,True)
    Y=torch.zeros(B,N,c).cuda()
    sims=torch.bmm(X,W.transpose(-1,-2))#bnc
    id=torch.argmax(sims,-1,True)
    #Y[id]=1
    id0=torch.arange(Y.size(0)).unsqueeze(1).repeat(1,Y.size(1)).view(-1)
    id1=torch.arange(Y.size(1)).unsqueeze(0).repeat(Y.size(0),1).view(-1)
    id=id.view(-1)
    Y[id0,id1,id]=1
    Y0=Y

    t=W.unsqueeze(1).repeat(1,1,1,1)#bqcd
    X=torch.normal(t,sigma).cuda()#bqcd
    X=X/X.norm(2,-1,True)
    Y=torch.zeros(B,1,c,c).cuda()
    Y[:,:,0,0]=1
    Y[:,:,1,1]=1
    
    Y=Y.view(B,-1,c)
    X=X.view(B,-1,d)
    shuffle=torch.normal(0.0,1.0,(1,))
    if shuffle<0:
        X=torch.cat([X[:,1:2],X[:,0:1]],1)
        Y=torch.cat([Y[:,1:2],Y[:,0:1]],1)

    '''X=torch.cat([X,X0[:,2:]],1)
    Y=torch.cat([Y,Y0[:,2:]],1)'''
    Y=torch.cat((Y0,torch.zeros_like(Y0)),-1)
    return X0.squeeze(-2),Y

def generate_task_sim(c=2,nc=2,d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,c,nc,d).normal_(mu,sigma)
    #W=W/W.norm(2,-1,True)
    return W.cuda()

def sample_sim(W,N=1,mu=0,sigma=1):
    '''B,c,d=W.size()
    X= torch.FloatTensor(B,N, d).normal_(mu,sigma).cuda()
    X=X/X.norm(2,-1,True)
    Y=torch.zeros(B,N,c).cuda()
    sims=torch.bmm(X,W.transpose(-1,-2))#bnc
    id=torch.argmax(sims,-1,True)
    #Y[id]=1
    id0=torch.arange(Y.size(0)).unsqueeze(1).repeat(1,Y.size(1)).view(-1)
    id1=torch.arange(Y.size(1)).unsqueeze(0).repeat(Y.size(0),1).view(-1)
    id=id.view(-1)
    Y[id0,id1,id]=1
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X,Y'''
    B,c,nc,d=W.size()
    X= torch.FloatTensor(B,N, d).normal_(mu,sigma).cuda()
    X0=X/X.norm(2,-1,True)#b,n,d
    W0=(W/W.norm(3,-1,True)).view(B,-1,d)#b,c*nc,d
    Y=torch.zeros(B,N,c).cuda()
    sims=torch.bmm(X0,W0.transpose(-1,-2)).view(B,N,c,nc)
    sims=torch.sum(sims,-1)#bnc
    id=torch.argmax(sims,-1,True)
    #Y[id]=1
    id0=torch.arange(Y.size(0)).unsqueeze(1).repeat(1,Y.size(1)).view(-1)
    id1=torch.arange(Y.size(1)).unsqueeze(0).repeat(Y.size(0),1).view(-1)
    id=id.view(-1)
    Y[id0,id1,id]=1

    '''yw=torch.zeros(B,c,nc,c).cuda()
    for i in range(c):
        yw[:,i,:,i]=1
    X=torch.cat([W.view(B,-1,d),X],1)
    Y=torch.cat([yw.view(B,-1,c),Y],1)'''
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    return X,Y



def generate_task_ave(d=1,B=1000, mu=0,sigma=1):
    W= torch.FloatTensor(B,d).normal_(mu,sigma)
    return W.cuda()

def sample_ave(W,N=1,sigma=1):
    B,d=W.size()
    c=2
    W=W.unsqueeze(1).repeat(1,N,1)#bnd
    X=torch.normal(W,sigma).cuda()#bnd
    #Y=torch.sum(W+X,-1,True)
    #t=(W+X).view(-1,1,d)
    #Y=torch.bmm(t,t.transpose(-1,-2))
    #Y=Y.view(B,-1,1)
    x0=torch.normal(W-0.1,0.1).cuda()
    x1=torch.normal(W+0.1,0.1).cuda()
    
    
    d=(W-X).sum(-1)#bn
    n=(d<0).float().unsqueeze(-1)
    p=torch.ones_like(n)-n
    Y=torch.cat([n,p],-1).cuda()
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    X=torch.cat([x0[:,0:1,:],x1[:,0:1,:],X[:,2:,:]],1)
    y0=torch.cat([torch.ones((Y.size(0),1,1)),torch.zeros((Y.size(0),1,Y.size(2)-1))],-1).cuda()
    y1=torch.cat([torch.zeros((Y.size(0),1,1)),torch.ones((Y.size(0),1,1)),torch.zeros((Y.size(0),1,Y.size(2)-2))],-1).cuda()
    Y=torch.cat([y0,y1,Y[:,2:]],1)

    shuffle=torch.normal(0.0,1.0,(1,))
    if shuffle<0:
        X=torch.cat([X[:,1:2],X[:,0:1],X[:,2:]],1)
        Y=torch.cat([Y[:,1:2],Y[:,0:1],Y[:,2:]],1)
    return X,Y

def generate_task_rad(B=1000, start=0,end=1):
    W= torch.FloatTensor(B,1).uniform_(start,end)
    return W.cuda()

def sample_rad(W,N=1,d=1):
    B,_=W.size()
    c=2
    W=W.unsqueeze(1).repeat(1,N,1)#bn1
    Wd=W.repeat(1,1,d)#bnd
    #sigma=torch.pow(torch.Tensor([0.5]),1/(2*d)).cuda()*W.repeat(1,1,d)#bnd
    sigma=0.849*Wd
    X=torch.normal(0,sigma).cuda()#bnd

    r=torch.norm(X,dim=-1,keepdim=True)#bn1
    n=(r>W).float()
    p=torch.ones_like(n)-n
    Y=torch.cat([n,p],-1).cuda()
    Y=torch.cat((Y,torch.zeros_like(Y)),-1)
    #print(Y.sum(0).sum(0))
    #ss
    return X,Y

def get_batch_mix(d=1,B=1000, mu=0,sigma=1,r_proto=1,r_sim=1,r_ave=1,N=1,sigma_in=1):
    b_proto=int(B*r_proto/(r_proto+r_sim+r_ave))
    b_ave=int(B*r_ave/(r_proto+r_sim+r_ave))
    b_sim=int(B*r_sim/(r_proto+r_sim+r_ave))

    proto=generate_task_proto(2,d,b_proto,mu,sigma)
    ave=generate_task_ave(d,b_ave,mu,sigma)
    sim=generate_task_sim0(2,d,b_sim,mu,sigma)
    #print(proto.size(),N)
    x1,y1=sample_proto(proto,N)
    x2,y2=sample_ave(ave,N)
    x3,y3=sample_sim0(sim,N)

    X=torch.cat([x1,x2,x3],0)
    Y=torch.cat([y1,y2,y3],0)

    return X,Y

def test(model,tx,ty,Type):
    test_loss=[]
    with torch.no_grad():
        X,Y=tx,ty
        for i in range(0,N):
            if Type=='icl':
                x=X[:,:i+1,]
                y=torch.cat([Y[:,:i,],torch.zeros(Y.size(0),1,2*C).cuda()],1)
                z=torch.cat((x,y),-1)
                pred=model(z)
                ac=acc(pred[:,-1],Y[:,i,])
            else:
                xs=X[:,:i,]
                ys=Y[:,:i,]
                xq=X[:,i:i+1,]
                yq=Y[:,i:i+1,]
                pred=model(xs,ys,xq)
                ac=acc(pred,yq)
            test_loss.append(ac.cpu().item())
        print(test_loss)
        return test_loss

def test_p(model,tx,ty,Type,sn):
    test_loss=[]
    x_sup=[]
    y_sup=[]
    x_qry=[]
    p_qry=[]
    y_qry=[]
    with torch.no_grad():
        X,Y=tx,ty
        for i in range(sn,N):
            if Type=='icl':
                x=torch.cat([X[:,:sn,],X[:,i:i+1,]],1)
                y=torch.cat([Y[:,:sn,],torch.zeros(Y.size(0),1,2*C).cuda()],1)
                z=torch.cat((x,y),-1)
                pred=model(z)
                ac=acc(pred[:,-1],Y[:,i,])
            else:
                xs=X[:,:i,]
                ys=Y[:,:i,]
                xq=X[:,i:i+1,]
                yq=Y[:,i:i+1,]
                pred=model(xs,ys,xq)
                ac=acc(pred,yq)
            test_loss.append(ac.cpu().item())
            x_qry.append(X[:,i,])
            p_qry.append(pred[:,-1])
            y_qry.append(Y[:,i,])
            #samples.append([X[:,i:i+1,].cpu().numpy(),pred[:,-1].cpu().numpy(),Y[:,i,].cpu().numpy()])
        print(test_loss)
        t=np.mean(test_loss)
        print(t)
        
        return test_loss,X[:,:sn,].cpu().numpy(),Y[:,:sn,].cpu().numpy(),torch.stack(x_qry,0).cpu().numpy(),torch.stack(p_qry,0).cpu().numpy(),torch.stack(y_qry,0).cpu().numpy()
    
C=2
dim=2
N=4000
#train_num=2**15
loss=[]
Type='-'

def test_cross_task_model():
    test_set=generate_task_proto(c=C,d=dim,B=2048)
    tx_p,ty_p=sample_proto(test_set,N=N)

    test_set=generate_task_sim(c=C,nc=4,d=dim,B=2048)
    tx_s,ty_s=sample_sim(test_set,N=N)

    test_set=generate_task_ave(d=dim,B=2048)
    tx_a,ty_a=sample_ave(test_set,N=N)

    #model = MatchNet(dim,dim,dim).cuda()
    model= ProtoNet(dim,dim,dim).cuda()
    #model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model.load_state_dict(torch.load("uproto_meta_proto_dim2.pt"))
    p_p=test(model,tx_p,ty_p,'-')

    model= ProtoNet(dim,dim,dim).cuda()
    model.load_state_dict(torch.load("usim_meta_proto_dim2.pt"))
    s_p=test(model,tx_s,ty_s,'-')

    model= ProtoNet(dim,dim,dim).cuda()
    model.load_state_dict(torch.load("uave_meta_proto_dim2.pt"))
    a_p=test(model,tx_a,ty_a,'-')

    model = MatchNet(dim,dim,dim).cuda()
    model.load_state_dict(torch.load("uproto_meta_sim_dim2.pt"))
    p_s=test(model,tx_p,ty_p,'-')

    model = MatchNet(dim,dim,dim).cuda()
    model.load_state_dict(torch.load("usim_meta_sim_dim2.pt"))
    s_s=test(model,tx_s,ty_s,'-')

    model = MatchNet(dim,dim,dim).cuda()
    model.load_state_dict(torch.load("uave_meta_sim_dim2.pt"))
    a_s=test(model,tx_a,ty_a,'-')

    model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model.load_state_dict(torch.load("uproto_meta_ave_dim2.pt"))
    p_a=test(model,tx_p,ty_p,'-')

    model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model.load_state_dict(torch.load("usim_meta_ave_dim2.pt"))
    s_a=test(model,tx_s,ty_s,'-')

    model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model.load_state_dict(torch.load("uave_meta_ave_dim2.pt"))
    a_a=test(model,tx_a,ty_a,'-')

    model = ICLer(
        dim = dim+2*C,
        heads = 1,
        depth = 8,
        max_seq_len = 8192,
        label_dim=2*C,
        ff_list=[1,1,1,1,1,1,1,1],
        readout='linear',
    ).cuda()
    model.load_state_dict(torch.load("uproto_dim2.pt"))
    p_i=test(model,tx_p,ty_p,'icl')

    model.load_state_dict(torch.load("usim_dim2.pt"))
    s_i=test(model,tx_s,ty_s,'icl')

    model.load_state_dict(torch.load("uave_dim2.pt"))
    a_i=test(model,tx_a,ty_a,'icl')

    td={}
    td['pp']=p_p
    td['sp']=s_p
    td['ap']=a_p
    td['ps']=p_s
    td['ss']=s_s
    td['as']=a_s
    td['pa']=p_a
    td['sa']=s_a
    td['aa']=a_a
    td['pi']=p_i
    td['si']=s_i
    td['ai']=a_i
    print(td)

    with open('./plot_alg/test_cross.json', 'w') as file:
        json.dump(td, file)
    
def test_shift():
    model_m = ICLer(
    dim = dim+2*C,
    heads = 1,
    depth = 8,
    max_seq_len = 8192,
    label_dim=2*C,
    ff_list=[1,1,1,1,1,1,1,1],
    readout='linear',
    ).cuda()

    model_m.load_state_dict(torch.load("umix_dim2_sim1.pt"))

    
    td={}
    p=[]
    for sft in range (5):
        print(sft)
        test_set=generate_task_proto(c=C,d=dim,B=2048)
        tx_p,ty_p=sample_proto(test_set,N=N,sigma=1+sft)

        test_set=generate_task_sim(c=C,nc=4,d=dim,B=2048)
        tx_s,ty_s=sample_sim(test_set,N=N,sigma=1+sft)

        test_set=generate_task_ave(d=dim,B=2048)
        tx_a,ty_a=sample_ave(test_set,N=N,sigma=1+sft)
        p=test(model_m,tx_p,ty_p,'icl')
        s=test(model_m,tx_s,ty_s,'icl')
        a=test(model_m,tx_a,ty_a,'icl')
        if 'mix_p' not in td.keys():
            td['mix_p']=[]
            td['mix_s']=[]
            td['mix_a']=[]
        td['mix_p'].append(np.mean(p))
        td['mix_s'].append(np.mean(s))
        td['mix_a'].append(np.mean(a))
    
    with open('./plot_alg/test_shift.json', 'w') as file:
        json.dump(td, file)


#test_cross_task_model()
#ss

'''if Type=='icl':
    model = ICLer(
        dim = dim+2*C,
        heads = 1,
        depth = 8,
        max_seq_len = 8192,
        label_dim=2*C,
        ff_list=[1,1,1,1,1,1,1,1],
        readout='linear',
    ).cuda()
    save_name='umix.pt'
    sl_name='umix_acc.npy'

elif Type=='proto':
    model = ProtoNet(dim,dim,dim).cuda()
    save_name='umix_proto.pt'
    sl_name='umix_proto_acc.npy'

elif Type=='ave':
    model = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    #model = Amortization(dim+2*C,dim+2*C,dim,dim,dim+2*C,2*C).cuda()
    save_name='umix_ave.pt'
    sl_name='umix_ave_acc.npy'

elif Type=='sim':
    model = MatchNet(dim,dim,dim).cuda()
    save_name='umix_sim.pt'
    sl_name='umix_sim_acc.npy'


tx,ty=get_batch_mix(dim,3072,N=50)
opt=torch.optim.Adam(model.parameters(),lr=3e-3)

l=train(model,tx,ty,1000000,128,opt,loss_func,N=N,test_every=100,save=save_name,sl=sl_name)'''

def test_trained_umix():
    model_icl=model = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('umix_dim2_sim1.pt'))

    model_proto = ProtoNet(dim,dim,dim).cuda()
    model_proto.load_state_dict(torch.load('umix_proto.pt'))

    model_ave = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model_ave.load_state_dict(torch.load('umix_ave.pt'))

    model_sim = MatchNet(dim,dim,dim).cuda()
    model_sim.load_state_dict(torch.load('umix_sim.pt'))

    test_set=generate_task_proto(c=2,d=dim,B=3072)
    tx_proto,ty_proto=sample_proto(test_set,N=N)

    test_set=generate_task_ave(d=dim,B=3072)
    tx_ave,ty_ave=sample_ave(test_set,N=N)

    test_set=generate_task_sim(c=2,d=dim,B=3072)
    tx_sim,ty_sim=sample_sim(test_set,N=N)

    tx_mix,ty_mix=get_batch_mix(dim,3072,N=N)

    td={}
    td['umix_icl_proto']=test(model_icl,tx_proto,ty_proto,'icl')
    td['umix_icl_sim']=test(model_icl,tx_sim,ty_sim,'icl')
    td['umix_icl_ave']=test(model_icl,tx_ave,ty_ave,'icl')
    td['umix_icl_mix']=test(model_icl,tx_mix,ty_mix,'icl')
    
    td['umix_proto_proto']=test(model_proto,tx_proto,ty_proto,'proto')
    td['umix_proto_sim']=test(model_proto,tx_sim,ty_sim,'proto')
    td['umix_proto_ave']=test(model_proto,tx_ave,ty_ave,'proto')
    td['umix_proto_mix']=test(model_proto,tx_mix,ty_mix,'proto')

    td['umix_sim_proto']=test(model_sim,tx_proto,ty_proto,'sim')
    td['umix_sim_sim']=test(model_sim,tx_sim,ty_sim,'sim')
    td['umix_sim_ave']=test(model_sim,tx_ave,ty_ave,'sim')
    td['umix_sim_mix']=test(model_sim,tx_mix,ty_mix,'sim')

    td['umix_ave_proto']=test(model_ave,tx_proto,ty_proto,'ave')
    td['umix_ave_sim']=test(model_ave,tx_sim,ty_sim,'ave')
    td['umix_ave_ave']=test(model_ave,tx_ave,ty_ave,'ave')
    td['umix_ave_mix']=test(model_ave,tx_mix,ty_mix,'ave')

    with open('test_acc_trained_umix.json', 'w') as file:
        json.dump(td, file)

def test_trained_umix_rad():
    model_icl = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('umix_dim2_sim1.pt'))

    test_set=generate_task_rad(10240,2,4)
    tx,ty=sample_rad(test_set,N,dim)
    acc=test(model_icl,tx,ty,'icl')
    np.save('test_model_selection_dim2.npy',acc)
    ss
    
    model_proto = ProtoNet(dim,dim,dim).cuda()
    model_proto.load_state_dict(torch.load('umix_proto_dim2_sim1.pt'))

    model_ave = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model_ave.load_state_dict(torch.load('umix_ave_dim2_sim1.pt'))

    model_sim = MatchNet(dim,dim,dim).cuda()
    model_sim.load_state_dict(torch.load('umix_sim_dim2_sim1.pt'))

    acc=test(model_proto,tx,ty,'proto')
    np.save('test_model_rad_proto_dim2.npy',acc)
    acc=test(model_ave,tx,ty,'ave')
    np.save('test_model_rad_ave_dim2.npy',acc)
    acc=test(model_sim,tx,ty,'sim')
    np.save('test_model_rad_sim_dim2.npy',acc)

def test_plot_alg_proto(i=1):
    model_icl = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('uproto.pt'))

    test_set=generate_task_proto(c=4,d=dim,B=8,sigma=1)
    tx,ty=sample_proto(test_set,N=N,sigma=1)
    acc,xs,ys,xq,pq,yq=test_p(model_icl,tx,ty,'icl',480)
    np.save('./plot_alg/proto_'+str(i)+'_xs.npy',xs)
    np.save('./plot_alg/proto_'+str(i)+'_ys.npy',ys)
    np.save('./plot_alg/proto_'+str(i)+'_xq.npy',xq)
    np.save('./plot_alg/proto_'+str(i)+'_pq.npy',pq)
    np.save('./plot_alg/proto_'+str(i)+'_yq.npy',yq)

def test_plot_alg_ave(i=0):
    model_icl = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('uave.pt'))

    test_set=generate_task_ave(d=dim,B=8,sigma=1)
    tx,ty=sample_ave(test_set,N=N,sigma=1)
    acc,xs,ys,xq,pq,yq=test_p(model_icl,tx,ty,'icl',480)
    np.save('./plot_alg/ave_'+str(i)+'_xs.npy',xs)
    np.save('./plot_alg/ave_'+str(i)+'_ys.npy',ys)
    np.save('./plot_alg/ave_'+str(i)+'_xq.npy',xq)
    np.save('./plot_alg/ave_'+str(i)+'_pq.npy',pq)
    np.save('./plot_alg/ave_'+str(i)+'_yq.npy',yq)

def test_plot_alg_sim(i=0):
    model_icl = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('usim.pt'))

    test_set=generate_task_sim(c=4,nc=32,d=dim,B=8,sigma=1)
    tx,ty=sample_sim(test_set,N=N,sigma=1)
    acc,xs,ys,xq,pq,yq=test_p(model_icl,tx,ty,'icl',144)
    np.save('./plot_alg/sim_'+str(i)+'_xs.npy',xs)
    np.save('./plot_alg/sim_'+str(i)+'_ys.npy',ys)
    np.save('./plot_alg/sim_'+str(i)+'_xq.npy',xq)
    np.save('./plot_alg/sim_'+str(i)+'_pq.npy',pq)
    np.save('./plot_alg/sim_'+str(i)+'_yq.npy',yq)

def test_plot_alg_rad(i=0):
    model_icl = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('umix_dim2_sim1.pt'))

    test_set=generate_task_rad(B=16,start=1,end=4)
    tx,ty=sample_rad(test_set,N=N,d=dim,)
    acc,xs,ys,xq,pq,yq=test_p(model_icl,tx,ty,'icl',32)
    np.save('./plot_alg/rad_'+str(i)+'_xs.npy',xs)
    np.save('./plot_alg/rad_'+str(i)+'_ys.npy',ys)
    np.save('./plot_alg/rad_'+str(i)+'_xq.npy',xq)
    np.save('./plot_alg/rad_'+str(i)+'_pq.npy',pq)
    np.save('./plot_alg/rad_'+str(i)+'_yq.npy',yq)

def test_all():
    model_icl=model = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl.load_state_dict(torch.load('umix_dim2_sim1.pt'))

    model_proto = ProtoNet(dim,dim,dim).cuda()
    model_proto.load_state_dict(torch.load('umix_proto_dim2_sim1.pt'))

    model_ave = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model_ave.load_state_dict(torch.load('umix_ave_dim2_sim1.pt'))

    model_sim = MatchNet(dim,dim,dim).cuda()
    model_sim.load_state_dict(torch.load('umix_sim_dim2_sim1.pt'))

    test_set=generate_task_proto(c=2,d=dim,B=3072)
    tx_proto,ty_proto=sample_proto(test_set,N=N)

    test_set=generate_task_ave(d=dim,B=3072)
    tx_ave,ty_ave=sample_ave(test_set,N=N)

    test_set=generate_task_sim(c=2,nc=2,d=dim,B=3072)
    tx_sim,ty_sim=sample_sim(test_set,N=N)

    tx_mix,ty_mix=get_batch_mix(dim,3072,N=N)

    td={}
    td['umix_icl_proto']=test(model_icl,tx_proto,ty_proto,'icl')
    td['umix_icl_sim']=test(model_icl,tx_sim,ty_sim,'icl')
    td['umix_icl_ave']=test(model_icl,tx_ave,ty_ave,'icl')
    td['umix_icl_mix']=test(model_icl,tx_mix,ty_mix,'icl')
    
    td['umix_proto_proto']=test(model_proto,tx_proto,ty_proto,'proto')
    td['umix_proto_sim']=test(model_proto,tx_sim,ty_sim,'proto')
    td['umix_proto_ave']=test(model_proto,tx_ave,ty_ave,'proto')
    td['umix_proto_mix']=test(model_proto,tx_mix,ty_mix,'proto')

    td['umix_sim_proto']=test(model_sim,tx_proto,ty_proto,'sim')
    td['umix_sim_sim']=test(model_sim,tx_sim,ty_sim,'sim')
    td['umix_sim_ave']=test(model_sim,tx_ave,ty_ave,'sim')
    td['umix_sim_mix']=test(model_sim,tx_mix,ty_mix,'sim')

    td['umix_ave_proto']=test(model_ave,tx_proto,ty_proto,'ave')
    td['umix_ave_sim']=test(model_ave,tx_sim,ty_sim,'ave')
    td['umix_ave_ave']=test(model_ave,tx_ave,ty_ave,'ave')
    td['umix_ave_mix']=test(model_ave,tx_mix,ty_mix,'ave')

    model_icl_proto = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl_proto.load_state_dict(torch.load('uproto_dim2.pt'))

    model_icl_sim = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl_sim.load_state_dict(torch.load('usim_dim2.pt'))

    model_icl_ave = ICLer(
            dim = dim+2*C,
            heads = 1,
            depth = 8,
            max_seq_len = 8192,
            label_dim=2*C,
            ff_list=[1,1,1,1,1,1,1,1],
            readout='linear',
        ).cuda()
    model_icl_ave.load_state_dict(torch.load('uave.pt'))

    model_proto = ProtoNet(dim,dim,dim).cuda()
    model_proto.load_state_dict(torch.load('uproto_meta_dim2.pt'))

    model_ave = Amortization(dim+2*C,dim+2*C,dim+2*C,2*dim+2*C,dim+2*C,2*C).cuda()
    model_ave.load_state_dict(torch.load('uave_meta_dim2.pt'))

    model_sim = MatchNet(dim,dim,dim).cuda()
    model_sim.load_state_dict(torch.load('usim_meta_dim2.pt'))

    td['uproto_icl']=test(model_icl_proto,tx_proto,ty_proto,'icl')
    td['uproto_meta']=test(model_proto,tx_proto,ty_proto,'proto')

    td['uave_icl']=test(model_icl_ave,tx_ave,ty_ave,'icl')
    td['uave_meta']=test(model_ave,tx_ave,ty_ave,'ave')

    td['usim_icl']=test(model_icl_sim,tx_sim,ty_sim,'icl')
    td['usim_meta']=test(model_sim,tx_sim,ty_sim,'sim')

    

    with open('test_acc_trained_all_dim2.json', 'w') as file:
        json.dump(td, file)

#test_trained_umix()

#test_trained_umix_rad()
test_plot_alg_ave(1)
#ss
#test_all()




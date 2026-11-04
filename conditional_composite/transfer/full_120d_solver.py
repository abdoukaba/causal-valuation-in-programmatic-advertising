"""Independent reimplementation of the GitHub repository's explicit finite-difference MFG.
Source aligned to code/verify_auction.py and code/calib_model.py (main branch as accessed 2026-11-03).
NOT an interval/verified numerical solver.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logsumexp

class Model:
    def __init__(self,np_grid=21,nt=60,eta=.5,eps=.05,T=1.,lam=2.25,
                 zeta=.2,base=(.2,.4,.4),delta=(-.01,.01),values=None,mgrid=None,p0=.5,rho=None):
        self.np=int(np_grid);self.nt=int(nt);self.eta=float(eta);self.entropy=float(eps)
        self.T=float(T);self.lam=float(lam);self.dt=self.T/self.nt;self.sigma=.2
        self.actions=np.array([0.,.05,.1]);self.nact=3;self.nb=5
        self.budgets=np.arange(5)*.05
        self.p=np.linspace(0,1,self.np);self.h=self.p[1]-self.p[0]
        self.up=(self.sigma*self.p*(1-self.p))**2/(2*self.h**2)
        self.up[-1]=0
        self.down=self.up.copy();self.down[0]=0
        assert self.dt*(self.lam+max(self.up+self.down))<.99
        self.mgrid=np.asarray(mgrid if mgrid is not None else self.actions, dtype=float)
        self.base=np.asarray(base,dtype=float)
        assert len(self.base)==len(self.mgrid)
        self.delta=np.asarray(delta,dtype=float)
        self.rho=None if rho is None else np.asarray(rho,dtype=float)
        self.p0=float(p0)
        self.noise=float(zeta)
        if values is None:
            self.value=np.stack([.060+.03*self.mgrid,.095+.03*self.mgrid])
        else: self.value=np.asarray(values,dtype=float)
        self.w=np.zeros((2,3,len(self.mgrid)))
        for s in range(2):
            for ia,a in enumerate(self.actions):
                if ia>0: self.w[s,ia]=expit((a-self.mgrid-self.delta[s])/self.noise)

    def market(self,qt):
        mix=(1-self.eta)*self.base.copy()
        mix[:3]+=self.eta*qt
        ws=np.einsum('sam,m->sa',self.w,mix)
        gs=np.einsum('sam,sm,m->sa',self.w,self.value,mix)-self.actions[None,:]*ws
        pc=self.p[:,None]
        pw=(1-pc)*ws[0][None,:]+pc*ws[1][None,:]
        g=(1-pc)*gs[0][None,:]+pc*gs[1][None,:]
        wbel=np.clip(pc*ws[1][None,:]/np.maximum(pw,1e-100),0,1)
        lbel=np.clip(pc*(1-ws[1][None,:])/np.maximum(1-pw,1e-100),0,1)
        pw[:,0]=0;g[:,0]=0;wbel[:,0]=self.p;lbel[:,0]=self.p
        return pw,g,wbel,lbel

    def locate(self,bel):
        jj=np.minimum(np.floor(np.asarray(bel)/self.h).astype(int),self.np-2)
        fr=np.clip((bel-self.p[jj])/self.h,0,1)
        return jj,fr

    def interp(self,row,bel):
        jj,fr=self.locate(bel)
        return (1-fr)*row[jj]+fr*row[jj+1]

    def gen(self,v):
        r=np.zeros_like(v)
        r[:,:-1]+=self.up[:-1][None,:]*(v[:,1:]-v[:,:-1]); r[:,1:]+=self.down[1:][None,:]*(v[:,:-1]-v[:,1:])
        return r

    def scores(self,v,qt):
        pw,g,bw,bl=self.market(qt)
        score=np.full((self.nb,self.np,3),-1e6)
        for b in range(self.nb):
            score[b,:,0]=v[b]
            for a in range(1,min(b,2)+1):
                score[b,:,a]=g[:,a]+pw[:,a]*self.interp(v[b-a],bw[:,a])+(1-pw[:,a])*self.interp(v[b],bl[:,a])
        return score

    def backward(self,q):
        V=np.zeros((self.nt+1,self.nb,self.np));pol=np.zeros((self.nt,self.nb,self.np,3))
        for t in range(self.nt-1,-1,-1):
            score=self.scores(V[t+1],q[t]);adv=(score-V[t+1,:,:,None])/self.entropy
            if self.rho is None:
                lse=logsumexp(adv,axis=-1,keepdims=True)
                pol[t]=np.exp(adv-lse)
                hj=self.entropy*lse[...,0]
            else:
                z=adv+np.log(self.rho)[None,None,:]
                lse=logsumexp(z,axis=-1,keepdims=True)
                pol[t]=np.exp(z-lse)
                feasible=(score>-1e5)
                norm=np.sum(feasible*self.rho[None,None,:],axis=-1)
                hj=self.entropy*(lse[...,0]-np.log(norm))
            V[t]=V[t+1]+self.dt*(self.gen(V[t+1])+self.lam*hj)
        return V,pol

    def forward(self,q,pol):
        mass=np.zeros((self.nt+1,self.nb,self.np))
        jj=min(int(np.floor(self.p0/self.h)),self.np-2)
        f=(self.p0-self.p[jj])/self.h
        mass[0,-1,jj]=1.-f;mass[0,-1,jj+1]=f
        qnew=np.zeros_like(q)
        jump=self.lam*self.dt
        for t in range(self.nt):
            m=mass[t]; nxt=m.copy();qnew[t]=(m[:,:,None]*pol[t]).sum((0,1))
            up=self.dt*self.up[:-1][None,:]*m[:,:-1];dn=self.dt*self.down[1:][None,:]*m[:,1:]
            nxt[:,:-1]-=up;nxt[:,1:]+=up;nxt[:,1:]-=dn;nxt[:,:-1]+=dn
            pw,g,bw,bl=self.market(q[t]);nxt-=jump*m
            for b in range(self.nb):
                for a in range(min(b,2)+1):
                    amt=jump*m[b]*pol[t,b,:,a]
                    if a==0: nxt[b]+=amt;continue
                    for prob,belief,nb in ((pw[:,a],bw[:,a],b-a),(1-pw[:,a],bl[:,a],b)):
                        src=amt*prob;j,fr=self.locate(belief)
                        np.add.at(nxt[nb],j,src*(1-fr));np.add.at(nxt[nb],j+1,src*fr)
            mass[t+1]=nxt
        return mass,qnew

    def phi(self,q):return self.forward(q,self.backward(q)[1])[1]

    def monetary(self,q,policy=None):
        v=np.zeros((self.nt+1,self.nb,self.np))
        for t in range(self.nt-1,-1,-1):
            sc=self.scores(v[t+1],q[t])
            if policy is None: pay=np.max(sc,axis=-1)-v[t+1]
            else:
                pay=np.sum(policy[t]*np.where(sc<-1e5,0,sc-v[t+1,:,:,None]),axis=-1)
            v[t]=v[t+1]+self.dt*(self.gen(v[t+1])+self.lam*pay)
        jj=min(int(np.floor(self.p0/self.h)),self.np-2)
        fr=(self.p0-self.p[jj])/self.h
        return float((1-fr)*v[0,-1,jj]+fr*v[0,-1,jj+1])

    def solve(self,seed=0,tol=1e-11,maxiter=500,damping=.7):
        q=np.random.default_rng(seed).dirichlet([1,1,1],size=self.nt)
        for k in range(maxiter):
            r=self.phi(q)
            err=np.max(np.abs(q-r))
            if err<tol:break
            q=(1-damping)*q+damping*r
        return q,err,k+1

    def branch_slack(self,q):
        # closest interior posterior distance to a belief grid node (not at boundary p=0 or 1)
        mins=[]
        for t in range(self.nt):
            _,_,bw,bl=self.market(q[t])
            for a in (1,2):
                for zz,posterior in (('win',bw[1:-1,a]),('loss',bl[1:-1,a])):
                    j=np.rint(posterior/self.h)
                    margin=np.abs(posterior/self.h-j)*self.h
                    ii=np.argmin(margin)
                    mins.append((float(margin[ii]),t,a,zz,int(ii)+1,float(posterior[ii])))
        return sorted(mins,key=lambda x:x[0])[:12]

if __name__=='__main__':
    import json
    m=Model();q,e,n=m.solve();print('residual',e,'steps',n,'initial',q[0],'final',q[-1]);print('branches',m.branch_slack(q)[:5]);print('mass dev',abs(m.forward(q,m.backward(q)[1])[0].sum(axis=(1,2))-1).max())

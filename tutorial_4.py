
# coding: utf-8

# In[1]:


#problem1
from __future__ import division, print_function
import numpy as np
n= 101
tfin= 2*np.pi
dt= tfin/(n-1)
s= np.arange(n)
y= np.sinc(dt*s)
fy= np.fft.fft(y)
wps= np.linspace(0,2*np.pi,n+1)[:-1]
basis= 1.0/n*np.exp(1.0j * wps * s[:,np.newaxis])
recon_y= np.dot(basis,fy)
yerr= np.max(np.abs(y-recon_y))

print('yerr:',yerr)

lin_fy= np.linalg.solve(basis,y)
fyerr= np.max(np.abs(fy-lin_fy))

print('fyerr',fyerr)


# In[24]:


#problem 2
import numpy as np

def simulate_lorentz(x,a=2.5,b=1,c=5):#a = sig, b=amp, c=cent

    dat=a/(b+(x-c)**2)
    dat+=np.random.randn(x.size)
    return dat

def get_trial_offset(sigs):
    return sigs*np.random.randn(sigs.size)


class Lorentz:

    def __init__(self,x,a=2.5,b=1,c=5,offset=0):

        self.x=x
        self.y=simulate_lorentz(x,a,b,c)+offset
        self.err=np.ones(x.size)
        self.a=a
        self.b=b
        self.c=c
        self.offset=offset

    def get_chisq(self,vec):
        a=vec[0]
        b=vec[1]
        c=vec[2]
        off=vec[3]
        pred=off+a/(b+(self.x-c)**2)
        chisq=np.sum(  (self.y-pred)**2/self.err**2)
        return chisq

def run_mcmc(data,start_pos,nstep,scale=None):
    nparam=start_pos.size
    params=np.zeros([nstep,nparam+1])
    params[0,0:-1]=start_pos
    cur_chisq=data.get_chisq(start_pos)
    cur_pos=start_pos.copy()

    if scale==None:

        scale=np.ones(nparam)

    for i in range(1,nstep):
        new_pos=cur_pos+get_trial_offset(scale)
        new_chisq=data.get_chisq(new_pos)

        if new_chisq<cur_chisq:

            accept=True

        else:

            delt=new_chisq-cur_chisq
            prob=np.exp(-0.5*delt)

            if np.random.rand()<prob:

                accept=True

            else:

                accept=False

        if accept: 

            cur_pos=new_pos
            cur_chisq=new_chisq

        params[i,0:-1]=cur_pos
        params[i,-1]=cur_chisq
    return params


if __name__=='__main__':

 #get a realization of a gaussian, with noise added
    x=np.arange(-5,5,0.01)
    dat=Lorentz(x,b=2.5)

    #pick a random starting position, and guess some errors
    guess=np.array([0.3,1.2,0.3,-0.2])
    scale=np.array([0.1,0.1,0.1,0.1])
    nstep=100000
    chain=run_mcmc(dat,guess,nstep,scale)
    nn=np.round(0.2*nstep)
    chain=chain[nn:,:]
    #pull true values out, compare to what we got
    param_true=np.array([dat.a,dat.b,dat.c,dat.offset])

    for i in range(0,param_true.size):

        val=np.mean(chain[:,i])

        scat=np.std(chain[:,i])

print ([param_true[i],val,scat])


# In[21]:


#problem3
import numpy

from matplotlib import pyplot as plt

class advect:

    def __init__(self,npart=300,u=1.0,dx=1.0):

        x=numpy.zeros(npart)

        x[npart/3:2*npart/3]=1.0;

        self.x=x

        self.u=u

        self.dx=dx

    def get_bc_periodic(self):


        self.x[0]=self.x[-2]

        self.x[-1]=self.x[1]

    def update(self,dt=1.0):

        self.get_bc_periodic()

        delt=self.x[1:]-self.x[0:-1]

        self.x[1:-1]+=self.u*dt/self.dx*delt[1:]

if __name__=='__main__':

    stuff=advect()

    plt.ion()

    plt.plot(stuff.x)

    plt.show()

    for i in range(0,300):

        stuff.update()

        plt.clf()

        plt.plot(stuff.x)

        plt.draw()


# In[23]:


#bonus1 
import numpy

from matplotlib import pylab as plt

def simulate_gaussian(t,sig=0.5,amp=1,cent=0):

    dat=numpy.exp(-0.5*(t-cent)**2/sig**2)*amp

    dat+=numpy.random.randn(t.size)

    return dat

def get_trial_offset(sigs):

    return sigs*numpy.random.randn(sigs.size)

class Gaussian:

    def __init__(self,t,sig=0.5,amp=1.0,cent=0,offset=0):

        self.t=t

        self.y=simulate_gaussian(t,sig,amp,cent)+offset

        self.err=numpy.ones(t.size)

        self.sig=sig

        self.amp=amp

        self.cent=cent

        self.offset=offset



    def get_chisq(self,vec):

        sig=vec[0]

        amp=vec[1]

        cent=vec[2]

        off=vec[3]



        pred=off+amp*numpy.exp(-0.5*(self.t-cent)**2/sig**2)

        chisq=numpy.sum(  (self.y-pred)**2/self.err**2)

        return chisq

def run_mcmc(data,start_pos,nstep,scale=None):

    nparam=start_pos.size

    params=numpy.zeros([nstep,nparam+1])

    params[0,0:-1]=start_pos

    cur_chisq=data.get_chisq(start_pos)

    cur_pos=start_pos.copy()

    if scale==None:

        scale=numpy.ones(nparam)

    #to get the accept fraction, let's keep track of it while running the chain.

    tot_accept=0.0

    tot_reject=0.0

    for i in range(1,nstep):

        new_pos=cur_pos+get_trial_offset(scale)

        new_chisq=data.get_chisq(new_pos)

        if new_chisq<cur_chisq:

            accept=True

        else:

            delt=new_chisq-cur_chisq

            prob=numpy.exp(-0.5*delt)

            if numpy.random.rand()<prob:

                accept=True

            else:

                accept=False

        if accept:

            tot_accept=tot_accept+1

            cur_pos=new_pos

            cur_chisq=new_chisq

        else:

            tot_reject=tot_reject+1

        params[i,0:-1]=cur_pos

        params[i,-1]=cur_chisq

        accept_frac=tot_accept/(tot_accept+tot_reject)

    return params,accept_frac

if __name__=='__main__': 

    t=numpy.arange(-5,5,0.01)

    dat=Gaussian(t,amp=2.5)

    guess=numpy.array([0.3,1.2,0.3,-0.2])

    scale=numpy.array([0.1,0.1,0.1,0.1])


    nstep=1000

    chain,accept=run_mcmc(dat,guess,nstep,scale)

    nn=numpy.round(0.2*nstep)

    chain=chain[nn:,:]

    scale=numpy.std(chain[:,0:-1],0)


    nstep=30000

    chain,accept2=run_mcmc(dat,chain[-1,0:-1],nstep,scale)

    print ("old accept was: " + repr(accept))

    print ("new accept is: " + repr(accept2))


    param_true=numpy.array([dat.sig,dat.amp,dat.cent,dat.offset])

    for i in range(0,param_true.size):

        val=numpy.mean(chain[:,i])

        scat=numpy.std(chain[:,i])

        print ([param_true[i],val,scat])


# In[ ]:





# In[17]:


#problem6&7
import numpy

from matplotlib import pyplot as plt

class Fluid:

    def __init__(self,npix=200,gamma=5.0/3.0,bc_type='periodic'):

        self.rho=numpy.zeros(npix)

        self.p=numpy.zeros(npix)

        self.v=numpy.zeros(npix)

        self.rhoE=numpy.zeros(npix)

        self.P=numpy.zeros(npix)

        self.gradrho=numpy.zeros(npix)

        self.gradp=numpy.zeros(npix)

        self.gradrhoE=numpy.zeros(npix)

        self.gamma=gamma

        self.bc_type=bc_type

        self.n=npix

        self.dx=1.0/npix

    def ic_bullet(self):

        self.rho[:]=1.0

        self.rhoE[:]=1.0

        self.p[:]=0

        self.p[self.n/2:3*self.n/5]=1.0



    def ic_shock_tube(self):

        #set up classic Sod shock tube problem.

        self.rho[0:self.n/2]=1.0

        self.P[0:self.n/2]=1.0



        self.rho[self.n/2:]=0.125

        self.P[self.n/2:]=0.1

        self.p[:]=0

        #we really need the energy, not the pressure

        self.rhoE=1.0/(self.gamma-1.0)*self.P



    def get_velocity(self):

        #get the velocity, if density is zero, set velocity to be zero

        self.v[:]=0

        ii=self.rho>0

        self.v[ii]=self.p[ii]/self.rho[ii]

    def get_bc(self):

        if self.bc_type=='periodic':

            self.rho[0]=self.rho[-2]

            self.rho[-1]=self.rho[1]

            self.p[0]=self.p[-2]

            self.p[-1]=self.p[1]

            self.rhoE[0]=self.rhoE[-2]

            self.rhoE[-1]=self.rhoE[1]

            return

        if self.bc_type=='smooth':

            self.rho[0]=self.rho[1]

            self.rho[-1]=self.rho[-2]

            self.p[0]=self.p[1]

            self.p[-1]=self.p[-2]

            self.rhoE[0]=self.rhoE[1]

            self.rhoE[-1]=self.rhoE[-2]

            return    

        assert(1==0)  #why did we do this?  Tutorial problem 5

    def do_eos(self):

        thermal=self.rhoE-self.rho*0.5*self.v**2

        self.P=(self.gamma-1.0)*thermal

    def get_derivs(self):

        frho=self.p

        fp=self.p*self.v

        fE=self.v*self.rhoE+self.v*self.P

        

        drho=0*self.rho

        dp=0*self.p

        drhoE=0*self.rhoE

        for ii in range(1,self.n-1):            

            if self.v[ii]>0:

                drho[ii+1]+=frho[ii]

                drho[ii]-=frho[ii]

                dp[ii+1]+=fp[ii]

                dp[ii]-=fp[ii]

                drhoE[ii+1]+=fE[ii]

                drhoE[ii]-=fE[ii]

            if self.v[ii]<0:

                drho[ii-1]+=frho[ii]

                drho[ii]-=frho[ii]

                dp[ii-1]+=fp[ii]

                dp[ii]-=fp[ii]

                drhoE[ii-1]+=fE[ii]

                drhoE[ii]-=fE[ii]

        #Why is there a factor of 1/2 in the pressure gradient? Tutorial problem 6

        gradP=0.5*(self.P[2:]-self.P[0:-2])

        dp[1:-1]-=gradP

        self.gradrho=drho

        self.gradp=dp

        self.gradrhoE=drhoE

    def get_timestep(self,dt=0.1):


        c_s=numpy.sqrt(self.gamma*self.P/self.rho)
        
        
        vmax=numpy.max(numpy.abs(c_s)+numpy.abs(self.v))


        ts= self.dx/vmax*dt

        return ts

    def take_step(self):

        self.get_bc()

        self.get_velocity()

        self.do_eos()

        self.get_derivs()

        dt=self.get_timestep()

    
        self.rho+=self.gradrho*dt

        self.p+=self.gradp*dt

        self.rhoE+=self.gradrhoE*dt

        return dt

def pp(x):

    plt.clf()

    plt.plot(x)

    plt.show()

if __name__=='__main__':

    plt.ion()

    fluid=Fluid(npix=500,bc_type='smooth')

    plt.plot(fluid.p)

    fluid.ic_shock_tube()

    plt.plot(fluid.rho)

    plt.draw()

    fluid.take_step()

    for i in range(0,1000):

        tt=0

        while (tt<0.02):

            tt=tt+fluid.take_step()

        plt.clf()

        plt.plot(fluid.rho)

        plt.draw()

    plt.savefig('tut3.png')


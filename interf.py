import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate

def make_w_axis(w_min=0,w_step=4,w_max=4096,taxis=0):
    #returns a frequency axis (cm-1) given either w min, step, max or a time axis in seconds
    if taxis==0:
        waxis = np.arange(w_min,w_max,w_step,dtype=float)
    else:
        dt = taxis[1] - taxis[0]
        waxis = np.linspace(-1/dt, 1/dt, len(taxis))
    return waxis

def gaussian(w, mu, sigma):
    #returns a gaussian centered at mu with width sigma
    #unnormalized
    return np.exp(-np.divide(np.power(w-mu,2.0),2.0*sigma*sigma))

def main(args):
    ##Define BBIR spectrum
    c = 29979245800 #speed of light in cm/s
    _to_ps = 1e12 # convert from s to ps
    w = make_w_axis(0,1,8192)#define frequency axis
    mu1, mu2 = 1320, 1680 #peak centers for the spectrum
    sig1, sig2 = 130, 130 #peak widths
    sig_w = 0.85*gaussian(w,mu1,sig1) + 1.0*gaussian(w,mu2,sig2) #define the spectrum

    ##Initlize Plot
    fig = plt.figure(dpi=600, figsize=[12, 24], num=1) #initialize figure A4 size
    lw = 0.1 #default linewidth
    lfs = 'xx-small'#legend font size
    wlabel = 'wavenumber (cm^{-1})' #w axis label
    Elabel = 'electric field (AU)' #electric field label
    tlabel = 'time (s)'
    gs = mpl.gridspec.GridSpec(2,3, figure=fig) #make two rows and three columns

    #Make first subplot
    ax = fig.add_subplot(gs[0,0], title='|E(w)|^2')
    ax.plot(w, sig_w, 'b-', lw=lw)
    ax.set(xlabel=wlabel, ylabel=Elabel)
    ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_axisbelow(True)
    plt.xlim(left=800, right=2200)
    plt.ylim(bottom=0, top=1)

    ##Define BBIR E field from FFT of spectrum
    t = np.arange(len(w)) #define time axis
    t = np.divide(t - np.mean(t), c*np.amax(w)) #set the units
    t0 = np.argmin(np.abs(t))
    dt = t[1]-t[0]
    
    if args.debug:
        print('Made time axis with ranging from {:.2f} to {:.2f} ps with center at index {} and delta_t of {:.2f} ps'.format(np.min(t)*_to_ps,np.max(t)*_to_ps,t[t0],dt*_to_ps))
    sig_t = np.fft.ifft(np.sqrt(sig_w)) #complex (transform-limited) electric field in the time domain (note this sqrt was not in the MATLAB version)
    sig_t = np.fft.fftshift(sig_t)

    #Make second subplot
    ax = fig.add_subplot(gs[0,1], title='E(t)')
    ax.plot(t, sig_t.real, 'b-', lw=lw, label='real')
    ax.plot(t, sig_t.imag, 'r', lw=lw, label='imag')
    ax.plot(t, np.abs(sig_t), 'k', lw=lw, label='abs')
    ax.set(xlabel=tlabel, ylabel=Elabel)
    ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.xlim(left=t[t0]-0.2e-12, right=t[t0]+0.2e-12)

    #Make chirped pulses
    c1=2e10#chirp for arm1
    c2=1e10#chirp for arm2
    sig_1=np.multiply(np.fft.ifft(np.sqrt(sig_w)), np.exp(-1j*(c1*np.multiply(w,t)+c1*np.multiply(w,t**2))))
    sig_1 = np.real(np.fft.fftshift(sig_1))#sig_1 has a fixed phase, so we just keep the real part
    sig_2_0=np.multiply(np.fft.ifft(np.sqrt(sig_w)), np.exp(-1j*(c2*np.multiply(w,t)+c2*np.multiply(w,t**2))))#sig_1 has a changing phase, so we keep the full complex part
    sig_2_0 = np.real(np.fft.fftshift(sig_2_0))
    
    #Make third subplot
    ax = fig.add_subplot(gs[0,2], title='E(t)')
    ax.plot(t, sig_1.real, 'b-', lw=lw, label='Re[E1]')
    ax.plot(t, sig_2_0.real, 'r-', lw=lw, label='Re[E2]')
    ax.set(xlabel=tlabel, ylabel=Elabel)
    ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.xlim(left=t[t0-100], right=t[t0+100])

    LIVE=False
    n=0
    scan_range = np.arange(-300,300,1)
    interf_t = np.zeros(len(scan_range))
    
    for t_n in scan_range:
        sig_2 = np.roll(sig_2_0,t_n)#np.roll is used to scan the time axis
        interf_t[n] = integrate.simps(np.power(np.abs(sig_1+sig_2),2.0),t) #the interferogram is the frequency-integrated, magnitude squared of the sum of the electric fields
        n = n + 1

        if LIVE or t_n==scan_range[-1]:
            #Update third subplot
            ax.clear()
            ax = fig.add_subplot(gs[0,2], title='E(t)')
            ax.plot(t, np.real(sig_1+sig_2),'m-',lw=lw, label='Re[E1+E2]')
            ax.plot(t, np.real(sig_1),'r-',lw=lw, label='Re[E1]')
            ax.plot(t, np.real(sig_2),'b-',lw=lw, label='Re[E2]')
            ax.set(xlabel=tlabel, ylabel=Elabel)
            ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.xlim(left=t[t0-500], right=t[t0+500])    

            #Make forth subplot
            ax = fig.add_subplot(gs[1,0], title='E(t)')
            ax.plot(t, np.real(sig_1+sig_2),'m-',lw=lw, label='Re[E1+E2]')
            ax.plot(t, np.real(sig_1),'r-',lw=lw, label='Re[E1]')
            ax.plot(t, np.real(sig_2),'b-',lw=lw, label='Re[E2]')
            ax.plot(t, sig_2.real, 'r-', lw=lw, label='Re[E2]')
            ax.set(xlabel=tlabel, ylabel=Elabel)
            ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.xlim(left=t[t0-100], right=t[t0+100])
            
            #Make fifth subplot
            ax = fig.add_subplot(gs[1,1], title='E(t)')
            ax.plot(t[0:n], interf_t[0:n], lw=lw, label='\int dt |E1+E2|^2')
            ax.set(xlabel=tlabel, ylabel='Interferogram Signal')
            ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_axisbelow(True)
            plt.tight_layout()
            if LIVE: plt.show()

    w_interf = np.arange(len(interf_t))
    w_interf = w_interf - np.mean(w_interf)
    w_interf = w_interf / np.max(w_interf)/2
    w_interf = w_interf/dt/c
        
    #Make fifth subplot
    ax = fig.add_subplot(gs[1,2], title='E(t)')
    interferogram_spectrum = np.abs(np.fft.fftshift(np.fft.fft(interf_t)))
    interferogram_spectrum = np.sqrt(interferogram_spectrum)

    ax.plot(w_interf, interferogram_spectrum, lw=lw, label='FFT of interferogram')
    ax.set(xlabel=tlabel, ylabel='FFT Power')
    ax.legend(fontsize=lfs, loc='best', bbox_to_anchor=[1, 0, 0.5, 1])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.xlim(left=-3200, right=3200)
    plt.ylim(bottom=0, top=1.1*np.amax(interferogram_spectrum[w_interf>1000]))

    fs = 8 #default fontsize
    plt.rc('font', size=fs)
    plt.rc('lines', linewidth=1)
    #plt.show()
    plt.savefig('test.png', orientation='landscape')
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true', help='turns on debugging output')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(str(e),0)
        raise e

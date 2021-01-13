import os
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate


c = 29979245800.0 #speed of light in cm/s
_fs_per_nm = 3.33564095e-3 #time it takes light to travel 1 nm in fs
_s_to_ps = 1.0e12 # convert from s to ps
_fs_to_s = 1.0e-15

def make_w_axis(w_min=0,w_step=4,w_max=4096,taxis=0):
    #returns a frequency axis (cm-1) given either w min, step, max or a time axis in seconds
    if taxis==0:
        waxis = np.arange(w_min,w_max,w_step,dtype=float)
    else:
        dt = taxis[1] - taxis[0]
        waxis = np.linspace(-1.0/dt, 1.0/dt, len(taxis))
    return waxis

def gaussian(w, mu, sigma):
    #returns a gaussian centered at mu with width sigma
    #unnormalized
    return np.exp(-np.divide(np.power(w-mu,2.0),2.0*sigma*sigma))

def loadInterferograms(directory):
    template = 'scan {}.txt'
    data_dict = {}
    filenum = 1
    while True:
        try:
            filename = template.format(filenum)
            temp_mct, temp_stage = np.loadtxt(directory+filename, unpack=True)
            data_dict['mct_'+str(filenum-1)] = temp_mct
            data_dict['t_'+str(filenum-1)] = temp_stage*2*_fs_per_nm
            filenum += 1
        except:
            data_dict['N_scans'] = filenum-2
            print('Found {} files'.format(filenum-2))
            return data_dict

def main(args):
    ##Define BBIR spectrum
    data_dict = loadInterferograms(args.dir)
    N_scans = data_dict['N_scans']
    min_t = np.zeros((N_scans),dtype=float)
    max_t = np.zeros((N_scans),dtype=float)
    mean_dt = np.zeros((N_scans),dtype=float)

    for n in np.arange(N_scans):
        t_temp = data_dict['t_'+str(n+1)]
        min_t[n] = np.amin(t_temp)
        max_t[n] = np.amax(t_temp)
        mean_dt[n] = np.mean(np.diff(t_temp))
    dt_global = np.mean(mean_dt)/2.0
    t_global = np.arange(np.amax(min_t), np.amin(max_t), dt_global)

    mct_mean = np.zeros_like(t_global)
    for n in np.arange(N_scans):
        t_temp = data_dict['t_'+str(n+1)]
        mct_temp = data_dict['mct_'+str(n+1)]
        mct_interp = np.interp(t_global, t_temp, mct_temp)
        data_dict['mct_'+str(n+1)+'interp'] = mct_interp
        mct_mean = mct_mean + mct_interp/N_scans

    ##Initlize Plot
    fig = plt.figure(dpi=600, figsize=[12, 5], num=1) #initialize figure A4 size
    lw = 0.5 #default linewidth
    lfs = 'xx-small'#legend font size
    if(args.loadref):
        gs = mpl.gridspec.GridSpec(2,2, figure=fig)
    else:
        gs = mpl.gridspec.GridSpec(1,2, figure=fig)

    #Plot mean interferogram
    ax = fig.add_subplot(gs[0,0], title='mean interferogram')
    ax.plot(t_global, mct_mean, 'b-', lw=lw)
    ax.set(xlabel='time (fs)')
    ax.set(ylabel='Interferogram Intensity')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_axisbelow(True)
    plt.tight_layout()

    #Calculate FFT spectrum
    mct_fft = np.abs(np.fft.fft(mct_mean))
    mct_fft = np.fft.fftshift(mct_fft)
    w_fft = np.fft.fftfreq(n=mct_fft.size, d=dt_global*_fs_to_s)
    w_fft = np.fft.fftshift(w_fft)*2.0/c

    #Perform spectrum statistics
    w_stats = np.arange(1000,2000,1,dtype=float)
    spec_stats = np.interp(w_stats, w_fft, mct_fft)
    spec_center = np.divide(np.trapz(np.multiply(w_stats,spec_stats)), np.trapz(spec_stats))
    spec_max = np.amax(spec_stats)
    w_max = w_stats[np.argmax(spec_stats)]
    print('Spectrum center at {:.2f} and max at {:.2f}, {:.2f}'.format(spec_center,w_max,spec_max))

    #Load reference if asked
    if(args.loadref):
        print('loading reference')
        temp_w, temp_fft = np.loadtxt(args.refname)
        ref_fft = np.interp(w_fft, temp_w, temp_fft)

    #Plot FFT spectrum
    ax = fig.add_subplot(gs[0,1], title='FFT mean interferogram')
    ax.plot(w_fft, mct_fft, 'b-', lw=lw)
    if(args.loadref): 
        ax.plot(w_fft, ref_fft, 'r-', lw=lw)
    ax.set(xlabel='Wavenumber ($cm^{-1}$)')
    ax.set(ylabel='Interferogram Power (AU)')
    plt.xlim(left=0, right=3000)
    plt.ylim(bottom=0, top=1.1*spec_max)

    if(args.loadref):
        #Calculate difference spectrum
        diff_spec = ref_fft - mct_fft
        w_min, w_max = 1000, 2000
        scale_range = np.logical_and(w_fft > w_min, w_fft < w_max)

        #Plot difference spectrum
        ax = fig.add_subplot(gs[1,0], title='FFT mean interferogram')
        ax.plot(w_fft, diff_spec, 'b-', lw=lw)
        ax.set(xlabel='Wavenumber ($cm^{-1}$)')
        ax.set(ylabel='Interferogram Power Diff (AU)')
        plt.xlim(left=w_min, right=w_max)
        plt.ylim(bottom=1.1*np.amin(diff_spec[scale_range]), top=1.1*np.amax(diff_spec[scale_range]))

        
        #Calculate normalized difference spectrum
        diff_spec_n = -np.log(np.divide(mct_fft, ref_fft))

        #Plot ref normalized difference spectrum
        ax = fig.add_subplot(gs[1,1], title='FFT mean interferogram')
        ax.plot(w_fft, diff_spec_n, 'b-', lw=lw)
        ax.plot(w_fft, np.zeros_like(w_fft), 'k-', lw=lw)
        ax.set(xlabel='Wavenumber ($cm^{-1}$)')
        ax.set(ylabel='Interferogram Power Diff Norm (AU)')
        plt.xlim(left=w_min, right=w_max)
        plt.ylim(bottom=1.1*np.amin(diff_spec_n[scale_range]), top=1.1*np.amax(diff_spec_n[scale_range]))

    fs = 8 #default fontsize
    plt.rc('font', size=fs)
    plt.rc('lines', linewidth=1)
    #plt.show()
    if args.loadref:
        plt.figtext(0.01, 0.99, str(args.dir)+' ref '+args.refname, horizontalalignment='left')
    else:
        plt.figtext(0.01, 0.99, str(args.dir), horizontalalignment='left')
    
    out_num = 1
    while True:
        out_fname = args.outname + 'outflip'+str(out_num)+'.png'
        if os.path.exists(out_fname):
            out_num += 1
        else:
            break
    plt.savefig(out_fname, bbox_inches='tight')
    print('saved ' + str(args.dir) + ' analysis to ' + out_fname)
    if not args.loadref:
        np.savetxt('out'+str(out_num)+'.txt', (w_fft, mct_fft))
        print('saved data to ' + 'out'+str(out_num)+'.txt')
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true', help='turns on debugging output')
    parser.add_argument('--dir', help='which directory to analyze (no default)', required=True)
    parser.add_argument('--loadref', action='store_true', help='loads and subtracts reference')
    parser.add_argument('--refname', help='reference name')
    parser.add_argument('--outname', default='', help='output name to prepend')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(str(e),0)
        raise e

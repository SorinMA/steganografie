from scipy.io import wavfile
import wave, struct, math
import numpy as np

SAMPLE_RATE = 44100 # Hz
NYQUIST_RATE = SAMPLE_RATE / 2.0
FFT_LENGTH = 512

def filter_pass(cutoff1, cutoff2=0, what_filter_do_we_got=0): # 0 lowpass, 1 highpass, 2 bandpasss # MARTINESCU
        cutoff1 /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))
        cutoff2 /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))

        # create FFT filter mask
        mask = []
        negatives = []
        l = FFT_LENGTH // 2
        for f in range(0, l+1):
                rampdown = 1.0
                if not what_filter_do_we_got:
                    if f > cutoff1:
                            rampdown = 0
                    mask.append(rampdown)
                    if f > 0 and f < l:
                            negatives.append(rampdown)
                elif what_filter_do_we_got == 1:
                    if f < cutoff1:
                            rampdown = 0
                    mask.append(rampdown)
                    if f > 0 and f < l:
                            negatives.append(rampdown)
                else:
                    if f < cutoff1 or f > cutoff2:
                            rampdown = 0
                    mask.append(rampdown)
                    if f > 0 and f < l:
                            negatives.append(rampdown)
                            

        negatives.reverse()
        mask = mask + negatives

        # Convert FFT filter mask to FIR coefficients
        impulse_response = np.fft.ifft(mask).real.tolist()

        # swap left and right sides
        left = impulse_response[:FFT_LENGTH // 2]
        right = impulse_response[FFT_LENGTH // 2:]
        impulse_response = right + left

        b = FFT_LENGTH // 2
        # apply triangular window function
        for n in range(0, b):
                    impulse_response[n] *= (n + 0.0) / b
        for n in range(b + 1, FFT_LENGTH):
                    impulse_response[n] *= (FFT_LENGTH - n + 0.0) / b

        return impulse_response

def lowpass(original, cutoff): # MARTINESCU
        coefs = filter_pass(cutoff1=cutoff, what_filter_do_we_got=0)
        return np.convolve(original, coefs)

def fromNto1Ch(inpt, filtr): # MARTINESCU
    rate, audio = wavfile.read(inpt)
    try:
        nrOfChannels = len(audio[0])   
        monoChFrame = (np.array([0]*len(audio[...,0]))).astype(np.float)
        for i in range(nrOfChannels):
            monoChFrame += 1/nrOfChannels*(audio[...,i]).astype(np.float)
        if filtr == True:
            monoChFrame = 2*lowpass(1/2*monoChFrame,4500)
        wavfile.write(inpt+'aux.wav',rate,monoChFrame.astype(np.int16))
        print('Step - ok')
    except:
        if filtr == True:
            audio = 2*lowpass(1/2*audio,4500)
        wavfile.write(inpt+'aux.wav',rate,audio.astype(np.int16))
        print('Step - ok but exception')

def am_modulation(inpt1, inpt2, output): # MARTINESCU
    baseband1 = wave.open(inpt1+'aux.wav', "r")
    baseband2 = wave.open(inpt2+'aux.wav', "r")
    amsc = wave.open(output, "w")
    for f in [amsc]:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
    
    contor = 0
    for n in range(0, baseband1.getnframes()):
        base = (struct.unpack('h', baseband1.readframes(1))[0] / 32768.0)/2
        if contor < baseband2.getnframes():
            encodFrame = struct.unpack('h', baseband2.readframes(1))[0] / 32768.0
            carrier_sample = math.cos(22050.0 * (contor / 44100.0) * math.pi * 2)
            base += encodFrame * carrier_sample / 4
            contor += 1
        amsc.writeframes(struct.pack('h', int(base * 32767)))

def mod_am(inpt1, inpt2, output): # MARTINESCU
    fromNto1Ch(inpt2, True)
    fromNto1Ch(inpt1, False)
    am_modulation(inpt1, inpt2, output)


def fromNto1Ch_low_midi_high(inpt, filtr, l_m_h=0,c1=4500, c2=0): # MARTINESCU
    rate, audio = wavfile.read(inpt)
    var = 'aux.wav'
   
    if l_m_h == 0:
        var = '_L_'+var
    elif l_m_h == 2:
        var = '_M_'+var
    else:
        var = '_H_'+var
        
    try:
        nrOfChannels = len(audio[0])   
        monoChFrame = (np.array([0]*len(audio[...,0]))).astype(np.float)
        for i in range(nrOfChannels):
            monoChFrame += 1/nrOfChannels*(audio[...,i]).astype(np.float)
        if filtr == True:
            monoChFrame = np.convolve(monoChFrame, filter_pass(cutoff1=c1, cutoff2=c2, what_filter_do_we_got=l_m_h))
        wavfile.write(inpt+var,rate,monoChFrame.astype(np.int16))
        print('Step - ok')
    except:
        if filtr == True:
            audio = np.convolve(audio, filter_pass(cutoff1=c1, cutoff2=c2, what_filter_do_we_got=l_m_h))
        wavfile.write(inpt+var,rate,audio.astype(np.int16))
        print('Step - ok but exception')


def low_midi_high(inpt, l_m_h=0,c1=4500, c2=0): # MARTINESCU
    rate, audio = wavfile.read(inpt)
    var = 'aux.wav'
   
    if l_m_h == 0:
        var = '_L_'+var
    elif l_m_h == 2:
        var = '_M_'+var
    else:
        var = '_H_'+var
        
    audio_aux = ((np.array([0]*2*len(audio[...,0]))).astype(np.float)).reshape(len(audio[...,0]), len(audio[0]))
    print(len(audio[...,0]), len(audio[0]))
    print(audio_aux)
    print(len(audio))
    for i in range(len(audio[0])):
        audio_aux[...,i] = np.convolve(audio[...,i], filter_pass(cutoff1=c1, cutoff2=c2, what_filter_do_we_got=l_m_h))[:len(audio_aux[...,i])]
    wavfile.write(inpt+var,rate,audio_aux.astype(np.int16))
    print('Step - ok')

def get_plot(file_wav): # MARTINESCU
  import matplotlib.pyplot as plt
  from scipy.fftpack import fft
  from scipy.io import wavfile # get the api
  fs, data = wavfile.read(file_wav) # load the data
  a = data.T # this is a two channel soundtrack, I get the first track
  b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
  c = fft(b) # calculate fourier transform (complex numbers list)
  d = len(c)/2  # you only need half of the fft list (real signal symmetry)
  plt.plot(abs(c[:int(d-1)]),'r') 
  plt.show()

  plt.plot(b[:int(d-1)],'r') 
  plt.show()

from easygui import *
import sys

def get_plot_v2(file_wav): # MARTINESCU
    import tkinter
    from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    import numpy as np

    from matplotlib.figure import Figure
    from scipy.fftpack import fft
    from scipy.io import wavfile # get the api
    fs, data = wavfile.read(file_wav) # load the data
    try:
        from operator import add
        a = list(map(add, data.T[0], data.T[1])) # this is a two channel soundtrack, I get the first track
    except:
        print('EX')
        a = data.T[0]
    b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b) # calculate fourier transform (complex numbers list)
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)
    
    root = tkinter.Tk()
    root.wm_title((file_wav.split('\\'))[-1])
    
    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(211).plot(abs(c[:int(d-1)]),'r')
    fig.add_subplot(212).plot(b[:int(d-1)],'r')
    
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


    def on_key_press(event): 
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)


    canvas.mpl_connect("key_press_event", on_key_press)


    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)

    tkinter.mainloop()


def get_plot_v2_l_m_h(file_wav): # ORBISOR
    import tkinter
    from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    import numpy as np

    from matplotlib.figure import Figure
    from scipy.fftpack import fft
    from scipy.io import wavfile # get the api
    fs1, data1 = wavfile.read((file_wav.split('\\'))[-1]+ '_L_aux.wav') # load the data
    a1 = data1.T # this is a two channel soundtrack, I get the first track
    b1=[(ele/2**8.)*2-1 for ele in a1] # this is 8-bit track, b is now normalized on [-1,1)
    c1 = fft(b1) # calculate fourier transform (complex numbers list)
    d1 = len(c1)/2  # you only need half of the fft list (real signal symmetry)
    
    fs2, data2 = wavfile.read((file_wav.split('\\'))[-1]+ '_M_aux.wav') # load the data
    a2 = data2.T # this is a two channel soundtrack, I get the first track
    b2=[(ele/2**8.)*2-1 for ele in a2] # this is 8-bit track, b is now normalized on [-1,1)
    c2 = fft(b2) # calculate fourier transform (complex numbers list)
    d2 = len(c2)/2  # you only need half of the fft list (real signal symmetry)
    
    fs3, data3 = wavfile.read((file_wav.split('\\'))[-1]+ '_H_aux.wav') # load the data
    a3 = data3.T # this is a two channel soundtrack, I get the first track
    b3=[(ele/2**8.)*2-1 for ele in a3] # this is 8-bit track, b is now normalized on [-1,1)
    c3 = fft(b3) # calculate fourier transform (complex numbers list)
    d3 = len(c3)/2  # you only need half of the fft list (real signal symmetry)
    
    root = tkinter.Tk()
    root.wm_title((file_wav.split('\\'))[-1] + 'Low_Mid_Hi')
    
    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(611).plot(abs(c1[:int(d1-1)]),'r')
    fig.add_subplot(612).plot(b1[:int(d1-1)],'r')
    
    fig.add_subplot(613).plot(abs(c2[:int(d2-1)]),'r')
    fig.add_subplot(614).plot(b2[:int(d2-1)],'r')
    
    fig.add_subplot(615).plot(abs(c3[:int(d3-1)]),'r')
    fig.add_subplot(616).plot(b3[:int(d3-1)],'r')
    
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)


    canvas.mpl_connect("key_press_event", on_key_press)


    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)

    tkinter.mainloop()



def get_plot_v2_l_m_h_lr(file_wav,lr): # DANOIU
    tt = 'Left'
    if lr:
        tt = 'Right'
    import tkinter
    from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    import numpy as np

    from matplotlib.figure import Figure
    from scipy.fftpack import fft
    from scipy.io import wavfile # get the api
    fs1, data1 = wavfile.read((file_wav.split('\\'))[-1]+ '_L_aux.wav') # load the data
    a1 = data1.T[lr] # this is a two channel soundtrack, I get the first track
    b1=[(ele/2**8.)*2-1 for ele in a1] # this is 8-bit track, b is now normalized on [-1,1)
    c1 = fft(b1) # calculate fourier transform (complex numbers list)
    d1 = len(c1)/2  # you only need half of the fft list (real signal symmetry)
    
    fs2, data2 = wavfile.read((file_wav.split('\\'))[-1]+ '_M_aux.wav') # load the data
    a2 = data2.T[lr] # this is a two channel soundtrack, I get the first track
    b2=[(ele/2**8.)*2-1 for ele in a2] # this is 8-bit track, b is now normalized on [-1,1)
    c2 = fft(b2) # calculate fourier transform (complex numbers list)
    d2 = len(c2)/2  # you only need half of the fft list (real signal symmetry)
    
    fs3, data3 = wavfile.read((file_wav.split('\\'))[-1]+ '_H_aux.wav') # load the data
    a3 = data3.T[lr] # this is a two channel soundtrack, I get the first track
    b3=[(ele/2**8.)*2-1 for ele in a3] # this is 8-bit track, b is now normalized on [-1,1)
    c3 = fft(b3) # calculate fourier transform (complex numbers list)
    d3 = len(c3)/2  # you only need half of the fft list (real signal symmetry)
    
    root = tkinter.Tk()
    
    root.wm_title(tt+ (file_wav.split('\\'))[-1] + 'Low_Mid_Hi'+tt)
    
    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(611).plot(abs(c1[:int(d1-1)]),'r')
    fig.add_subplot(612).plot(b1[:int(d1-1)],'r')
    
    fig.add_subplot(613).plot(abs(c2[:int(d2-1)]),'r')
    fig.add_subplot(614).plot(b2[:int(d2-1)],'r')
    
    fig.add_subplot(615).plot(abs(c3[:int(d3-1)]),'r')
    fig.add_subplot(616).plot(b3[:int(d3-1)],'r')
    
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)


    canvas.mpl_connect("key_press_event", on_key_press)


    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)

    tkinter.mainloop()

filename1 = ''
filename2 = ''
def get_name_1(): # DANOIU
    global filename1
    filename1 = fileopenbox()

def get_name_2(): # DANOIU
    global filename2
    filename2 = fileopenbox()

def get_plot_1(): # DANOIU
    global filename1
    get_plot_v2(filename1)

def get_plot_2(): # DANOIU
    global filename2
    get_plot_v2(filename2)
    
def get_plot_12(): # DANOIU
    global filename1
    global filename2
    aux_file = (filename1.split('\\'))[-1]+(filename2.split('\\'))[-1]
    try:
        wavfile.read(aux_file)
    except:
        mod_am(filename1, filename2, aux_file)
    get_plot_v2(aux_file)
   

def get_l_m_h(filename): # DANOIU
    fromNto1Ch_low_midi_high(filename, True,0, 255,0)
    fromNto1Ch_low_midi_high(filename, True,2, 256,2000)
    fromNto1Ch_low_midi_high(filename, True,1, 2001,0)
    get_plot_v2_l_m_h(filename)

def get_l_m_h_l(filename): # DANOIU
    print('LR')
    low_midi_high(filename,0, 255,0)
    low_midi_high(filename,2, 256,2000)
    low_midi_high(filename,1, 2001,0)
    print('L')
    get_plot_v2_l_m_h_lr(filename, 0)
   

def get_l_m_h_r(filename): # DANOIU
    print('LR')
    low_midi_high(filename,0, 255,0)
    low_midi_high(filename,2, 256,2000)
    low_midi_high(filename,1, 2001,0)
    print('R')
    get_plot_v2_l_m_h_lr(filename, 1)
    
def get_l_m_h1l(): # DANOIU
    global filename1
    rate, audio = wavfile.read(filename1)
    try:
        nrOfChannels = len(audio[0]) 
        rate, audio = None, None
        print('hi there')
        get_l_m_h_l(filename1)
    except Exception as e:
        rate, audio = None, None
        print('hi there_x', e, audio)
        get_l_m_h(filename1)

def get_l_m_h2l(): # DANOIU
    global filename2
    rate, audio = wavfile.read(filename2)
    try:
        nrOfChannels = len(audio[0]) 
        rate, audio = None, None
        get_l_m_h_l(filename2)
    except:
        rate, audio = None, None
        get_l_m_h(filename2)
        
        
def get_l_m_h1r(): # DANOIU
    global filename1
    rate, audio = wavfile.read(filename1)
    try:
        nrOfChannels = len(audio[0]) 
        rate, audio = None, None
        print('hi there')
        get_l_m_h_r(filename1)
    except Exception as e:
        rate, audio = None, None
        print('hi there_x', e, audio)
        get_l_m_h(filename1)

def get_l_m_h2r(): # DANOIU
    global filename2
    rate, audio = wavfile.read(filename2)
    try:
        nrOfChannels = len(audio[0]) 
        rate, audio = None, None
        get_l_m_h_r(filename2)
    except:
        rate, audio = None, None
        get_l_m_h(filename2)

def get_l_m_h12(): # DANOIU
    global filename1
    global filename2
    aux_file = (filename1.split('\\'))[-1]+(filename2.split('\\'))[-1]
    
    get_l_m_h(aux_file)
    
def exit_m8(): # ORBISOR
    import os
    os._exit(0)

# ORBISOR ------ 
import tkinter as tk

root = tk.Tk()
version = "MCT_MDO"
root.title(version)
#root.geometry("675x200")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

main_options=["import song1","import song2","plot song1","plot song2","plot_song_stegano"]

tk.Label(root, text="Martinescu_Danoiu_Orbisor_343A3").grid(row=0, column=0, columnspan = len(main_options), stick="n", pady=(15,0))


frame2 = tk.Frame(root)
frame2.grid(row=2, column=0)


tk.Button(frame2, text=main_options[0],command = get_name_1).grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text=main_options[1], command = get_name_2).grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='low mid high L song1', command = get_l_m_h1l).grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='low mid high R song1', command = get_l_m_h1r).grid(row=0, column=3, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")


tk.Button(frame2, text=main_options[2], command = get_plot_1).grid(row=0, column=4, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text=main_options[3], command = get_plot_2).grid(row=0, column=5, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='low mid high L song2', command = get_l_m_h2l).grid(row=0, column=6, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='low mid high R song2', command = get_l_m_h2r).grid(row=0, column=7, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")

tk.Button(frame2, text=main_options[4], command = get_plot_12).grid(row=0, column=8, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='low mid high both songs', command = get_l_m_h12).grid(row=0, column=9, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")
tk.Button(frame2, text='EXIT', command = exit_m8).grid(row=0, column=10, ipadx=5, ipady=5, padx=5, pady=(30, 5), stick="ew")

root.mainloop()




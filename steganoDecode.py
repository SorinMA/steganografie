import argparse
import numpy, math
from numpy import fft


import wave, struct, math
from scipy.io import wavfile
import numpy as np


SAMPLE_RATE = 44100 # Hz
NYQUIST_RATE = SAMPLE_RATE / 2.0
FFT_LENGTH = 512

def lowpass_coefs(cutoff):
        cutoff /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))

        # create FFT filter mask
        mask = []
        negatives = []
        l = FFT_LENGTH // 2
        for f in range(0, l+1):
                rampdown = 1.0
                if f > cutoff:
                        rampdown = 0
                mask.append(rampdown)
                if f > 0 and f < l:
                        negatives.append(rampdown)

        negatives.reverse()
        mask = mask + negatives

        # Convert FFT filter mask to FIR coefficients
        impulse_response = fft.ifft(mask).real.tolist()

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
    

def lowpass(original, cutoff):
        coefs = lowpass_coefs(cutoff)
        return numpy.convolve(original, coefs)



def stegoMod1(inpt1, output):

    song = wave.open(inpt1, mode='rb')
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))
    extracted = [frame_bytes[i] & 1 for i in range(len(frame_bytes))]
    string = "".join(chr(int("".join(map(str,extracted[i:i+8])),2)) for i in range(0,len(extracted),8))
    decoded = string.split("###")[0]
    with open(output, "w") as text_file:
        text_file.write(decoded)
    song.close()

def stegoMod2(inpt1,output):
    #-extract
    modulated = wave.open(inpt1, "r")

    demod_amsc_ok = wave.open(output, "w")
    for f in [demod_amsc_ok]:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)

    for n in range(0, modulated.getnframes()):
        signal = struct.unpack('h', modulated.readframes(1))[0] / 32768.0
        carrier = math.cos(22050.0 * (n / 44100.0) * math.pi * 2)
        base = signal * carrier
        demod_amsc_ok.writeframes(struct.pack('h', int(base* 32767)))

def stegoMod2_2(output):
    rate1, audio1 = wavfile.read(output)
    audio1 = 8*lowpass(1/2*audio1,4500)
    audiox = audio1.astype(np.int16)

    wavfile.write(output,rate1,audiox)

def stegoMod3(inpt1, output):

    rate1, audio1 = wavfile.read(inpt1)
    left = audio1[...,0].copy()
    right = audio1[...,1].copy()

    a1 = right[0] 
    a2 = right[1] 
    a = a1*1000+a2

    frame =  np.array([])
    add = int((len(left) / a - 1))
    contor = 4
    index = 0
    narray = []
    while index < a:
        aux = 0
        aux += int(np.abs(right[contor])%10)  
        aux = aux * 10 + int(np.abs(left[contor+1])%10)  
        aux = aux * 10 + int(np.abs(right[contor+2])%10)
        aux = aux * 10 + int(np.abs(left[contor+3])%10)  
        aux = aux * 10 + int(np.abs(right[contor+4])%10)
        signA = int(np.abs(left[contor + 5])%10)
        aux = aux * (signA - 1)
        narray.append(aux)
        contor += add
        index = index + 1
    frame = np.append(frame, narray)

    wavfile.write(output,rate1,(frame.T).astype(np.int16))

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT1", help="Name of the wave file")
    parser.add_argument("-t", "--text", help="Input text.", action='store_true')
    parser.add_argument("-s", "--sound", help="Input sound for AM modulation.", action='store_true')
    parser.add_argument("-c", "--sCrypto", help="Input sound for cypto-stegano.", action='store_true')
    parser.add_argument("-o", "--output", help="Name of the output wav file. Default value: out.wav).")
    args = parser.parse_args()

    text = True
    sound = False
    sCrypto = False
    output  = "out.wav"
    operation = "Text"
    contor = 0
    if args.text:
        text = args.text
        contor += 1
    if args.sound:
        sound = args.sound
        operation = "Sound"
        contor += 1
        text = False
    if args.sCrypto:
        sCrypto = args.sCrypto
        operation = "Crypto"
        contor += 1
        text = False
    if args.output:
        output = args.output
   
    print('Input file1: %s' % args.INPUT1)
 
    print('Operation: %s' % operation)
    print('Output: %s' % output)
    if contor > 1:
        print('Error, more the just 1 operation selected!')
        text = sound = sCrypto = False
    return (args.INPUT1,output, text, sound, sCrypto)


def mod1(inpt1, output):
    stegoMod1(inpt1, output)
def mod2(inpt1, output):
    stegoMod2(inpt1, output)
    stegoMod2_2(output)
def mod3(inpt1, output):
    stegoMod3(inpt1, output)

if __name__ == '__main__':
    inpt = parser()
    if inpt[2] == True:
        mod1(inpt[0], inpt[1])
    if inpt[3] == True:
        mod2(inpt[0], inpt[1])
    if inpt[4] == True:
        mod3(inpt[0], inpt[1])

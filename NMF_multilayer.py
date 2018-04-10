#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
from utils.audio.audio_utils import *
import matplotlib.pyplot as plt


def create_mel_spectrogram(wav_file):
    #  wave, samplerate = read_wav(wav_file, resample=8000)
    wave, samplerate = read_wav(wav_file)
    melspec = mel_spectrogram(wave, samplerate, n_fft=1024)
    return normalize(np.abs(melspec))


def generate_random_2D_matrix(n, m):
    """Generate random matrix (n x m). 

    :n:         n dimension (vertical)
    :m:         m dimension (horizontal)
    :returns:   random 2D matrix
    """
    return np.random.random([n, m]) 


def left_shifted(matrix, t):
    aux = np.zeros(matrix.shape)

    if t == 0:
        aux = matrix 
    else:
        aux[:,0:-t] = matrix[:,t:]
    
    return aux


#@print_matrix_shapes
def calc_predicted_matrix(V, Wperc, Hperc, Wharm):
    """Calculate predicted matrix.

    :V:         input matrix
    :Wperc:     W percussive (dictionary)
    :Hperc:     H percussive 
    :Wharm:     W harmonic 
    :returns:   updated V_p 

    """
    Tau = Wperc.shape[0]

    #  print("Wharm T: ", Wharm.T)
    #  print("V:", V)
    aux = np.matmul(Wharm.T, V)
    #  print("aux: ", aux)
    harmonic_part = np.matmul(Wharm, aux)
    #  print("harm part: ", harmonic_part)

    #  V_p = np.zeros(harmonic_part.shape)
    V_p = np.zeros(V.shape)

    for t in range(Tau):
        V_p += np.matmul(Wperc[t], left_shifted(Hperc, t))

    V_p += harmonic_part 
     
    print("V predicted: ", V_p)
    #  print("V: ", V)
    return V_p


#@print_matrix_shapes
def update_Wharm_Hperc(V, Wperc, Hperc, Wharm):
    """Update W harmonic and H percussive.

    :V:         input matrix
    :Wperc:     W percussive (dictionary)
    :Hperc:     H percussive
    :Wharm:     W harmonic
    :returns:   updated tuple of Wharm, Hperc

    """
    def _calc_Z(V, V_p_inv):
        """Calculate Z."""
        return V * V_p_inv 

    def _calc_phi(V_p_inv):
        """Calculate phi."""
        I = np.ones(V_p_inv.shape)
        return I * V_p_inv

    Tau = Wperc.shape[0]

    V_p = calc_predicted_matrix(V, Wperc, Hperc, Wharm)
    V_p_inv = np.power(V_p, -1)

    Z = _calc_Z(V, V_p_inv)
    phi = _calc_phi(V_p_inv)

    lhs = np.matmul(V.T, Wharm)    
    lhs = np.matmul(Z, lhs)

    rhs = np.matmul(Z.T, Wharm)
    rhs = np.matmul(V, rhs)
    
    W_numerator = lhs + rhs

    lhs = np.matmul(V.T, Wharm)
    lhs = np.matmul(phi, lhs)

    rhs = np.matmul(phi.T, Wharm)
    rhs = np.matmul(V, rhs)

    W_denominator = lhs + rhs

    ####---------------------------------------####

    H_numerator_sum = np.zeros(Hperc.shape)
    H_denominator_sum = np.zeros(Hperc.shape)
    p = Hperc.shape[1]
    n = Wperc[0].shape[0]
    #  print("p y n: ", p, n)
    M_of_ones = np.ones((n, p)) # has to match transposed Wperc
    #  print("M of ones:", M_of_ones.shape)
    
    for t in range(Tau):
        aux = left_shifted((V * V_p_inv), t)
        H_numerator_sum += np.matmul(Wperc[t].T, aux)

        H_denominator_sum += np.matmul(Wperc[t].T, left_shifted(M_of_ones, t))

    Wharm = Wharm * (W_numerator / W_denominator)
    #  print("Wharm: ", Wharm)
    Hperc = Hperc * (H_numerator_sum / H_denominator_sum)
    #  print("Hperc: ", Hperc)
    return Wharm, Hperc


def get_files(directory):
    """
    Create dictionary of files in audio directory with file name as key and file
    path as value.
    """ 
    dir_ls = listdir(directory)
    keys = [ name.split('.')[0] for name in dir_ls ]
    files = list(map(lambda f: './audio/' + f, dir_ls))
    return  {k: v for (k, v) in  zip(keys, files)}


def generate_sample_dict(samples):
    """
    Generate drum dictionary of samples.

    :samples:  list of spectrograms for individual drum samples
    :returns:  drum dictionary Wperc, 3D matrix (t x n x m)

    """
    m_length = len(samples)         # defined by number of samples
    n_length = samples[0].shape[0]  # defined by height of any sample spectrogram
    t_length = samples[0].shape[1]  # defined by width of any sample spectrogram
    target = np.zeros((t_length, n_length, m_length))

    for index, sample in enumerate(samples): 
        for row in range(t_length):
            for col in range(n_length):
                target[row][col][index] = sample[col][row]

    return target


def plot_matrix(m):
    plt.matshow(m)
    plt.show()


###############################################################################

d = get_files('./audio')                # create file dict

# generate spectrogram V for all samples combined
# order: bd - hh - sd
V = create_mel_spectrogram(d['all'])    # create spectrogram for all sounds

# Create spectrograms for all individual samples and add them to sample list
sample_list = []
sd_spec = create_mel_spectrogram(d['sd'])
sample_list.append(sd_spec)
bd_spec = create_mel_spectrogram(d['bd'])
sample_list.append(sd_spec)
hh_spec = create_mel_spectrogram(d['hh'])
sample_list.append(sd_spec)

# Generate Wperc (sample dictionary)
Wperc = generate_sample_dict([bd_spec, hh_spec, sd_spec])

# Generate inicial random matrices Hperc, Wharm
n, p = V.shape
Hperc = generate_random_2D_matrix(len(sample_list), p)  # create H matrix
#  print(Hperc)
Wharm = generate_random_2D_matrix(n, len(sample_list))  # create W matrix
#  print(Wharm)

for i in range(100):
    Wharm, Hperc = update_Wharm_Hperc(V, Wperc, Hperc, Wharm)

#  plot Wharm x Hperc = V_p
plot_matrix(np.matmul(Wharm, Hperc))
#  plot original #V
plot_matrix(V)

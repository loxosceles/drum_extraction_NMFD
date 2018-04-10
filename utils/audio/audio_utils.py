#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import librosa
import soundfile as sf

from scipy.signal import correlate
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing

from .gammatogram import fft_weights


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_samplerate_from_file(path):
    return sf.SoundFile(path).samplerate


def get_duration_from_file(path):
    """ Regresa la duración de un archivo, sin leer el wav entero.

    Args:
        path (str): direacción absoluta del archivo a leer.

    Regresa:
        dur (int): duración del archivo de audio.
    """
    return librosa.core.get_duration(filename=path)


def read_wav(path, resample=False, channels=1):
    """ Lectura de archivo wav.

    Args :
        path (str): dirección absoluta del archivo a leer.
        resample (int; opcional) : frecuencia de muestreo a la cual
            remuestrear el archivo.
        channels (int; opcional) : número de canales deseados. Si
            el número de canales del archivo es mayor al indicado,
            se mezclan los canales. Si el número de canales indicado
            es mayor al del archivo, se leen los canales del archivo.

    Regresa:
        wav (np.float32): el wav leído y normalizado (rango =[-1,1])
            en un arreglo de numpy, de tamaño [channels,-1] si
            channels > 1, o bien de tamaño [-1].
        fr (int): la frecuencia de muestreo
    """
    if channels == 1:
        mono = True
    else:
        mono = False

    if resample:
        return librosa.core.load(path, sr=resample, mono=mono)
    else:
        return librosa.core.load(path, sr=None, mono=mono)


def resample(wav, freq_vieja, freq_nueva):
    """ Remuestreo de un arreglo 1-dimensional.

    Args:
        wav (np.array): Un arreglo de numpy de forma (2,-1) o
            bien (-1) en caso de mono.
        freq_vieja (int) : frecuencia de muestreo de wav.
        freq_nueva (int) : frecuencia de muestreo del nuevo wav.

    Regresa:
        wav_resamp (np.array) : Wav remuestreado a la nueva frecuencia
            dada en freq_nueva. Tiene la misma forma que wav.
    """
    return librosa.core.resample(wav, freq_vieja, freq_nueva)


def write_wav(wav, path, freq, verb=False, **kwargs):
    """ Escribe wavs a archivo.

    Args:
        wav (np.array): Arreglo de numpy forma (2,-1) o (-1).
        path (str): dirección donde guardar el archivo.
        freq (str): frecuencia de muestreo.
        verb (bool; opcional): si está presente imprime mensaje
            de guardado.

        **kwargs: vease la documentacion de pysoundfile.write
            https://pysoundfile.readthedocs.io/en/0.9.0/#soundfile.write
    Regresa:
        None

    """
    sf.write(path, wav, freq, **kwargs)
    if verb:
        print("Se escribio exitosamente el archivo %s" % path)


def pitch_shift(wav, porciento):
    """ Hace un shift de tono dado por un porcentaje.

    Args:
        wav (np.array): El audio a procesar.
        porciento (float): Si porciento=.1 entonces
            el tono del audio se aumenta en un 10%.
            Si porciento=-.1, el tono del audio
            se disminuye en un 10%.
    Regresa:
        shifted (np.array): Audio con el shift en tono.
    """
    porciento = min(.5, porciento)
    return librosa.core.resample(wav, 1, (1 - porciento))


def random_pitch_shift(wav):
    per = np.random.normal(scale=.1)
    return pitch_shift(wav, per)


def temp_shift(wav, punto_corte):
    """ Función que cícla un arreglo 1-dimensional. Sirve
        para crear versiones de audio con inicio en el punto_corte.

    Args:
        wav (np.array): audio (o arreglo de numpy 1d) a procesar.
        punto_corte (int): Punto a partir del cual se incia
            el nuevo audio.
    Regresa:
        shifted (np.array).
    """
    wav_len = len(wav)
    cut_index = wav_len - int(wav_len * punto_corte)
    return np.roll(wav, cut_index)


def random_temp_shift(wav):
    cut = np.random.uniform(0, len(wav))
    return temp_shift(wav, cut)


def autocorrelacion_local(
        espectro,
        radio=1,
        norm=True,
        mask=True,
        thresh=1e-5):
    """ Funcion que limpia los espectrogramas.

        Puntualmente se toma una ventana W[i,j] de determinado radio
        alrededor del punto, i.e W[i,j] = espetro[i-radio:i+radio, j-radio:j+radio]
        y se calcula el producto punto de la ventana con la traslacion
        de la ventana a puntos cercanos. Finalmente se toma un promedio.

        R[i,j] = Promedio( W[i,j] * W[i+x, j+y] | i,j in [-radio,radio])

        No es autocorrelacion, (en todo caso sería más cercano a
        autovarianza local), pero es un modo de probar la coherencia
        espacial del espectro.

        Args:
            espectro (np.array 2d): Arreglo de numpy 2-dimensional.
            radio (int; opcional): Tamaño de vencindad con la que
                comparar.
            norm (bool; opcional): Si norm=True, se normaliza
                el resultado para que de valores en el rango [0,1]
            mask (bool; opcional): Si mask=True, después de calcular
                la autocorrelacion_local se aplica un límite a cada
                pixel. Regresa una matriz de booleanos.
            thresh (float; opcional): El límite a usar por mask.

        Regresa:
            autocorr_loc (np.array 2d): Arreglo de numpy 2-dimensional de la misma
                forma que espectro.
    """

    def transladar(matriz, direccion):
        """ Función que translada una matriz hasta que el punto
            de origen (i.e. matriz[0,0]) sea el vector dado en
            dirección.
        """
        shifts = (direccion % matriz.shape)
        res = matriz
        for axis, shift in enumerate(shifts):
            res = np.roll(res, shift, axis=axis)
        return res

    def correlaciones_locales(matriz_1, matriz_2, radio):
        """ Función que calcula las "correlaciones locales"
        entre dos matrices. Se asume que las matrices tienen
        la misma forma. Si

            W[i,j,k] = matriz_k[i-radio:i+radio+1,j-radio,j+radio+1]

        Entonces se regresa la matriz R con

            R[i,j] = W[i,j,1] * W[i,j,2]

        donde * denota producto punto.
        """
        nucleo = np.ones([2 * radio + 1, 2 * radio + 1])
        return correlate(matriz_1 * matriz_2, nucleo, 'same')

    # Se añaden 0 a los lados para que al trasladar la matriz original
    # no se cíclen las entradas no nulas.
    espectro_pad = np.pad(
        espectro,
        [(radio, radio), (radio, radio)],
        'constant')

    # Puntos con los que se transladara la matriz original
    centro = np.array([radio, radio])
    translados = [
        punto - centro
        for punto in np.ndindex(2 * radio + 1, 2 * radio + 1)
        if (punto != centro).any()
    ]

    # Se calculan las correlaciones_locales con las matrices transladadas
    autocorrelaciones = np.stack([
        correlaciones_locales(
            espectro_pad,
            transladar(espectro_pad, direccion), radio)
        for direccion in translados])
    autocorrelaciones = np.sum(autocorrelaciones, axis=0)

    # Se recorta al tamaño original
    autocorrelaciones = autocorrelaciones[radio:-radio, radio:-radio]

    if norm:
        autocorrelaciones /= autocorrelaciones.max()

    if mask:
        autocorrelaciones *= (autocorrelaciones >= thresh)

    return autocorrelaciones


def blend_wav_files(wavs, pitch_shift=True):
    """ Blend aleatorio de audios.

        Args:
            wavs (lista [np.array 1d]): Lista de arrays
                de numpy 1-dimensionales que se mezclaran.
            pitch_shift (bool; opcional): Si True, a algunos
                de los archivos de la lista wav se les aplica un
                pitch_shift con porcentaje aleatorio. Los porcentajes
                están distribuidos N(0,0.1)

        Regresa:
            blend: array de numpy 1-d. Al array más largo
                de la lista wavs se le añaden los otros
                arrays en posiciones aleatorias.
    """
    if pitch_shift:
        shift = [np.random.binomial(1, 0.1) for file in wavs]
        shift_percent = [np.random.normal(0, 0.1) for file in wavs]

        for index, wav in enumerate(wavs):
            if shift[index]:
                wav = pitch_shift(wav, shift_percent[index])

    # Se elige el wav mas largo
    lengths = [len(x) for x in wavs]
    max_wav = np.argmax(lengths)
    max_len = lengths[max_wav]

    # Se elige un punto de inicio para los demás wavs
    starting_point = [np.random.random_integers(0, max_len - length)
                      for length in lengths]
    mixed_wav = np.zeros_like(wavs[max_wav])

    # Se mezclan
    for index, wav in enumerate(wavs):
        sp = starting_point[index]
        mixed_wav[sp:sp + lengths[index]] += wav

    return mixed_wav


def _removekey(d, key):
    r = dict(d)
    if r.get(key, False):
        del r[key]
    return r


def stft(wav, n_fft=2**10, **kwargs):
    """ Un wrapper de sftf de librosa.core.sftf
        Vease la documentación de librosa:
            https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa.core.stft

        Regresa:
            sftf(np.array(np.complex64))
    """
    kwargs = _removekey(kwargs, 'channels')
    return librosa.core.stft(wav, n_fft=n_fft, **kwargs)


def istft(spec, **kwargs):
    """ Un wrapper de la funcion inversa de sftf de librosa.core.isftf
        Vease la documentación de librosa:
            https://librosa.github.io/librosa/generated/librosa.core.istft.html#librosa.core.istft
        Regresa:
            wav(np.array())
    """
    kwargs = _removekey(kwargs, 'channels')
    return librosa.core.istft(spec, **kwargs)


def db_spectrogram(wav, n_fft=2**10, **kwargs):
    """ Funcion que calcula los decibeles de cada bin de un espectrograma
        de wav.

        Args:
            wav (np.array 1d): Array de numpy 1 dimensional.
            n_fft (int; opcional): tamaño de ventana con la cual
                hacer la fft.
            **kwargs: ver documentacion de librosa.core.sftf

        Regresa:
            powerspec (np.array(np.float32) 2d)
    """
    kwargs = _removekey(kwargs, 'channels')
    spec = librosa.core.stft(wav, n_fft=n_fft, **kwargs)
    return librosa.amplitude_to_db(np.abs(spec))


def mel_spectrogram(wav, sr, db=True, n_fft=2**10, **kwargs):
    """ Funcion que calcula el espectrograma usando la escala mel.

        Args:
            wav (np.array 1d): Array de numpy 1 dimensional.
            sr (int): frecuencia de muestreo de wav.
            db (bool:opcional): Si es verdadero regresa los decibeles
                de cada bin de frecuencia.
            n_fft (int; opcional): tamaño de ventana con la cual
                hacer la fft.
            **kwargs: ver documentacion de librosa.core.sftf

        Regresa:
            powerspec (np.array(np.float32) 2d)
    """
    M = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, **kwargs)
    if db:
        return librosa.power_to_db(M, ref=np.max)
    else:
        return M


def median_clip_clef(s, threshx=3, threshy=3):
    """ Funcion auxiliar para la construcción de las máscaras utilizadas
        por la entrada ganadora de birdclef 2016
            http://ceur-ws.org/Vol-1609/16090547.pdf

        Se regresa una matriz booleana M tal que M[i,j] = 1 si
            s[i,j] >= threshx * media(s[i,:])
                y
            s[i,j] >= threshy * media(s[:,j])

        Args:
            s (np.array 2d): Una matriz dos dimensional de numpy.
            threshx (float; opcional): Niveles de corte de los renglones.
            threshy (float; opcional): Niveles de corte de las columnas.

        Regresa:
            mask (np.array (bool) 2d): Una matriz booleana de los pixeles
                en los que se satisfacen las condiciones.
    """
    median1 = np.median(s, axis=1)
    median2 = np.median(s, axis=0)
    a = (s >= (threshy * median1)[:, None])
    b = (s >= (threshx * median2)[None, :])
    return np.logical_and(a, b)


def average_clip_clef(s, threshx=3, threshy=3):
    """ Funcion auxiliar para la construcción de las máscaras utilizadas
        por la entrada ganadora de birdclef 2016
            http://ceur-ws.org/Vol-1609/16090547.pdf

        Se regresa una matriz booleana M tal que M[i,j] = 1 si
            s[i,j] >= threshx * promedio(s[i,:])
                y
            s[i,j] >= threshy * promedio(s[:,j])

        Args:
            s (np.array 2d): Una matriz dos dimensional de numpy.
            threshx (float; opcional): Niveles de corte de los renglones.
            threshy (float; opcional): Niveles de corte de las columnas.

        Regresa:
            mask (np.array (bool) 2d): Una matriz booleana de los pixeles
                en los que se satisfacen las condiciones.
    """
    median1 = np.average(s, axis=1)
    median2 = np.average(s, axis=0)
    a = (s >= (threshy * median1)[:, None])
    b = (s >= (threshx * median2)[None, :])
    return np.logical_and(a, b)


def wav_select(wav, mask, hop_length=512):
    """ Funcion que selecciona los pedazos de wav indicados por mask.
        mask es un arreglo booleano de longitud len(wav)/hop_length

        Args:
            wav (np.array 1d): Arreglo numpy 1 dimensional.
            mask (np.array(bool)): Arreglo de longitud len(wav)/hop_length
            hop_length (int): La tasa temporal entre wav y mask
                (por ejemplo cuando mask proviene de ciertas condiciones
                en un espectrograma, hop_length debe de ser el tamaño de
                salto usado para generar el espectrograma).

        Regresa:
            select_wav (np.array 1d): Arreglo con el mismo dtype que
                wav.
    """

    new_len = mask.sum()
    new_wav = np.zeros(new_len * hop_length, dtype=wav.dtype)

    count = 0
    for i, frame in enumerate(mask[:-2]):
        if frame:
            new_wav[count * hop_length:(count + 1) * hop_length] \
                = wav[i * hop_length:(i + 1) * hop_length]
            count += 1
    return new_wav


def extract_mask_clef(
        spec,
        ero=[4, 4],
        clip_type='median',
        thresh=3):
    """ Funcion que extrae una mascara binaria de un espectrograma
        (aunque funciona para cualquier matriz de numpy) usada por
        la entrega ganadora de birdclef 2016.
            http://ceur-ws.org/Vol-1609/16090547.pdf

        Después de hacer median(average)_clip, se hacen varias operaciones
        morfológicas sobre la máscara resultante. Vease:
            http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

        Args:
            spec (np.array 2d): matriz de la cual se extrae la mascara.
            ero (list [int,int]): estructura binaria con la cual calcular
                las operaciones de erosion y dilatación.
            clip_type (str): modo usado para seleccionar pixeles: 'median'
                o 'average'. Llama las funciones median_clip_clef o
                average_clip_clef
            thresh (int) : Lo niveles límite usados en las funciones de
                clip.

        Regresa:
            mask (np.array(bool) 2d)

    """
    if clip_type == 'median':
        mask = median_clip_clef(spec, threshx=thresh, threshy=thresh)
    else:
        mask = average_clip_clef(spec, threshx=thresh, threshy=thresh)

    mask = binary_closing(mask)
    binary_struc = np.ones(ero)
    mask = binary_erosion(mask, structure=binary_struc)
    mask = binary_dilation(mask, structure=binary_struc)

    return mask


def extract_from_wav(
        wav,
        mode,
        n_fft=2**10,
        hop_length=2**9,
        ero=[4, 4],
        clip_type='median',
        thresh=3,
        close_size=7,
        dil_size=3,
        **kwargs):
    """ Funcion que extrae ruido o lo opuesto de un arreglo numpy 1d.

        La función hace lo siguiente:

        wav --(sftf)--> spec --(extract_mask_clef)--> mask

        Luego hace una proyección de mask al eje temporal seguido
        de operaciones morfológicas para cerrar huecos y padear
        los cantos.

        Args:
            wav (np.array 1d): El numpy array del que se extrae.
            mode (str): Un string con el modo de extracción. Si
                mode = 'noise', se extrae el ruido. Si mode = 'song'
                se extrae el canto.
            n_fft, hop_lenght (int): vease la función de sftf.
            ero, clip_type, thresh: vease la funcion extract_mask_clef.
            close_size (int): tamaño de ventana para la cerradura
                temporal.
            dil_size (int): tamaño de dilatacion final.
            **kwargs: vease la funcion sftf.

        Regresa:
            canto_wav (np.array 1d): Array de numpy que contiene
                la parte de canto del wav.
            ruido_wav (np.array 1d): Array de numpy que contiene
                la parte del ruido.
            temp_mask (np.array(bool)): Array booleano de numpy
                de que contiene información de que bins temporales
                del espectrogramas contienen canto (True) o ruido.
    """
    assert mode in ['ruido', 'canto', 'ambas'], \
        "El modo usado no es una de las \
        opciones posibles: ['ruido', 'canto','ambas']."

    spec = np.abs(stft(wav, n_fft=n_fft, hop_length=hop_length, **kwargs))

    mask = extract_mask_clef(spec, ero=ero, clip_type=clip_type, thresh=thresh)

    temp_mask = np.any(mask, axis=0)
    temp_mask = binary_closing(temp_mask, structure=np.ones(close_size))
    temp_mask = binary_dilation(temp_mask, structure=np.ones(dil_size))

    if mode == 'ruido':
        ruido_mask = np.logical_not(temp_mask)
        ruido_wav = wav_select(wav, ruido_mask, hop_length=hop_length)
        return ruido_wav, temp_mask
    elif mode == 'canto':
        canto_wav = wav_select(wav, temp_mask, hop_length=hop_length)
        return canto_wav, temp_mask
    else:
        ruido_mask = np.logical_not(temp_mask)
        canto_wav = wav_select(wav, temp_mask, hop_length=hop_length)
        ruido_wav = wav_select(wav, ruido_mask, hop_length=hop_length)
        return canto_wav, ruido_wav, temp_mask


def gammatogram(wav, freq, n_fft, hop_length, channels=100, f_min=100):

    gt_weights = fft_weights(
        n_fft,
        freq,
        channels,
        f_min,
        int(freq / 2),
        int(n_fft / 2) + 1)

    sgram = stft(wav, n_fft=n_fft, hop_length=hop_length)

    result = gt_weights.dot(np.abs(sgram)) / n_fft
    return result


def reduce_horizontal_bands(spec, normed=True, factor=2):
    ep = 1e-15

    spec_ = normalize(spec)
    spec_row_normalized = spec_ / np.sum(spec_, axis=1)[:, None]
    entropy = -spec_row_normalized * np.log(spec_row_normalized + ep)
    entropy_per_row = normalize(np.sum(entropy, axis=1))

    H = factor * entropy_per_row + 1
    result = spec / H[:, None]

    if normed:
        result = normalize(result)
    return result


def reduce_vertical_bands(spec, normed=True, factor=3):
    ep = 1e-15

    spec_ = normalize(spec)
    spec_col_normalized = spec_ / np.sum(spec_, axis=0)[None, :]
    entropy = -spec_col_normalized * np.log(spec_col_normalized + ep)
    entropy_per_col = normalize(np.sum(entropy, axis=0))
    print(entropy_per_col.min(), entropy_per_col.max())

    H = factor * entropy_per_col * (entropy_per_col > .6) + 1
    print(H.max(), H.min())
    result = spec / H[None, :]

    if normed:
        result = normalize(result)
    return result


def rmse(wav, **kwargs):
    return librosa.feature.rmse(wav, **kwargs)[0, :]

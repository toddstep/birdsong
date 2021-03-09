import io

import boto3
import librosa
import numpy as np


def load_audio(bucket, file_prefix, offset=0., duration=None):
    """
    Return audio and sampling rate in S3 object
    
    https://stackoverflow.com/questions/44043036/how-to-read-image-file-from-s3-bucket-directly-into-memory
    https://librosa.org/doc/latest/ioformats.html#read-file-like-objects
    
    Keyword arguments:
    file_prefix -- prefix for audio object to load
    bucket -- S3 bucket containing the object (default 'toddstep')
    offset -- time within object (in seconds) to start load (default 0.)
    duration -- time length (in seconds) for retrieved audio (default None)
    """
    
    s3 = boto3.resource('s3')
    key = '{}'.format(file_prefix)
    io_stream = io.BytesIO()
    s3.Bucket(bucket).Object(key).download_fileobj(io_stream)
    io_stream.seek(0)
    wav, sr = librosa.load(io_stream, offset=offset, duration=duration, sr=None)
    return wav, sr

def compute_spec(wav, sr, n_mels=128, fmin=200, fmax=8000):
    """
    Compute spectrogram from audio.
    It does not normalize the spectra.
    
    Keyword arguments:
    wav -- audio
    sr -- sampling rate
    n_mels -- number of mel filter banks (default 128)
    fmin -- minimum frequency in spectrogram (default 200)
    fmax -- maximum frequency in spectrogram (default 8000)
    """

    n_fft = int(.2*sr)
    hop_length = int(.5*n_fft)
    spec = librosa.feature.melspectrogram(wav, n_fft=n_fft, hop_length=hop_length,
                                                 n_mels=n_mels, fmin=fmin, fmax=fmax).T
    spec = np.log(spec)
    return spec, hop_length

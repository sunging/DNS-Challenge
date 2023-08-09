import argparse
from typing import Union
import librosa
import soundfile as sf
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class Dnsmos:
    """
    A class representing the DNS MOS model.

    Attributes:
        onnx_sess (ort.InferenceSession): The ONNX inference session for the primary model.
        p808_onnx_sess (ort.InferenceSession): The ONNX inference session for the P.808 model.
        personalized_MOS (bool): Whether to use the personalized MOS model.
    """

    def __init__(self, model_dir:str = os.path.dirname(os.path.abspath(__file__)), personalized_MOS: bool = False, num_threads: int = 0) -> None:
        """
        Initialize the DNS MOS model.

        Args:
            model_dir (str, optional): The directory containing the DNS MOS model files. Defaults to the directory of the current file.
            personalized_MOS (bool, optional): Whether to use the personalized MOS model. Defaults to False.
            num_threads (int, optional): The number of threads to use for inference. Defaults to 0 (use all available threads).
        """
        p808_model_path = os.path.join(model_dir, 'DNSMOS', 'model_v8.onnx')

        if personalized_MOS:
            primary_model_path = os.path.join(model_dir, 'pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = os.path.join(model_dir, 'DNSMOS', 'sig_bak_ovr.onnx')
        
        sess_options = ort.SessionOptions()
        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads

        self.onnx_sess = ort.InferenceSession(primary_model_path, sess_options=sess_options)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, sess_options=sess_options)
        self.personalized_MOS = personalized_MOS
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        """
        Compute the mel spectrogram of the given audio signal.

        Args:
            audio (np.ndarray): The audio signal as a numpy array.
            n_mels (int, optional): The number of mel bands to generate. Defaults to 120.
            frame_size (int, optional): The size of the FFT window. Defaults to 320.
            hop_length (int, optional): The number of samples to advance between frames. Defaults to 160.
            sr (int, optional): The sampling rate of the audio signal. Defaults to 16000.
            to_db (bool, optional): Whether to convert the power spectrogram to decibels. Defaults to True.

        Returns:
            np.ndarray: The mel spectrogram as a numpy array.
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        """
        Compute the polynomial fit values for the given signal, background, and overlap-add scores.

        Args:
            sig (float): The signal score.
            bak (float): The background score.
            ovr (float): The overall score.

        Returns:
            Tuple[float, float, float]: The polynomial fit values for the signal, background, and overall scores.
        """
        if self.personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def compute_score(self, input: Union[str, np.ndarray], sampling_rate: int = 16000, return_all: bool = False):
        """
        Compute the DNS MOS score for the given audio input.

        Args:
            input (Union[str, np.ndarray]): The audio input as either a file path or a numpy array.
            sampling_rate (int, optional): The sampling rate of the audio input. Defaults to 16000.
            return_all (bool, optional): Whether to return all the intermediate scores. Defaults to False.

        Returns:
            float: The DNS MOS score for the given audio input.
        """
        fs = sampling_rate
        if isinstance(input, str):
            aud, input_fs = sf.read(input)
            if aud.ndim == 2:
                aud = aud[:,0]
            if input_fs != fs:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
            else:
                audio = aud
        else:
            if input.ndim == 1:
                audio = input
            else:
                audio = input[:,0]
        
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype('float32')/np.iinfo(audio.dtype).max

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {}
        if input is str:
            clip_dict['filename'] = input
        clip_dict['len_in_sec'] = actual_audio_len/fs
        clip_dict['sr'] = fs
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)

        if return_all:
            clip_dict['all'] = {
                'OVRL_raw': predicted_mos_ovr_seg_raw,
                'SIG_raw': predicted_mos_sig_seg_raw,
                'BAK_raw': predicted_mos_bak_seg_raw,
                'OVRL': predicted_mos_ovr_seg,
                'SIG': predicted_mos_sig_seg,
                'BAK': predicted_mos_bak_seg,
                'P808_MOS': predicted_p808_mos
            }
        return clip_dict

def main():
    """Run main function of the DNS MOS script. Parses command line arguments, computes MOS scores for one or more WAV files, and optionally dumps the scores to a CSV file."""
    parser = argparse.ArgumentParser(description='Compute score for one or more WAV files')
    parser.add_argument('wav_paths', type=str, nargs='+', help='Paths to the WAV files or directory')
    parser.add_argument('-dump', action='store_true', help='Dump the scores to a CSV file')
    args = parser.parse_args()
    dump = args.dump

    dns_challenge = Dnsmos()

    rsts = []

    def run(wav_path):
        mos_score = dns_challenge.compute_score(wav_path)

        rst = {
            'filename': wav_path,
            'OVRL': mos_score['OVRL'],
            'SIG': mos_score['SIG'],
            'BAK': mos_score['BAK'],
            'P808_MOS': mos_score['P808_MOS']
        }
        rsts.append(rst)

        print(f'MOS score for {wav_path}:')
        print(f'OVRL score: {mos_score["OVRL"]:.4f}')
        print(f'SIG score: {mos_score["SIG"]:.4f}')
        print(f'BAK score: {mos_score["BAK"]:.4f}')
        print(f'P808_MOS score: {mos_score["P808_MOS"]:.4f}')
        print()

    wav_list = []
    for wav_path in args.wav_paths:
        if os.path.isdir(wav_path):
            for root, dirs, files in os.walk(wav_path):
                for file in files:
                    if file.endswith('.wav'):
                        wav_file = os.path.join(root, file)
                        wav_list.append(wav_file)
        else:
            wav_list.append(wav_path)
    
    for wav_path in tqdm(wav_list):
        run(wav_path)
    
    if dump:
        df = pd.DataFrame(rsts)
        df.to_excel('mos_scores.xlsx', index=False)
            
if __name__ == '__main__':
    main()
    
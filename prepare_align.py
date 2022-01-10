import argparse
import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import yaml
from text.symbols import symbols, my_symbols



# BZNSYP数据集处理
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    speaker = "BZNSYP"

    print('sampling_rate',sampling_rate)

    for root,dir,files in os.walk(os.path.join(in_dir,'PhoneLabeling')):
        for i, file in tqdm(enumerate(files)):
            if file.endswith('.interval'):
                base_name = file.split('.')[0]
                with open(os.path.join(in_dir, 'PhoneLabeling', file), encoding="utf-8") as f:
                    lines = f.readlines()
                    phones = []
                    for line in lines[11:]:
                        if line.strip()[0]=="\"":
                            p = line.strip().replace("\"","").replace('sp1','sp')
                            if p not in my_symbols:
                                print(base_name, p)
                            phones.append(p)
                    text = ' '.join(phones)


# aishell3数据集处理
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for dataset in ["train", "test"]:
        print("Processing {}ing set...".format(dataset))
        with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
            for line in tqdm(f):
                wav_name, text = line.strip("\n").split("\t")
                speaker = wav_name[:7]
                text = text.split(" ")[1::2]
                wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, wav_name),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),
                        "w",
                    ) as f1:
                        f1.write(" ".join(text))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    prepare_align(config)

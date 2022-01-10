import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_utils.get_model import get_model, get_vocoder
from my_utils.tools import to_device, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate
from scipy.io import wavfile


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    preprocess_config, model_config, train_config = configs
    step = args.restore_step + 1
    batch_size = 2
    epoch = 1
    grad_acc_step = 1
    grad_clip_thresh = 1.0
    total_step = 1  #900000
    log_step = 100
    save_step = 10000
    synth_step = 10000
    val_step = 1000

    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    vocoder = get_vocoder(model_config, device)
    Loss = FastSpeech2Loss(preprocess_config).to(device)


    while True:
        print('####### epoch {} start! ##########'.format(epoch))
        for batchs in tqdm(loader):
            for batch in batchs:
                # batch: (
                #     ids,
                #     raw_texts,
                #     speakers,
                #     texts,   # b, max_len
                #     text_lens,  # b
                #     max(text_lens),

                #     mels,  2, mel_max_len, mel_bin
                #     mel_lens,   # b
                #     max(mel_lens),
                #     pitches,  # b, max_len
                #     energies,  # b, max_len
                #     durations,  # b, max_len
                # )

                batch = to_device(batch, device)
                output = model(*(batch[2:]),device=device)
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "total loss: {:.4f}, " \
                               "mel loss: {:.4f}, " \
                               "mel posnet loss: {:.4f}, " \
                               "pitch loss: {:.4f}, " \
                               "energy loss Loss: {:.4f}, " \
                               "duration loss: {:.4f}"\
                        .format(*losses)
                    print('\n' + message1 + message2)

                if step % val_step == 0:
                    model.eval()
                    print('###################### evaluate ##################')
                    message = evaluate(model, step, configs, vocoder)
                    print('\n', message)
                    model.train()

                if step % save_step == 0:
                    torch.save({
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                                }, os.path.join('save_models', "{}.pth.tar".format(step)))

                if step == total_step:
                    quit()
                step += 1
        epoch += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    args = parser.parse_args()
    args.preprocess_config = "config/BZNSYP/preprocess.yaml"
    args.model_config = "config/BZNSYP/model.yaml"
    args.train_config = "config/BZNSYP/train.yaml"

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)

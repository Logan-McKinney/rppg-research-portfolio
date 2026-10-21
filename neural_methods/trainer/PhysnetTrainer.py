import logging
import os
from collections import OrderedDict
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from natsort import natsorted
import scipy
from scipy.sparse import spdiags
from scipy.signal import butter
import math
from scipy import linalg
from scipy import signal
from scipy import sparse
from scipy.signal import find_peaks
import pywt
from scipy.stats import pearsonr
import torch.nn as nn


def process_signal(signal, wavelet='db4', level=2):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Adaptive thresholding for noise reduction
    def adaptive_threshold(coeff, level):
        sigma = np.median(np.abs(coeff)) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(coeff)))
        return uthresh

    for i in range(1, len(coeffs)):
        magnitude = np.abs(coeffs[i])
        uthresh = adaptive_threshold(coeffs[i], i)
        coeffs[i] = pywt.threshold(coeffs[i], value=uthresh, mode='soft')
        coeffs[i][magnitude < 1e-10] = 0  # Set coefficients with very small magnitude to zero

    # Reconstruct the signal
    filtered_signal = pywt.waverec(coeffs, wavelet)

    # Ensure the length of the filtered signal is the same as the input signal
    if len(filtered_signal) > len(signal):
        filtered_signal = filtered_signal[:len(signal)]
    elif len(filtered_signal) < len(signal):
        filtered_signal = np.pad(filtered_signal, (0, len(signal) - len(filtered_signal)), 'constant')

    return filtered_signal


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
            if config.TRAIN.LOAD_MODEL_PATH:
                self.load_model(config.TRAIN.LOAD_MODEL_PATH)
        elif config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
            self.model.eval()
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def load_model(self, load_model_path):
        print(f"Loading model from: {load_model_path}")
        self.model.load_state_dict(torch.load(load_model_path))

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)

                # Debug print statements
                print(f"Original labels shape: {labels.shape}")

                # Assuming labels is a 3D tensor with shape (N, D, L)
                if labels.dim() == 3:
                    labels = labels.view(N * D, -1)
                else:
                    raise ValueError(f"Unexpected labels shape: {labels.shape}")

                print(f"Reshaped labels shape: {labels.shape}")

                # Process labels
                processed_labels = torch.tensor(np.array([process_signal(label.cpu().numpy()) for label in labels]),
                                                dtype=torch.float32).to(labels.device).view(N * D, -1)

                self.optimizer.zero_grad()
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(data)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                processed_labels = (processed_labels - torch.mean(processed_labels)) / torch.std(
                    processed_labels)  # normalize
                loss = self.loss_model(rPPG, processed_labels)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(N * D, -1)

                # Process labels
                processed_labels_valid = torch.tensor(
                    np.array([process_signal(label.cpu().numpy()) for label in labels_valid]), dtype=torch.float32).to(
                    labels_valid.device).view(N * D, -1)

                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(data_valid)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                processed_labels_valid = (processed_labels_valid - torch.mean(processed_labels_valid)) / torch.std(
                    processed_labels_valid)  # normalize
                loss = self.loss_model(rPPG, processed_labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            current_dir = os.path.basename(self.model_dir)
            hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()
            full_path = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name)

            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    full_path, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    full_path, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)

                # Process labels
                processed_labels_test = torch.tensor(
                    np.array([process_signal(label.cpu().numpy()) for label in labels_test]), dtype=torch.float32).to(
                    labels_test.device).view(-1, 1)

                pred_ppg_test, pred_peaks_test, pred_troughs_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = processed_labels_test[
                                                     idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)

        def _reform_data_from_dict(data, flatten=True):
            sort_data = sorted(data.items(), key=lambda x: x[0])
            sort_data = [i[1] for i in sort_data]
            sort_data = torch.cat(sort_data, dim=0)
            if flatten:
                sort_data = np.reshape(sort_data.cpu(), (-1))
            else:
                sort_data = np.array(sort_data.cpu())
            return sort_data

        chunk_size = 180
        chunk_num = 0
        fs = 30

        for subj_index in predictions.keys():
            prediction = np.array(_reform_data_from_dict(predictions[subj_index]))
            label = np.array(_reform_data_from_dict(labels[subj_index]))
            original_label = label.copy()  # Copy original labels for comparison

            prediction = process_signal(prediction)
            label = process_signal(label)

            if chunk_size == -1:
                chunk_size = len(prediction)
                chunk_num = 0

            start = (chunk_num) * chunk_size
            stop = (chunk_num + 1) * chunk_size
            samples = stop - start
            x_time = np.linspace(0, samples / fs, num=samples)

            plt.figure(figsize=(10, 5))
            plt.plot(x_time, prediction[start:stop], color='r', label='Predicted PPG')
            plt.plot(x_time, label[start:stop], color='black', label='Processed Ground Truth')
            plt.plot(x_time, original_label[start:stop], linestyle='dotted', color='grey',
                     label='Original Ground Truth')  # Add original signal with dotted line
            plt.title(f'Trial: {subj_index} - Batch: {chunk_num}')
            plt.legend(['Predictions', 'Processed Ground Truth', 'Original Ground Truth'])
            plt.xlabel('Time (s)')

            current_dir = os.path.basename(self.model_dir)
            hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()
            waveform_dir = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name, "test_waveforms")
            if not os.path.exists(waveform_dir):
                os.makedirs(waveform_dir)

            plt.savefig(os.path.join(waveform_dir, f"waveform_trial_{subj_index}_batch_{chunk_num}.png"))
            plt.close()

        print('')
        # calculate_metrics(predictions, labels, self.config)

    def plot_waveforms(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(data_loader)
            batch = next(data_iter)
            if len(batch) == 2:
                data, labels = batch
            else:
                data, labels, *other = batch
            data, labels = data.to(self.device), labels.to(self.device)
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)

            pred_ppg, pred_peaks, pred_troughs = self.model(data)
            pred_ppg = pred_ppg.cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()

            plt.figure(figsize=(10, 4))
            plt.plot(pred_ppg, label="Predicted PPG", color='blue')
            plt.plot(labels, label="Ground Truth", color='green')
            plt.legend()
            plt.title(f"Epoch {epoch} - Predicted PPG vs Ground Truth")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            # Ensure the directory exists
            current_dir = os.path.basename(self.model_dir)
            hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()
            waveform_dir = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name, "waveforms")
            if not os.path.exists(waveform_dir):
                os.makedirs(waveform_dir)

            plt.savefig(os.path.join(waveform_dir, f"waveform_epoch_{epoch}.png"))
            plt.close()

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        current_dir = os.path.basename(self.model_dir)
        hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()
        full_path = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name)

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        model_file_name = self.model_file_name + f'_Epoch{index}.pth'
        model_path = os.path.join(full_path, model_file_name)
        print(f"Saving model to: {model_path}")
        torch.save(self.model.state_dict(), model_path)
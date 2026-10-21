import logging
import os
from collections import OrderedDict
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.TS_CAN_Adam import TS_CAN_Adam
from neural_methods.model.TS_CAN_v1 import TS_CAN_v1
from tqdm import tqdm
import hashlib
import matplotlib.pyplot as plt
import numpy as np
# import ipywidgets as widgets
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
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt
from dtw import accelerated_dtw
from scipy.stats import pearsonr
import torch.nn as nn
import pickle


def generate_peak_labels(labels):
    labels_np = labels.cpu().numpy().flatten()
    peaks, _ = find_peaks(labels_np, prominence=1.0, distance=10)
    peak_labels = np.zeros_like(labels_np)
    peak_labels[peaks] = 1
    return torch.tensor(peak_labels, dtype=torch.float32).to(labels.device).view(-1, 1)

def generate_trough_labels(labels):
    labels_np = labels.cpu().numpy().flatten()
    peak_indices, _ = find_peaks(labels_np, prominence=1.0, distance=10)
    trough_labels = np.zeros_like(labels_np)

    for i in range(len(peak_indices) - 1):
        start = peak_indices[i]
        end = peak_indices[i + 1]
        troughs, _ = find_peaks(-labels_np[start:end], prominence=0.5)
        if len(troughs) > 0:
            deepest_trough = start + troughs[np.argmax(-labels_np[start:end][troughs])]
            trough_labels[deepest_trough] = 1

    return torch.tensor(trough_labels, dtype=torch.float32).to(labels.device).view(-1, 1)

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


class TS_CAN_v1_Trainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = TS_CAN_v1(img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.criterion_peak = torch.nn.MSELoss()
            self.criterion_trough = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
            if config.TRAIN.LOAD_MODEL_PATH:
                self.load_model(config.TRAIN.LOAD_MODEL_PATH)

        elif config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
            self.model.eval()
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

    def calculate_snr_loss(self, signal, noise):
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        snr_loss = -snr
        #print(f'Signal Power: {signal_power.item()}, Noise Power: {noise_power.item()}, SNR: {snr.item()}')
        return snr_loss

    def train(self, data_loader):
        if self.config.TOOLBOX_MODE == "only_test":
            print("Skipping training as TOOLBOX_MODE is set to only_test.")
            return

        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss_ppg = 0.0
            running_loss_peaks = 0.0
            running_loss_troughs = 0.0
            running_loss_snr = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                processed_labels = torch.tensor(np.array([process_signal(label.cpu().numpy()) for label in labels]), dtype=torch.float32).to(labels.device).view(-1, 1)
                peak_labels = generate_peak_labels(processed_labels)
                trough_labels = generate_trough_labels(processed_labels)

                self.optimizer.zero_grad()
                pred_ppg, pred_peaks, pred_troughs = self.model(data)
                weights = torch.abs(labels) + 1e-6


                loss_ppg = torch.mean(weights * self.criterion(pred_ppg, processed_labels))
                loss_peaks = torch.mean(weights * self.criterion_peak(pred_peaks, peak_labels))
                loss_troughs = torch.mean(weights * self.criterion_trough(pred_troughs, trough_labels))
                snr_loss = self.calculate_snr_loss(pred_ppg, pred_ppg - processed_labels)
                loss = loss_ppg
                # loss = 3 * loss_ppg + 4 * loss_peaks + 4 * loss_troughs + 0.13 * snr_loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss_ppg += loss_ppg.item()
                running_loss_peaks += loss_peaks.item()
                running_loss_troughs += loss_troughs.item()
                running_loss_snr += snr_loss.item()
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

            avg_loss_ppg = running_loss_ppg / len(data_loader["train"])
            avg_loss_peaks = running_loss_peaks / len(data_loader["train"])
            avg_loss_troughs = running_loss_troughs / len(data_loader["train"])
            avg_loss_snr = running_loss_snr / len(data_loader["train"])
            avg_loss = sum(train_loss) / len(train_loss)

            print(
                f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, PPG Loss: {avg_loss_ppg:.4f}, Peak Loss: {avg_loss_peaks:.4f}, Trough Loss: {avg_loss_troughs:.4f}, SNR Loss: {avg_loss_snr:.4f}")

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, valid_loss_ppg, valid_loss_peaks, valid_loss_troughs, valid_loss_snr = self.valid(data_loader)
                print(
                    f'Epoch {epoch}: Validation Loss: {valid_loss:.4f}, PPG Loss: {valid_loss_ppg:.4f}, Peak Loss: {valid_loss_peaks:.4f}, Trough Loss: {valid_loss_troughs:.4f}, SNR Loss: {valid_loss_snr:.4f}')
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

            self.plot_waveforms(epoch, data_loader["train"])
            self.save_model(epoch)

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        valid_loss_ppg = 0.0
        valid_loss_peaks = 0.0
        valid_loss_troughs = 0.0
        valid_loss_snr = 0.0
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)

                processed_labels_valid = torch.tensor(np.array([process_signal(label.cpu().numpy()) for label in labels_valid]), dtype=torch.float32).to(labels_valid.device).view(-1, 1)
                peak_labels_valid = generate_peak_labels(processed_labels_valid)
                trough_labels_valid = generate_trough_labels(processed_labels_valid)

                pred_ppg_valid, pred_peaks_valid, pred_troughs_valid = self.model(data_valid)

                weights = torch.abs(labels_valid) + 1e-6
                loss_ppg = torch.mean(weights * self.criterion(pred_ppg_valid, processed_labels_valid))
                loss_peaks = torch.mean(weights * self.criterion_peak(pred_peaks_valid, peak_labels_valid))
                loss_troughs = torch.mean(weights * self.criterion_trough(pred_troughs_valid, trough_labels_valid))
                snr_loss = self.calculate_snr_loss(pred_ppg_valid, pred_ppg_valid - processed_labels_valid)
                loss = 3 * loss_ppg + 4 * loss_peaks + 4 * loss_troughs + 0.13 * snr_loss
                valid_loss_ppg += loss_ppg.item()
                valid_loss_peaks += loss_peaks.item()
                valid_loss_troughs += loss_troughs.item()
                valid_loss_snr += snr_loss.item()
                valid_loss.append(loss.item())
                valid_step += 1

        avg_loss_ppg = valid_loss_ppg / valid_step
        avg_loss_peaks = valid_loss_peaks / valid_step
        avg_loss_troughs = valid_loss_troughs / valid_step
        avg_loss_snr = valid_loss_snr / valid_step
        avg_loss = sum(valid_loss) / len(valid_loss)
        return avg_loss, avg_loss_ppg, avg_loss_peaks, avg_loss_troughs, avg_loss_snr


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
            for test_batch_idx, test_batch in enumerate(data_loader['test']):
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

        fs = 30
        segment_duration = 4
        segment_length = segment_duration * fs

        for subj_index in predictions.keys():
            prediction = np.array(_reform_data_from_dict(predictions[subj_index]))
            label = np.array(_reform_data_from_dict(labels[subj_index]))

            prediction = process_signal(prediction)
            label = process_signal(label)

            num_segments = len(prediction) // segment_length + (1 if len(prediction) % segment_length != 0 else 0)

            all_predictions = []
            all_labels = []

            for segment_idx in range(num_segments):
                start = segment_idx * segment_length
                end = min((segment_idx + 1) * segment_length, len(prediction))
                x_time = np.linspace(0, (end - start) / fs, num=(end - start))

                segment_prediction = prediction[start:end]
                segment_label = label[start:end]

                all_predictions.append(segment_prediction)
                all_labels.append(segment_label)

                plt.figure(figsize=(10, 5))
                plt.plot(x_time, segment_prediction, color='r', linewidth=2.5, label='Predicted PPG')
                plt.plot(x_time, segment_label, color='black', linewidth=2.5, label='Ground Truth')

                plt.legend(prop={'size': 18, 'weight': 'bold'})
                plt.xlabel('Time (s)', fontsize=22, fontweight='bold')
                plt.ylabel("Amplitude", fontsize=22, fontweight='bold')

                plt.xticks(fontsize=13, fontweight='bold')
                plt.yticks(fontsize=13, fontweight='bold')

                plt.tight_layout()
                plt.margins(0)

                current_dir = os.path.basename(self.model_dir)
                hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()
                waveform_dir = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name, "test_waveforms")
                if not os.path.exists(waveform_dir):
                    os.makedirs(waveform_dir)

                plt.savefig(os.path.join(waveform_dir, f"waveform_trial_{subj_index}_segment_{segment_idx}.png"))
                plt.close()

            full_waveform_dir = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name, "full_waveforms")
            if not os.path.exists(full_waveform_dir):
                os.makedirs(full_waveform_dir)

            with open(os.path.join(full_waveform_dir, f"predictions_trial_{subj_index}.pkl"), 'wb') as f:
                pickle.dump(all_predictions, f)

            with open(os.path.join(full_waveform_dir, f"labels_trial_{subj_index}.pkl"), 'wb') as f:
                pickle.dump(all_labels, f)

        print('')
        calculate_metrics(predictions, labels, self.config)

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


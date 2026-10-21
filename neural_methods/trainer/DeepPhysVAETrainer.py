"""Trainer for DeepPhysVAE."""

import logging
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.DeepPhysVAE import DeepPhysVAE, VAE
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
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


class DeepPhysVAETrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
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

        if config.TOOLBOX_MODE == "train_and_test":
            self.deepphys = DeepPhys(img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
            self.vae = VAE().to(self.device)
            self.model = DeepPhysVAE(self.deepphys, self.vae).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.deepphys = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.H).to(self.device)
            self.vae = VAE().to(self.device)
            self.model = DeepPhysVAE(self.deepphys, self.vae).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhysVAE trainer initialized in incorrect toolbox mode!")

    def vae_loss_function(self, recon_x, x, mu, logvar):
        BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

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
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                pred_ppg, vae_reconstructed, mu, logvar = self.model(data)
                vae_reconstructed = vae_reconstructed[:, :, :H, :W]
                # Loss function
                pred_loss = self.criterion(pred_ppg, labels)
                vae_loss = self.vae_loss_function(vae_reconstructed, data[:, 3:, :, :], mu, logvar)
                total_loss = pred_loss + 1.0 * vae_loss # We can adjust vae weight by using coefficeint here

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                running_loss += total_loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] total_loss: {running_loss / 100:.3f}, pred_loss: {pred_loss.item():.3f}, vae_loss: {vae_loss.item():.3f}')
                    running_loss = 0.0
                train_loss.append(total_loss.item())
                tbar.set_postfix({"total_loss": total_loss.item(), "pred_loss": pred_loss.item(), "vae_loss": vae_loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})
                #print(f"Batch {idx}: total_loss = {total_loss.item()}, pred_loss = {pred_loss.item()}, vae_loss = {vae_loss.item()}")
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
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
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid, vae_reconstructed, mu, logvar = self.model(data_valid)
                vae_reconstructed = vae_reconstructed[:, :, :H, :W]
                pred_loss = self.criterion(pred_ppg_valid, labels_valid)
                vae_loss = self.vae_loss_function(vae_reconstructed, data_valid[:, 3:, :, :], mu, logvar)
                total_loss = pred_loss + vae_loss
                valid_loss.append(total_loss.item())
                valid_step += 1
                vbar.set_postfix({"total_loss": total_loss.item(), "pred_loss": pred_loss.item(), "vae_loss": vae_loss.item()})
                print(f"Validation Batch {valid_idx}: total_loss = {total_loss.item()}, pred_loss = {pred_loss.item()}, vae_loss = {vae_loss.item()}")
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
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
                last_epoch_model_path = os.path.join(full_path, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(full_path, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
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
                pred_ppg_test, vae_reconstructed, mu, logvar = self.model(data_test)
                vae_reconstructed = vae_reconstructed[:, :, :H, :W]
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

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

        def _process_signal(signal, fs=30, diff_flag=True):
            use_bandpass = True
            if diff_flag:
                gt_bvp = _detrend(np.cumsum(signal), 100)
            else:
                gt_bvp = _detrend(signal, 100)
            if use_bandpass:
                [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
                signal = filtfilt(b, a, np.double(signal))
            return signal

        def _detrend(input_signal, lambda_value):
            signal_length = input_signal.shape[0]
            H = np.identity(signal_length)
            ones = np.ones(signal_length)
            minus_twos = -2 * np.ones(signal_length)
            diags_data = np.array([ones, minus_twos, ones])
            diags_index = np.array([0, 1, 2])
            D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
            detrended_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
            return detrended_signal


        chunk_size = 180
        chunk_num = 0
        fs = 30

        for subj_index in predictions.keys():
            prediction = np.array(_reform_data_from_dict(predictions[subj_index]))
            label = np.array(_reform_data_from_dict(labels[subj_index]))

            # Process signals
            prediction = _process_signal(prediction, fs, diff_flag=True)
            label = _process_signal(label, fs, diff_flag=True)

            if chunk_size == -1:
                chunk_size = len(prediction)
                chunk_num = 0

            start = (chunk_num) * chunk_size
            stop = (chunk_num + 1) * chunk_size
            samples = stop - start
            x_time = np.linspace(0, samples / fs, num=samples)

            plt.figure(figsize=(10, 5))
            plt.plot(x_time, prediction[start:stop], color='r')
            plt.plot(x_time, label[start:stop], color='black')
            plt.title('Trial: ' + str(subj_index))
            plt.legend(['Predictions', 'Labels'])
            plt.xlabel('Time (s)')
            plt.show()

        print('')
        calculate_metrics(predictions, labels, self.config)


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        current_dir = os.path.basename(self.model_dir)
        hashed_subdir_name = hashlib.md5(current_dir.encode()).hexdigest()  # have to use hash to solve too long file name

        full_path = os.path.join(os.path.dirname(self.model_dir), hashed_subdir_name)

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        model_file_name = self.model_file_name + f'_Epoch{index}.pth'
        model_path = os.path.join(full_path, model_file_name)
        print(f"Saving model to: {model_path}")
        torch.save(self.model.state_dict(), model_path)

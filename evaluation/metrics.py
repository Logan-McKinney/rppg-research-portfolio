import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.signal import butter, filtfilt, find_peaks

def read_label(dataset):
    """Read manually corrected labels。"""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels。"""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts."""
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))

def calculate_snr(signal, noise):
    """Calculate the Signal-to-Noise Ratio (SNR)"""
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    return 10 * np.log10(signal_power / noise_power)

def calculate_peak_shape_score(predicted, original):
    """Calculate the Peak Shape Score"""
    peak_similarity = np.corrcoef(predicted, original)[0, 1]
    peak_shape_score = 1 - peak_similarity
    return peak_shape_score

def calculate_peak_trough_amplitude_distances(prediction, label):
    prediction_np = prediction.flatten()
    label_np = label.flatten()

    # Find peaks in the label
    peaks_label, _ = find_peaks(label_np, prominence=1.0, distance=10)
    peak_amplitude_distances = []
    for peak in peaks_label:
        # Find the corresponding peak in the prediction within the same range
        range_start = max(0, peak - 10)
        range_end = min(len(prediction_np), peak + 10)
        peaks_pred, _ = find_peaks(prediction_np[range_start:range_end], prominence=1.0)
        if peaks_pred.size > 0:
            # Adjust the index to the global position
            peak_pred = peaks_pred[np.argmax(prediction_np[range_start:range_end][peaks_pred])] + range_start
            peak_amplitude_distances.append(np.abs(label_np[peak] - prediction_np[peak_pred]))

    # Find troughs in the label
    troughs_label, _ = find_peaks(-label_np, prominence=1.0, distance=10)
    trough_amplitude_distances = []
    for trough in troughs_label:
        # Find the corresponding trough in the prediction within the same range
        range_start = max(0, trough - 10)
        range_end = min(len(prediction_np), trough + 10)
        troughs_pred, _ = find_peaks(-prediction_np[range_start:range_end], prominence=1.0)
        if troughs_pred.size > 0:
            # Adjust the index to the global position
            trough_pred = troughs_pred[np.argmax(-prediction_np[range_start:range_end][troughs_pred])] + range_start
            trough_amplitude_distances.append(np.abs(label_np[trough] - prediction_np[trough_pred]))

    return np.mean(peak_amplitude_distances) if peak_amplitude_distances else None, np.mean(trough_amplitude_distances) if trough_amplitude_distances else None

def preprocess_signal(signal):
    """Preprocess the signal by detrending and normalization."""
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)
    return signal

def calculate_secondary_peak_trough_distances(prediction, label):
    prediction_np = prediction.flatten()
    label_np = label.flatten()

    # Find main peaks in the label
    main_peaks_label, _ = find_peaks(label_np, prominence=1.0, distance=10)
    secondary_peak_distances = []
    secondary_trough_distances = []

    for i in range(len(main_peaks_label) - 1):
        start = main_peaks_label[i]
        end = main_peaks_label[i + 1]

        # Find secondary peaks in the label
        peaks_label, _ = find_peaks(label_np[start:end], prominence=0.5, distance=5)
        if len(peaks_label) > 0:
            for peak in peaks_label:
                secondary_peak_label = start + peak

                # Find corresponding peak in the prediction within the range of +/-2 from the secondary peak in label
                range_start = max(0, secondary_peak_label - 2)
                range_end = min(len(prediction_np), secondary_peak_label + 3)  # +3 to include secondary_peak_label + 2
                peaks_pred, _ = find_peaks(prediction_np[range_start:range_end], prominence=0.1)

                if len(peaks_pred) > 0:
                    # Ensure the found peak is not the main peak in the prediction
                    main_peaks_pred, _ = find_peaks(prediction_np, prominence=1.0, distance=10)
                    peaks_pred = [p for p in peaks_pred if range_start + p not in main_peaks_pred]

                    if len(peaks_pred) > 0:
                        secondary_peak_pred = range_start + peaks_pred[np.argmax(prediction_np[range_start:range_end][peaks_pred])]
                        secondary_peak_distances.append(np.abs(label_np[secondary_peak_label] - prediction_np[secondary_peak_pred]))

        # Find secondary troughs in the label
        troughs_label, _ = find_peaks(-label_np[start:end], prominence=0.5, distance=5)
        if len(troughs_label) > 0:
            for trough in troughs_label:
                secondary_trough_label = start + trough

                # Find corresponding trough in the prediction within the range of +/-2 from the secondary trough in label
                range_start = max(0, secondary_trough_label - 2)
                range_end = min(len(prediction_np), secondary_trough_label + 3)  # +3 to include secondary_trough_label + 2
                troughs_pred, _ = find_peaks(-prediction_np[range_start:range_end], prominence=0.1)

                if len(troughs_pred) > 0:
                    # Ensure the found trough is not the main trough in the prediction
                    main_troughs_pred, _ = find_peaks(-prediction_np, prominence=1.0, distance=10)
                    troughs_pred = [t for t in troughs_pred if range_start + t not in main_troughs_pred]

                    if len(troughs_pred) > 0:
                        secondary_trough_pred = range_start + troughs_pred[np.argmin(-prediction_np[range_start:range_end][troughs_pred])]
                        secondary_trough_distances.append(np.abs(label_np[secondary_trough_label] - prediction_np[secondary_trough_pred]))

    return np.mean(secondary_peak_distances) if secondary_peak_distances else None, np.mean(secondary_trough_distances) if secondary_trough_distances else None

def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    predict_waveforms = list()
    gt_waveforms = list()
    peak_amplitude_distances_all = list()
    trough_amplitude_distances_all = list()
    secondary_peak_distances_all = list()
    secondary_trough_distances_all = list()
    dtw_distances_all = list()

    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        predict_waveforms.append(preprocess_signal(prediction.cpu().numpy()))
        gt_waveforms.append(preprocess_signal(label.cpu().numpy()))

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

        # Calculate peak and trough amplitude distances
        peak_amplitude_distance, trough_amplitude_distance = calculate_peak_trough_amplitude_distances(prediction, label)
        if peak_amplitude_distance is not None:
            peak_amplitude_distances_all.append(peak_amplitude_distance)
        if trough_amplitude_distance is not None:
            trough_amplitude_distances_all.append(trough_amplitude_distance)

        # Calculate secondary peak and trough amplitude distances
        secondary_peak_distance, secondary_trough_distance = calculate_secondary_peak_trough_distances(prediction, label)
        if secondary_peak_distance is not None:
            secondary_peak_distances_all.append(secondary_peak_distance)
        if secondary_trough_distance is not None:
            secondary_trough_distances_all.append(secondary_trough_distance)

    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)

    predict_waveforms_concatenated = np.concatenate(predict_waveforms)
    gt_waveforms_concatenated = np.concatenate(gt_waveforms)

    # Find peaks in the concatenated waveforms
    peaks_pred, _ = find_peaks(predict_waveforms_concatenated, prominence=1.0, distance=10)
    peaks_label, _ = find_peaks(gt_waveforms_concatenated, prominence=1.0, distance=10)

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MSE":
            MSE = mean_squared_error(gt_waveforms_concatenated, predict_waveforms_concatenated)
            print("Waveform MSE: {0}".format(MSE))

        elif metric == "PSNR":
            mse = mean_squared_error(gt_waveforms_concatenated, predict_waveforms_concatenated)
            psnr = 20 * np.log10(np.max(gt_waveforms_concatenated) / np.sqrt(mse))
            print("Waveform PSNR: {0}".format(psnr))

        elif metric == "Correlation":
            corr = np.corrcoef(gt_waveforms_concatenated, predict_waveforms_concatenated)[0, 1]
            print("Waveform Correlation Coefficient: {0}".format(corr))

        elif metric == "DTW":
            dtw_distances = []
            for i in range(len(peaks_label)):
                if i < len(peaks_pred):
                    peak = peaks_label[i]
                    pred_peak = peaks_pred[i]
                    range_start = max(0, peak - 10)
                    range_end = min(len(gt_waveforms_concatenated), peak + 10)
                    pred_range_start = max(0, pred_peak - 10)
                    pred_range_end = min(len(predict_waveforms_concatenated), pred_peak + 10)
                    dtw_distance, _ = fastdtw(gt_waveforms_concatenated[range_start:range_end].flatten().astype(float), predict_waveforms_concatenated[pred_range_start:pred_range_end].flatten().astype(float), dist=lambda a, b: abs(float(a) - float(b))) # need to flatten because it may return a 2d vector instead of 1d
                    dtw_distances.append(dtw_distance)
            avg_dtw_distance = np.mean(dtw_distances)
            std_dtw_distance = np.std(dtw_distances)
            print("Average Waveform DTW Distance: {0}".format(avg_dtw_distance))
            print("STD Waveform DTW Distance: {0}".format(std_dtw_distance))

        elif metric == "SNR":
            signal = gt_waveforms_concatenated
            noise = signal - predict_waveforms_concatenated
            snr = calculate_snr(signal, noise)
            print("Waveform SNR: {0}".format(snr))

        elif metric == "Peak Shape Score":
            peak_shape_score = calculate_peak_shape_score(predict_waveforms_concatenated, gt_waveforms_concatenated)
            print("Peak Shape Score: {0}".format(peak_shape_score))

        elif metric == "Peak Distance":
            avg_peak_amplitude_distance = np.mean(peak_amplitude_distances_all)
            std_peak_amplitude_distance = np.std(peak_amplitude_distances_all)
            avg_secondary_peak_distance = np.mean(secondary_peak_distances_all)
            std_secondary_peak_distance = np.std(secondary_peak_distances_all)
            print("Average Peak Amplitude Distance: {0}".format(avg_peak_amplitude_distance))
            print("STD Peak Amplitude Distance: {0}".format(std_peak_amplitude_distance))
            print("Average Secondary Peak Amplitude Distance: {0}".format(avg_secondary_peak_distance))
            print("STD Secondary Peak Amplitude Distance: {0}".format(std_secondary_peak_distance))

        elif metric == "Trough Distance":
            avg_trough_amplitude_distance = np.mean(trough_amplitude_distances_all)
            std_trough_amplitude_distance = np.std(trough_amplitude_distances_all)
            avg_secondary_trough_distance = np.mean(secondary_trough_distances_all)
            std_secondary_trough_distance = np.std(secondary_trough_distances_all)
            print("Average Trough Amplitude Distance: {0}".format(avg_trough_amplitude_distance))
            print("STD Trough Amplitude Distance: {0}".format(std_trough_amplitude_distance))
            print("Average Secondary Trough Amplitude Distance: {0}".format(avg_secondary_trough_distance))
            print("STD Secondary Trough Amplitude Distance: {0}".format(std_secondary_trough_distance))

        else:
            raise ValueError("Wrong Test Metric Type")


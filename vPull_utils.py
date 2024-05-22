import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def vPull_analysis(input: np.ndarray, properties: dict, FPS: int, keypoint_mapping: list) -> dict:
    """
    Perform the vPull analysis which includes step detection, pull detection, and bending detection.
    
    Args:
        input (np.ndarray): 3D array of keypoint data.
        properties (dict): Dictionary containing analysis properties.
        FPS (int): Frames per second of the input data.
        keypoint_mapping (list): List of keypoint indices and names.

    Returns:
        dict: Analysis outcomes including latency, recovery, and bending metrics.
    """
    # Extract filter properties from the properties dictionary
    filter_props = properties.get('filter_properties', {})
    order = filter_props.get('order', 2)
    cutoff = filter_props.get('cutoff', 10)
    gap_thr = filter_props.get('gap_thr', 1)

    # Filter and interpolate the pose sequence data
    input = filter_pose_sequence(input, order=order, cutoff=cutoff, gap_thr=gap_thr, FPS=FPS)
    
    step_kpts = [int(index) for index, name in keypoint_mapping if name in properties['step_keypoints']]
    step_outcome = step_detection(input[step_kpts, :, :], properties=properties, FPS=FPS)
    
    pull_kpts = [int(index) for index, name in keypoint_mapping if name in properties['pull_keypoints']]
    pull_outcome = pull_detection(input[pull_kpts, :, :], FPS=FPS, step_properties=step_outcome, properties=properties)
    
    bending_kpts = [int(index) for index, name in keypoint_mapping if name in properties['bending_keypoints']]
    bending_outcome = bending_detection(input, input[bending_kpts, :, :], FPS=FPS, pull_properties=pull_outcome, properties=properties)

    step_outcome['1st_step_latency'] = step_outcome['step_onset'] - pull_outcome['pull_onset']
    step_outcome['step_termination'] = step_outcome['step_termination'] - pull_outcome['pull_onset']
    del step_outcome['step_onset']
    
    pull_outcome.update(step_outcome)
    pull_outcome.update(bending_outcome)
    pull_outcome['recover_latency'] = np.nanmax([pull_outcome['step_termination'], pull_outcome['bending_recover_latency']])
    pull_outcome['recovered'] = not np.isnan(pull_outcome['recover_latency'])
    
    return pull_outcome

def filter_pose_sequence(input: np.ndarray, order: int, cutoff: float, gap_thr: int, FPS: int) -> np.ndarray:
    """
    Filter and interpolate the pose sequence data.
    
    Args:
        input (np.ndarray): 3D array of keypoint data.
        order (int): The order of the filter.
        cutoff (float): The cutoff frequency of the filter.
        gap_thr (int): The threshold for gap interpolation.
        FPS (int): Frames per second of the input data.

    Returns:
        np.ndarray: Filtered and interpolated keypoint data.
    """
    b, a = butter(order, cutoff / (FPS * 2), btype='low')

    def interpolate_and_filter(ts: pd.Series) -> np.ndarray:
        if ts.isnull().any():
            ts = ts.interpolate(method='cubic', limit=int(FPS * gap_thr))
        ts = ts.fillna(0).values
        ts_filtered = filtfilt(b, a, ts)
        ts_filtered[np.isnan(ts)] = np.nan
        return ts_filtered

    return np.array([[interpolate_and_filter(pd.Series(input[i, j, :])) for j in range(input.shape[1])] for i in range(input.shape[0])])

def step_detection(keypoints: np.ndarray, properties: dict, FPS: int) -> dict:
    """
    Detect steps from the input keypoint data.
    
    Args:
        keypoints (np.ndarray): 3D array of keypoint data for step detection.
        properties (dict): Dictionary containing analysis properties.
        FPS (int): Frames per second of the input data.
    
    Returns:
        dict: Step detection outcomes.
    """
    baseline = int(properties['pull_baseline'] * FPS)
    timestamps, coordinates = keypoints[0, 0, :], keypoints[:, 1:4, :]
    distances = np.linalg.norm(coordinates, axis=1)

    time_diff = np.diff(timestamps)
    velocity = np.diff(distances, axis=1) / time_diff

    step_array = np.abs(velocity[:, baseline:]) > properties['step_vel_thr']
    step_events = []
    column = 0
    mode = True
    row = []
    while column < step_array.shape[1]:
        if mode:
            idx = np.argwhere(step_array[:, column:])
            if len(idx) > 0:
                idx = idx[np.argsort(idx[:, 1])]
                column += idx[0][1]
                row = idx[0][0]
                mode = False
            else:
                break
        else:
            idx = np.argwhere(~step_array[row, column:])
            if len(idx) > 0:
                step_init = column + baseline
                column += idx[0][0]
                step_end = column + baseline
                step_duration = timestamps[step_end + 1] - timestamps[step_init + 1]
                step_length = distances[row, step_end] - distances[row, step_init]
                step_direction = coordinates[row, 1, step_end] - coordinates[row, 1, step_init]
                if step_events:
                    step_interval = timestamps[step_init + 1] - step_events[-1][2]
                    if step_direction < 0:
                        break
                else:
                    step_interval = 0

                if step_length > properties['step_len_thr'] and step_interval < properties['step_interval_thr']:
                    step_events.append([row, timestamps[step_init + 1], timestamps[step_end + 1], step_duration, step_length])
                row = []
                mode = True
            else:
                break
    step_events = np.asarray(step_events)
    if step_events.any():
        stepping_response = True
        step_onset = step_events[0][1]
        step_termination = step_events[-1][2]
        first_step_duration = step_events[0][3]
        first_step_length = step_events[0][4]
        first_step_velocity = first_step_length / first_step_duration
        nsteps = step_events.shape[0]
    else:
        stepping_response = False
        step_onset = np.nan
        step_termination = np.nan
        first_step_duration = np.nan
        first_step_length = np.nan
        first_step_velocity = np.nan
        nsteps = 0
    outcome = {
        'stepping_response': stepping_response,
        'step_onset': step_onset,
        'step_termination': step_termination,
        'step_number': nsteps,
        '1st_step_duration': first_step_duration,
        '1st_step_length': first_step_length,
        '1st_step_velocity': first_step_velocity,
    }
    return outcome

def pull_detection(keypoints: np.ndarray, FPS: int, step_properties: dict, properties: dict) -> dict:
    """
    Detect pull from the input keypoint data.
    
    Args:
        keypoints (np.ndarray): 3D array of keypoint data for pull detection.
        FPS (int): Frames per second of the input data.
        step_properties (dict): Outcomes from the step detection.
        properties (dict): Dictionary containing analysis properties.
    
    Returns:
        dict: Pull detection outcomes.
    """
    baseline = int(properties['pull_baseline'] * FPS)
    timestamps, coordinates = keypoints[0, 0, :], keypoints[:, 1:4, :]
    coordinates = np.linalg.norm(coordinates, axis=1)
    time_diff = np.diff(timestamps)
    velocity = np.diff(coordinates, axis=1) / time_diff
    acceleration = np.diff(velocity, axis=1) / time_diff[1:]
    acceleration = np.nanmean(acceleration, axis=0)

    base_acc = acceleration[:baseline]
    base_acc_mean = np.nanmean(base_acc)
    base_acc_std = np.nanstd(base_acc)
    threshold = base_acc_mean + properties['pull_thr'] * base_acc_std

    # First guess for pull onset
    pull = acceleration[baseline:] > threshold
    if not pull.any():
        # Maybe too early pull? Try baseline acceleration at the end of the recording
        base_acc = acceleration[-baseline:]
        base_acc_mean = np.nanmean(base_acc)
        base_acc_std = np.nanstd(base_acc)
        threshold = base_acc_mean + properties['pull_thr'] * base_acc_std
        pull = acceleration > threshold
        baseline = 0

    if pull.any():
        if not np.isnan(step_properties['step_onset']):
            stop = np.nanargmin(np.abs(timestamps - step_properties['step_onset']))
        else:
            stop = len(acceleration)
        peaks, _ = find_peaks(acceleration[baseline:stop])
        if not peaks.any():
            peaks = np.array([np.nanargmax(acceleration[baseline:stop])])
        peaks = peaks + baseline
        if peaks.any():
            pull_magn = np.max(acceleration[peaks])
            pull_peak_time = peaks[np.argmax(acceleration[peaks])]
            # Get pull onset time = last minimum before pull peak
            peaks, _ = find_peaks(-1 * acceleration[:pull_peak_time])
            if not peaks.any():
                peaks = [np.nanargmax(-1 * acceleration[:pull_peak_time])]
            pull_onset = peaks[-1]
            if pull_onset:
                pull_time = timestamps[pull_onset + 2]
        else:
            pull_time = np.nan
            pull_magn = np.nan
    else:
        pull_time = np.nan
        pull_magn = np.nan
    outcome = {
        'pull_onset': pull_time,
        'pull_magnitude': pull_magn,
    }
    return outcome

def bending_detection(all_keypoints: np.ndarray, keypoints: np.ndarray, FPS: int, pull_properties: dict, properties: dict) -> dict:
    """
    Detect bending from the input keypoint data.
    
    Args:
        all_keypoints (np.ndarray): 3D array of all keypoint data.
        keypoints (np.ndarray): 3D array of keypoint data for bending detection.
        FPS (int): Frames per second of the input data.
        pull_properties (dict): Outcomes from the pull detection.
        properties (dict): Dictionary containing analysis properties.
    
    Returns:
        dict: Bending detection outcomes.
    """
    if not np.isnan(pull_properties['pull_onset']):
        baseline = int(properties['pull_baseline'] * FPS)
        bending_angle = []
        timestamps, coordinates = keypoints[0, 0, :], keypoints[:, [1, 2, 3], :]
        coordinates = coordinates[~np.isnan(coordinates).any(axis=(1, 2))]
        
        for frame_idx in range(coordinates.shape[2]):
            frame = coordinates[:, :, frame_idx]
            if len(frame) >= 3:
                cov_matrix = np.cov(frame, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                min_eigenvalue_index = np.argmin(eigenvalues)
                normal_vector = eigenvectors[:, min_eigenvalue_index]
                # Calculate the normal vector to the plane
                if normal_vector[1] > 0:
                    normal_vector *= -1  # Flip the normal vector if the sign of y-component changes
                # Calculate the angle between the normal vector and the Z-axis
                angle = np.degrees(np.arccos(np.dot(normal_vector, [0, 0, 1]) / (np.linalg.norm(normal_vector) * np.linalg.norm([0, 0, 1])))) - 90
                if np.any(~np.isnan(bending_angle)):
                    last_angle = next((value for value in reversed(bending_angle) if not np.isnan(value)), None)
                    if np.abs(last_angle - angle) < 20:
                        bending_angle.append(angle)
                    else:
                        bending_angle.append(np.nan)
                else:
                    bending_angle.append(angle)  

        bending_angle = np.array(bending_angle)
        # Interpolate gaps
        ts = pd.Series(bending_angle)
        if ts.isnull().any():
            ts = ts.interpolate(method='cubic')
        bending_angle = ts.values

        pull_onset = np.where(timestamps == pull_properties['pull_onset'])[0][0]
        base_bending = np.nanmean(bending_angle[:baseline])
        bending_amplitude = np.nanmin(bending_angle[pull_onset:]) - base_bending
        max_bending = np.argmin(bending_angle[pull_onset:])
        max_bending += pull_onset
        if max_bending < 0:
            max_bending = 0
        bending_thr = base_bending + properties['bending_thr'] * bending_amplitude
        bending_terminated = np.where(bending_angle[max_bending:] > bending_thr)[0]
        if len(bending_terminated) > 0:
            bending_terminated = max_bending + bending_terminated[0]
            bending_recovered = True
            bending_recover_latency = (bending_terminated - pull_onset) / FPS
        else:
            bending_terminated = np.nan
            bending_recovered = False
            bending_recover_latency = np.nan
    else:
        timestamps = keypoints[0, 0, :]
        bending_angle = np.full(np.shape(timestamps), np.nan)
        bending_amplitude = np.nan
        bending_terminated = np.nan
        max_bending = np.nan
        bending_thr = np.nan
        bending_recover_latency = np.nan
        bending_recovered = False
    
    outcome_bending = {
        'bending_amplitude': bending_amplitude,
        'bending_recovered': bending_recovered,
        'bending_recover_latency': bending_recover_latency 
    }

    if np.isnan(bending_terminated):
        max_bending = 0
        bending_terminated = 0
    return outcome_bending

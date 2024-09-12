"""
Utility functions for EEG Preprocessing, Microstate Computation & Preprocessing, Statistical Analysis and ROI Preprocessing & Selection

Enrique Almazán (2024)
"""

import utils as u

import os
import pickle
import json

import requests
import shutil
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

import csv

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA, create_eog_epochs
from scipy.signal import find_peaks

import scipy.stats as stats
from scipy.stats import t, kstest, shapiro, wilcoxon, friedmanchisquare

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.feature_selection import RFE, SequentialFeatureSelector

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

from itertools import combinations


# =============================================================================================================================================================
# DOWNLOAD DATA

def download_files_from_url(url, destination_directory):
    """
    Download files recursively from a given URL.

    Parameters:
    - url (str): The base URL to start downloading from.
    - destination_directory (str): The local directory where the files will be saved.

    Output:
    - Downloads files and directories to the specified local directory.
    """
    
    # Get the HTML content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all links on the page
    links = soup.find_all("a")
    
    # Iterate over each link found
    for index, link in enumerate(links):
        if index == 0:  # Skip the link to the parent directory
            continue
            
        href = link.get("href")
        new_directory_name = os.path.join(destination_directory, href)
        
        if os.path.exists(new_directory_name):  # Check if the directory already exists
            print(f"The directory '{new_directory_name}' already exists. Skipping download.")
            continue
            
        if href.endswith("/"):  # If the link is a directory, explore recursively
            subdir_url = urljoin(url, href)
            os.makedirs(new_directory_name, exist_ok=True)
            u.download_files_from_url(subdir_url, new_directory_name)
            
        else:  # If the link is a file, download it
            file_url = urljoin(url, href)
            filename = os.path.basename(urlparse(file_url).path)
            filepath = os.path.join(destination_directory, filename)
            u.download_file(file_url, filepath)



def download_file(url, destination):
    """
    Download a file from a URL.

    Parameters:
    - url (str): The URL of the file to download.
    - destination (str): The local path where the file will be saved.

    Output:
    - Saves the downloaded file to the specified local path.
    """
    
    response = requests.get(url)
    with open(destination, "wb") as file:
        file.write(response.content)





# =============================================================================================================================================================
# EEG SIGNAL PREPROCESSING

def preprocessing(eeg, sf=None, lf=1, hf=20, nc=7, t=1):
    """
    Function to preprocess the raw data, filtering, downsampling and removing ocular artifacts from EEG data using Independent Component Analysis (ICA).

    Parameters:
    - eeg (array-like): Input EEG data.
    - sf (int): Sampling frequency for sampling the EEG signal (default = 250 Hz). If None, the signal will not be sampled.
    - lf (float): Low-frequency cutoff for filtering EOG signal (default = 1 Hz).
    - hf (float): High-frequency cutoff for filtering EOG signal (default = 20 Hz).
    - nc (int): Number of components to decompose the EEG data into (default = 7).
    - t (float): Threshold for identifying EOG-related components (default = 1).

    Outputs:
    - cleaned_eeg (array-like): Cleaned EEG data with ocular artifacts removed.
    """
    
    # STEP 1: Setting montage for the electrode locations
    eeg.set_montage('standard_1020')
    
    # STEP 2: Downsample the raw signal if needed
    if sf == None:
        eeg_downsample = eeg.copy()
    else:
        eeg_downsample = eeg.copy().resample(sfreq=sf)
    
    # STEP 3: Filter the raw downsampled signal
    eeg_filtered = eeg_downsample.copy().filter(l_freq=lf, h_freq=hf, fir_design="firwin", verbose=False)

    # STEP 4: Apply the CAR (Common Average Reference) to the downsampled and filtered signal
    eeg_car, _ = mne.set_eeg_reference(eeg_filtered, ref_channels='average')
    
    # STEP 5: Compute epochs from ocular artifacts from the downsampled and filtered signal
    eog_epochs = create_eog_epochs(eeg_car, ch_name='VEOG', l_freq=lf, h_freq=hf, baseline=(-0.5, -0.2))
    
    # STEP 6: Train ica object, dividing in components the downsampled and filtered signal
    ica = ICA(n_components=nc, random_state=20)
    ica.fit(eeg_car)

    # STEP 7: Compute the indices which compose the ocular artefacts 
    eog_inds, _ = ica.find_bads_eog(eog_epochs, threshold=t, l_freq=lf, h_freq=hf)
    # In case no component is recognized as artifact we decrease the threshold as we know that each signal will have ocular artifacts due to the blinking
    while len(eog_inds) == 0:
        eog_inds, _ = ica.find_bads_eog(eog_epochs, threshold=t-0.1, l_freq=lf, h_freq=hf)

    # STEP 8: Remove the components related with the EOG from the raw data
    ica.exclude = eog_inds
    cleaned_eeg = eeg_car.copy()
    ica.apply(cleaned_eeg)

    return cleaned_eeg.pick_types(eeg=True, exclude=['VEOG'])



def dictionary(eeg):
    """
    Function to create a dictionary containing information about different types of events in EEG data.

    Parameters:
    - eeg (mne.io.Raw): Raw EEG data.
    - events (array-like): Array containing event information.

    Outputs:
    - d (dict): Dictionary containing information about different types of events in EEG data.
    """

    # Sampling frequency
    sf = eeg.info['sfreq']

    # Events of the eeg
    events, _ = mne.events_from_annotations(eeg)

    # Create a dictionary to store the information required
    d = {}

    # Create some instances setting them to 0
    c1 = 0
    c2 = 0
    c3 = 0

    # Create instance for starting time
    start_time = 0

    # Create an auxiliary list for the events
    #list_events = []
    
    for i, item in enumerate(events):
        #print('Event', i)
        
        if i != 0 and item[2] != events[i-1][2]:
            #print('Cambio:', item[2], events[i-1][2])
            
            prev_event = events[i-1][2]
            if prev_event == 99999 or prev_event == 10001:
                pass
            
            elif prev_event == 1:
                d['Dc' + str(c1 + 1)] = {
                    #'events': list_events,
                    'start_sample': start_time,
                    'end_sample': item[0],
                    #'start_time': start_time / sf,
                    #'end_time': item[0] / sf,
                    'time': (item[0] - start_time) / sf
                }
                c1 += 1
                
            elif prev_event == 200:
                d['Oe' + str(c2 + 1)] = {
                    #'events': list_events,
                    'start_sample': start_time,
                    'end_sample': item[0],
                    #'start_time': start_time / sf,
                    #'end_time': item[0] / sf,
                    'time': (item[0] - start_time) / sf
                }
                c2 += 1
                
            elif prev_event == 210 or prev_event == 208:
                d['Ce' + str(c3 + 1)] = {
                    #'events': list_events,
                    'start_sample': start_time,
                    'end_sample': item[0],
                    #'start_time': start_time / sf,
                    #'end_time': item[0] / sf,#
                    'time': (item[0] - start_time) / sf
                }
                c3 += 1
            
            # Reset the list of events and the time for the new event
            #list_events = []
            start_time = item[0]
        
        # Add event to the list
        #list_events.append(item)
    
    # Guardar los datos del último evento
    last_event = events[-1][2]
    if last_event == 99999 or last_event == 10001:
        pass
    elif last_event == 1:
        d['Dc' + str(c1 + 1)] = {
            #'events': list_events,
            'start_sample': start_time,
            'end_sample': item[0],
            #'start_time': start_time / sf,
            #'end_time': events[-1][0] / sf,
            'time': (events[-1][0] - start_time) / sf
        }
    elif last_event == 200:
        d['Oe' + str(c2 + 1)] = {
            #'events': list_events,
            'start_sample': start_time,
            'end_sample': item[0],
            #'start_time': start_time / sf,
            #'end_time': events[-1][0] / sf,
            'time': (events[-1][0] - start_time) / sf
        }
    elif last_event == 210 or prev_event == 208:
        d['Ce' + str(c3 + 1)] = {
            #'events': list_events,
            'start_sample': start_time,
            'end_sample': item[0],
            #'start_time': start_time / sf,
            #'end_time': events[-1][0] / sf,
            'time': (events[-1][0] - start_time) / sf
        }

    return d



def segment_times(d):
    """
    Function to extract segment start times for different event types from a dictionary.

    Parameters:
    - d (dict): Dictionary containing information about different types of events in EEG data.

    Outputs:
    - st_ce (list): List of segment start times for "Ce" events.
    - st_oe (list): List of segment start times for "Oe" events.
    """
    
    st_ce = []
    st_oe = []
    
    for key, value in d.items():
        if key.startswith('Ce'):
            st_ce.append(value['start_time'])
        elif key.startswith('Oe'):
            st_oe.append(value['start_time'])

    return st_ce, st_oe



def segment_samples(d):
    """
    Function to extract segment start and end samples for different event types from a dictionary containing information about different types of events in EEG data.

    Parameters:
    - d (dict): Dictionary containing information about different types of events in EEG data.

    Outputs:
    - ss_ce (list): List of segment start samples for "Closed Eyes" events.
    - ss_oe (list): List of segment start samples for "Open Eyes" events.
    - es_ce (list): List of segment end samples for "Closed Eyes" events.
    - es_oe (list): List of segment end samples for "Open Eyes" events.
    """

    ss_ce = []
    ss_oe = []
    es_ce = []
    es_oe = []
    
    for key, value in d.items():
        if key.startswith('Ce'):
            ss_ce.append(value['start_sample'])
        elif key.startswith('Oe'):
            ss_oe.append(value['start_sample'])
    
    for key, value in d.items():
        if key.startswith('Ce'):
            es_ce.append(value['end_sample'])
        elif key.startswith('Oe'):
            es_oe.append(value['end_sample'])

    return ss_ce, ss_oe, es_ce, es_oe



def pairs(ss_ce, ss_oe, es_ce, es_oe):
    """
    Matches the start and end samples, joins the corresponding pairs, and sorts them in ascending order (from first to last).

    Parameters:
    - ss_ce (list): List of start samples for condition 'ce'.
    - ss_oe (list): List of start samples for condition 'oe'.
    - es_ce (list): List of end samples for condition 'ce'.
    - es_oe (list): List of end samples for condition 'oe'.

    Output:
    - pairs (list): A sorted list of tuples, each containing a pair of start and end samples.
    """
    
    # Match the lists of start and end samples
    pairs_ce = list(zip(ss_ce, ss_oe))
    pairs_oe = list(zip(es_ce, es_oe))
    
    # Join the corresponding pairs and sort them in ascending order
    pairs = sorted(pairs_ce + pairs_oe)

    return pairs



def division_segments(data, ss, es):
    """
    Function to divide EEG data into segments based on given start and end sample indices.

    Parameters:
    - data (numpy.ndarray): EEG data array or a numpy array.
    - ss (list): List of segment start sample indices.
    - es (list): List of segment end sample indices.

    Outputs:
    - segments (list): List of EEG data segments.
    """

    segments = []

    for start, end in zip(ss, es):
        if len(data.shape) == 1:
            segment = data[start:end+1]
        else:
            segment = data[:, start:end+1]
        segments.append(segment)

    return segments



def segments_df(eeg, s_ce, s_oe):
    """
    Function to combine segments of "Closed Eyes" and "Open Eyes" events into a single list.

    Parameters:
    - eeg (mne.io.Raw): Raw EEG data.
    - s_ce (list): List of segments for "Ce" events.
    - s_oe (list): List of segments for "Oe" events.

    Ouput:
    - segments (list): Combined list of segments for "Closed Eyes" and "Open Eyes" events, each converted in a DataFrame.
    """
    
    segments = []
    if len(s_ce) == len(s_oe):
        for i in range(len(s_ce)):
            segments.append(pd.DataFrame(data=np.transpose(s_ce[i]), columns=eeg.ch_names))
            segments.append(pd.DataFrame(data=np.transpose(s_oe[i]), columns=eeg.ch_names))

    elif len(s_ce) > len(s_oe):
        for i in range(len(s_oe)):
            segments.append(pd.DataFrame(data=np.transpose(s_ce[i]), columns=eeg.ch_names))
            segments.append(pd.DataFrame(data=np.transpose(s_oe[i]), columns=eeg.ch_names))
        segments.append(pd.DataFrame(data=np.transpose(s_ce[-1]), columns=eeg.ch_names))

    elif len(s_oe) > len(s_ce):
        for i in range(len(s_ce)):
            segments.append(pd.DataFrame(data=np.transpose(s_ce[i]), columns=eeg.ch_names))
            segments.append(pd.DataFrame(data=np.transpose(s_oe[i]), columns=eeg.ch_names))
        segments.append(pd.DataFrame(data=np.transpose(s_oe[-1]), columns=eeg.ch_names))

    # Setting the float format, avoiding scientific notation
    pd.set_option('display.float_format', lambda x: '%.14f' % x)
        
    return segments



def union_segments(eeg, s_ce, s_oe):
    """
    Function to combine segments of "Closed Eyes" and "Open Eyes" events into a single list.

    Parameters:
    - eeg (mne.io.Raw): Raw EEG data.
    - s_ce (list): List of segments for "Ce" events.
    - s_oe (list): List of segments for "Oe" events.

    Output:
    - segments (list): Combined list of segments for "Closed Eyes" and "Open Eyes" events, each converted in a DataFrame.
    """
    
    segments = []
    if len(s_ce) == len(s_oe):
        for i in range(len(s_ce)):
            segments.append(s_ce[i])
            segments.append(s_oe[i])

    elif len(s_ce) > len(s_oe):
        for i in range(len(s_oe)):
            segments.append(s_ce[i])
            segments.append(s_oe[i])
        segments.append(s_ce[-1])

    S_df = []
    for i in range(61):
        channel = []
        for segment in segments:
            channel.extend(segment[i])
        #channel = [segment[i] for segment in segments]
        S_df.append(channel)

    df = pd.DataFrame(data=np.transpose(S_df), columns=eeg.ch_names)
    pd.set_option('display.float_format', lambda x: '%.14f' % x)
        
    return df





# =============================================================================================================================================================
# MICROSTATE COMPUTATION, VISUALIZATION & PREPROCESSING

def process_file(filepath, num_clusters):
    """
    Loads a file containing data processed by a clustering model, displays its contents, and plots the results.

    Parameters:
    - filepath (str): The path to the pickle file containing the clustering data.
    - num_clusters (int): The number of clusters used in the computation, which will be printed for reference.

    Output:
    - Displays the contents of the loaded data object if it has a display method.
    """
    
    try:
        # Open the pickle file
        with open(filepath, 'rb') as file:
            ModK = pickle.load(file)
        
        # Assuming data is an instance of ModK
        print(f"\n\nProcessing file: {filepath}")
        print(f"Number of clusters in computation: {num_clusters}")
        display(ModK) # show ModK information
        ModK.plot()  # Plot the data
        # Show the plot
        plt.show() 
    
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")



def gfp_data(eeg):
    """
    Computes the Global Field Power (GFP) from EEG data, identifies GFP peaks and valleys.

    Parameter
    - eeg (mne.Epochs or mne.Evoked): The EEG data structure from which to compute GFP. It should be an instance of mne.Epochs or mne.Evoked.

    Outputs:
    - gfp (ndarray): The computed GFP values.
    - peaks (ndarray): The indices of the peaks in the GFP.
    - valleys (ndarray): The indices of the valleys in the GFP.
    """
    
    # Compute gfp
    gfp = np.std(eeg.get_data(), axis=0)
    
    # Identify gfp peaks
    peaks, _ = find_peaks(gfp)
    print(f"There are {len(peaks)} peaks")
    
    # Identify gfp valleys
    valleys, _ = find_peaks(-gfp)
    print(f"There are {len(valleys)} valleys")

    return gfp, peaks, valleys



def microstate_segmentation(ModK, eeg, peaks, valleys, pairs, subject):
    """
    Segments EEG data into microstates, identifies changes, and writes the results to a CSV file.

    Parameters:
    - ModK (object): The model object used to predict microstates.
    - eeg (mne.Epochs or mne.Evoked): The EEG data structure to segment.
    - pairs (list of tuples): A list of tuples representing the start and end samples.
    - subject (str): The subject identifier used for naming the output CSV file.
    - peaks (ndarray): Indices of the GFP peaks.
    - valleys (ndarray): Indices of the GFP valleys.

    Output:
    - Save the results to a CSV file.
    """

    # Compute the prediction of the microstates
    segmentation = ModK.predict(
        eeg,
        reject_by_annotation=True,
        factor=10,
        half_window_size=10,
        min_segment_length=5,
        reject_edges=True,
    )
    
    # Get the list of microstates. Keep those segmentation values that correspond to the GFP peaks
    microstates = segmentation.labels[peaks]

    # Create a list to store each microstate with the change index, duration, and event
    data = []
    
    # Loop over the microstate list to compute each parameter
    for i in range(0, len(microstates) - 1):
        if microstates[i] != microstates[i + 1]:

            # Determine the microstate label
            if microstates[i] == 0:
                microstate = 'A'
            elif microstates[i] == 1:
                microstate = 'B'
            elif microstates[i] == 2:
                microstate = 'C'
            elif microstates[i] == 3:
                microstate = 'D'
            else:
                microstate = 'E'
    
            # Determine the event corresponding to the current peak
            event = 0
            for j, (start_ce, end_ce) in enumerate(pairs, start=1):
                if start_ce <= peaks[i] <= end_ce:
                    event = j
                    break
         
            data.append([microstate, valleys[i], event])
    
    # Headers for the CSV file
    headers = ['microstate', 'end time', 'event']
    
    # Name of the CSV file
    file_name = f"{subject}_microstate.csv"
    
    # Write data to the CSV file
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        csv_writer.writerow(headers)  # Write headers
        csv_writer.writerows(data)  # Write rows of data
    
    print("CSV file created successfully.")



def format_dataframe(df):
    """
    Formats a DataFrame to prevent scientific notation and ensure up to 14 decimal places for float values.

    This function sets the display precision for pandas DataFrame to 14 decimal places and then formats each float value in the DataFrame to ensure it has up to 14 decimal places without scientific notation.

    Parameter:
    - df (pd.DataFrame): The input DataFrame to format.

    Output:
    - pd.DataFrame: The formatted DataFrame with float values represented up to 14 decimal places.
    """
    # Set the display precision to 14 decimal places
    pd.set_option('display.float_format', lambda x: f'{x:.14f}')
    
    # Apply formatting to each element in the DataFrame
    formatted_df = df.applymap(lambda x: f'{x:.14f}' if isinstance(x, float) else x)
    
    return formatted_df



def process_microstates_data(source_path, target_path, segment_number):
    """
    Process microstates data from the specified folder and segment number, calculate means for each file, and save the result as a CSV.

    Parameters:
    - source_path (str): The base path of the folder containing the microstates data.
    - target_path (str): The path where the processed CSV file will be saved.
    - segment_number (int): The segment number to process.

    Output:
    - Saves the processed data as a CSV file in the specified folder.
    """
    
    # Construct the full folder path with segment number
    full_folder_path = os.path.join(source_path, f'segment_{segment_number}')

    # Dictionary to store mean values for each file
    mean_values = {}

    # Iterate over files in the folder
    for file_name in os.listdir(full_folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(full_folder_path, file_name)

            # Extract patient ID and data type from file name
            patient_id, m1, m2 = file_name.split('_')[0], file_name.split('_')[1].split('-')[0], file_name.split('_')[1].split('-')[1]
            index_name = f"{patient_id}_{m1}-{m2}"
            
            # Read the file and compute mean
            with open(file_path, 'r') as f:
                lines = f.readlines()
                data = [line.strip().split() for line in lines]
                df = pd.DataFrame(data, dtype=float)
                mean_values[index_name] = df.mean()

    # Create a DataFrame from the mean values
    mean_df = pd.DataFrame(mean_values)

    # Transpose the DataFrame
    mean_df = mean_df.transpose()

    # Extract the desired column names
    mean_df.columns = ['ROI_{}'.format(i) for i in range(1, len(mean_df.columns) + 1)]  # Column names as ROI1, ROI2, ...

    # Sort the DataFrame by patient ID
    mean_df = mean_df.sort_index()

    # Check if the target directory exists, if not, create it
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Directory {target_path} created.")
    else:
        print(f"Directory {target_path} already exists.")

    # Save the DataFrame as a CSV file
    csv_path = os.path.join(target_path, f'segment_{segment_number}.csv')
    mean_df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")



def process_segments(base_path, output_path):
    """
    Process microstate data segments, unify odd and even segments into separate files, and create new columns.

    Parameters:
    - base_path (str): The path to the directory containing the segment files.
    - output_path (str): The path to the directory where the combined files will be saved.

    Outputs: Saves two CSV files in the specified output directory:
    - odd_segments_combined.csv: Contains data from all odd-numbered segments.
    - even_segments_combined.csv: Contains data from all even-numbered segments.
    """
    
    # Lists to hold data from closed eyes and open eyes segments
    segments_ce = []
    segments_oe = []

    # Iterate over segment numbers from 1 to 16
    for i in range(1, 17):
        file_name = f'segment_{i}.csv'
        file_path = os.path.join(base_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Extract patient ID and microstate transitions from 'Unnamed: 0'
            df[['PatientID', 'MicrostateTransition']] = df['Unnamed: 0'].str.split('_', expand=True)
            df[['MicrostateFrom', 'MicrostateTo']] = df['MicrostateTransition'].str.split('-', expand=True)

             # Reorder columns to place the new ones at the beginning
            new_columns = ['PatientID', 'MicrostateTransition', 'MicrostateFrom', 'MicrostateTo']
            df = df[new_columns + [col for col in df.columns if col not in new_columns]]
            
            # Drop the 'Unnamed: 0' column
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
            # Append to the appropriate list based on segment number
            if i % 2 == 0:
                segments_oe.append(df)
            else:
                segments_ce.append(df)
        else:
            print(f"File {file_name} does not exist.")
    
    # Concatenate all odd segments which corresponds to those with closed eyes and save to a single file
    if segments_ce:
        combined_ce = pd.concat(segments_ce, ignore_index=True)
        combined_ce.to_csv(os.path.join(output_path, 'segments_ce.csv'), index=False)
        print("Closed eyes segments combined and saved.")
    else:
        print("No closed eyes segments found to combine.")

    # Concatenate all even segments which corresponds to those with open eyes and save to a single file
    if segments_oe:
        combined_oe = pd.concat(segments_oe, ignore_index=True)
        combined_oe.to_csv(os.path.join(output_path, 'segments_oe.csv'), index=False)
        print("Open eyes segments combined and saved.")
    else:
        print("No open eyes segments found to combine.")





# =============================================================================================================================================================
# STATISTICAL ANALYSIS

def microstate_filtering(df, m, c):
    """
    Filters a DataFrame based on a specified microstate or transition and extracts ROI columns.

    Parameters:
    - df (DataFrame): Contains microstate or transition data for each ROI.
    - m (str): The specific microstate or transition to filter by.
    - c (str): The column name to filter on. Valid choices are "MicrostateTransition", "MicrostateFrom", and "MicrostateTo".

    Output:
    - roi_data (DataFrame): A DataFrame containing the ROI columns filtered by the specified microstate or transition. 
    Returns None if the specified column does not exist in the DataFrame.
    """

    # Check if the column exists
    if c not in df.columns:
        print(f'You have introduced "{c}" as desired microstate or transition. However, the only valid choices are: "MicrostateTransition", "MicrostateFrom", and "MicrostateTo".')
        return None

    # Filter the dataset with the respective microstate or transition
    filtered_df = df[df[c] == m]

    # Extract ROI columns
    roi_columns = [f'ROI_{i}' for i in range(1, 211)]
    roi_data = filtered_df[roi_columns]

    return roi_data



def roi_means(df, n):
    """
    Calculates the mean of each column in a DataFrame and output the column(s) with the highest mean(s).

    Parameters:
    - df (DataFrame): The input DataFrame containing ROI data.
    - n (int): The number of top columns to return based on their mean values. 
         - If n = 1, the function returns the column with the highest mean and its value.
         - If n > 1, the function returns the top n columns with the highest means.

    Output:
    - top (tuple or Series): 
         - If n = 1, returns a tuple (max_mean_column, max_mean_value).
         - If n > 1, returns a pandas.Series containing the top n columns and their mean values.
    """

    # Calculate the mean of each column
    column_means = df.mean()

    if n == 1:  # Identify the column with the highest mean
        max_mean_column = column_means.idxmax()
        max_mean_value = column_means.max()
        return max_mean_column, max_mean_value

    else:  # Identify the top n columns with the highest means
        top = column_means.nlargest(n)
        return top



def print_top_features(*feature_dicts, microstate_name, comparison_type=None):
    """
    Print top features and their mean values for a variable number of datasets with alignment.

    Parameters:
    - *feature_dicts (dict): Variable number of dictionaries where keys are feature names and values are mean values for each dataset.
    - microstate_name (str): The name of the microstate or segment for which the features are being compared.
    - comparison_type (str): Type of comparison: 'segment' or 'microstate'.

    Output:
    - Prints a table of the top features with the highest mean values from the datasets, aligned for easy comparison.
    """

    num_dicts = len(feature_dicts)

    if num_dicts not in [2, 6]:
        print("The number of dictionaries should be 2 or 6.")
        return

    # Print header titles based on comparison type and number of dictionaries
    if num_dicts == 2:
        if comparison_type == None:
            comparison_type = 'segment'
        
        print(f"\nTop features with the highest means for Microstate {microstate_name}:\n")
        
        if comparison_type == 'segment':
            print(f"{'':<35} {'Closed Eyes VS Open Eyes'}")
            print()
            print(f"{'Rank':<10} {'Closed Eyes':<19} {'Mean Value (CE)':<29} {'Open Eyes':<19} {'Mean Value (OE)'}")
            
        elif comparison_type == 'microstate':
            print(f"{'':<27} {'Startpoint - Deactivation VS Endpoint - Activation'}")
            print()
            print(f"{'Rank':<10} {'Startpoint':<19} {'Mean Value (S)':<29} {'Endpoint':<19} {'Mean Value (E)'}")
            
        print("=" * 50 * num_dicts)
    
    elif num_dicts == 6:
        print(f"\nTop features with the highest means for Transitions of {microstate_name}:\n")
        print(f"{'':<36} {'Closed Eyes':<77} {'Open Eyes':<25}")
        
        if microstate_name == 'A':
            print(f"{'Rank':<10} {'A-B':<24} {'A-C':<24} {'A-D':<24} {'A-B':<24} {'A-C':<24} {'A-D':<25}")
        
        elif microstate_name == 'B':
            print(f"{'Rank':<10} {'B-A':<24} {'B-C':<24} {'B-D':<24} {'B-A':<24} {'B-C':<24} {'B-D':<25}")

        elif microstate_name == 'C':
            print(f"{'Rank':<10} {'C-A':<24} {'C-B':<24} {'C-D':<24} {'C-A':<24} {'C-B':<24} {'C-D':<25}")

        elif microstate_name == 'D':
            print(f"{'Rank':<10} {'D-A':<24} {'D-B':<24} {'D-C':<24} {'D-A':<24} {'D-B':<24} {'D-C':<25}")

        print("=" * 25 * num_dicts)

    
    
    # Sort each dictionary by mean values
    sorted_features = [dict(sorted(d.items(), key=lambda item: item[1], reverse=True)) for d in feature_dicts]

    # Determine the number of features to display
    top_features = min(len(d) for d in sorted_features)

    for i in range(top_features):
        row = [f"{i + 1:<10} "]
        for d in sorted_features:
            roi = list(d.keys())[i]
            mean_value = d[roi]
            if num_dicts == 2:
                row.extend([f"{roi:<20}", f"{mean_value:<30.4f}"])
            else:            
                row.extend([f"{roi:<10}", f"{mean_value:<15.4f}"])

        print("".join(row))



def zscore_outliers(roi_stats, threshold=3):
    """
    Detect high and low outliers based on Z-scores.
    
    Parameters:
    - roi_stats (pd.Series or pd.DataFrame): Contains the stats with respect the outliers are computed.
    - threshold (int): Z-score threshold for detecting outliers (default is 3)
    
    Outputs:
    - outliers_high (dict): Dictionary containing the high outliers with ROI names as keys and their values as values.
    - outliers_low (dict): Dictionary containing the low outliers with ROI names as keys and their values as values.
    """
    
    # Compute the Z-scores of the ROI statistics
    z_scores = (roi_stats - roi_stats.mean()) / roi_stats.std()
    
    # Identify high and low outliers based on the Z-score threshold
    outliers_high = roi_stats[z_scores > threshold].sort_values(ascending=False).to_dict()
    outliers_low = roi_stats[z_scores < -threshold].sort_values().to_dict()
    
    return outliers_high, outliers_low



def iqr_outliers(roi_stats, threshold=1.5):
    """
    Detect high and low outliers based on the Interquartile Range (IQR).
    
    Parameters:
    - roi_stats (pd.Series or pd.DataFrame): Contains the stats with respect the outliers are computed.
    - threshold: IQR multiplier for detecting outliers (default is 1.5)
    
    Outputs:
    - outliers_high (dict): Dictionary containing the high outliers with ROI names as keys and their values as values.
    - outliers_low (dict): Dictionary containing the low outliers with ROI names as keys and their values as values.
    """
    # Calculate the interquartile range (IQR)
    q1 = roi_stats.quantile(0.25)
    q3 = roi_stats.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate IQR boundaries
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    
    # Identify high and low outliers based on the IQR threshold
    outliers_high = roi_stats[roi_stats > upper_bound].sort_values(ascending=False).to_dict()
    outliers_low = roi_stats[roi_stats < lower_bound].sort_values().to_dict()
    
    return outliers_high, outliers_low



def find_outlier_rois(df, threshold=1.5, method='mean', method_type='iqr'):
    """
    Identify outliers in ROI columns based on the selected method (Z-score or IQR) and return them as dictionaries.

    Parameters:
    - df (DataFrame): Input DataFrame containing ROI columns.
    - threshold (float): The threshold for detecting outliers. For Z-scores, it's the number of standard deviations.
      For IQR, it's the multiplier for the interquartile range.
    - method (str): The method for calculating ROI statistics ('mean', 'median', or 'std').
    - method_type (str): The method for detecting outliers ('zscore' or 'iqr').

    Outputs: Returns the ouputs given by the respective function given by the chosen method:
    - outliers_high (dict): Dictionary containing the high outliers with ROI names as keys and their values as values.
    - outliers_low (dict): Dictionary containing the low outliers with ROI names as keys and their values as values.
    """
    
    # Extract ROI columns (assuming ROI columns start with 'ROI_')
    roi_columns = [col for col in df.columns if col.startswith('ROI_')]

    # Calculate the statistic (mean by default) for each ROI across all rows
    if method == 'mean':
        roi_stats = df[roi_columns].mean()
    elif method == 'median':
        roi_stats = df[roi_columns].median()
    elif method == 'std':
        roi_stats = df[roi_columns].std()
    else:
        raise ValueError("Method must be 'mean', 'median', or 'std'.")

    # Choose the method with which the outliers are computed
    if method_type == 'zscore':
        return u.zscore_outliers(roi_stats, threshold=threshold)

    elif method_type == 'iqr':
        return u.iqr_outliers(roi_stats, threshold=threshold)
    
    else:
        raise ValueError("method_type must be 'zscore' or 'iqr'.")



def print_outliers(*outliers_dicts, segment=None, m=None, o=None, scores_=False):
    """
    Prints dictionaries of outliers side by side as columns. For 2 dictionaries, it prints outlier names and scores.
    For more than 2 dictionaries (specifically 4 or 12), it can print just the names or names with scores based on the `show_scores` flag.

    Parameters:
    - *outliers_dicts (dict): Variable number of dictionaries where the keys are the outlier names and the values are the scores.
    - segment (str, optional): A segment label used in headers when the dictionaries are passed.
    - m (str, optional): A general label used in headers when 2 dictionaries are passed.
    - show_scores (bool, optional): If True, print the outliers with scores for more than 2 dictionaries. Defaults to False.

    Output:
    - Prints the outliers and their values in columns based on the number of dictionaries.
    """

    # Number of dictionaries provided
    num_dicts = len(outliers_dicts)
    
    # Validate that the number of dictionaries is within the supported range
    if num_dicts not in [2, 4, 12]:
        raise ValueError("The function only supports 2, 4, or 12 dictionaries.")
    
    # Convert dictionaries to lists of tuples (sorted by value)
    outliers_with_scores = [list(d.items()) for d in outliers_dicts]
    
    # Find the maximum length of all the lists
    max_length = max(len(ows) for ows in outliers_with_scores)
    
    # Extend all lists to the same length by padding with empty values
    for ows in outliers_with_scores:
        ows.extend([('', '')] * (max_length - len(ows)))

    
    # Case 1: Handling 2 dictionaries (High Outliers for {m})
    if num_dicts == 2:
        print(f"{' ':<21} {o + ' Outliers for ' + m}:\n")
        print(f"{'Closed Eyes':<15} {'Score':<24} {'Open Eyes':<15} {'Score':<25}")
        print("=" * (32 * num_dicts))
        
        # In case 1 the scores are always shown
        scores_ = True
        
        # Print each element in the lists side by side with their scores
        for row in zip(*outliers_with_scores):
            row_str = ""
            for outlier, score in row:
                if score == '':
                    row_str += f"{outlier:<15} {score:<25}"
                else:
                    row_str += f"{outlier:<15} {score:<25.4f}"
            print(row_str)

    
    # Case 2: Handling 4 dictionaries (Microstates A, B, C, D) or 12 dictionaries (Transitions)
    elif num_dicts in [4, 12]:
        # Predefined transitions for headers (only for 12 dictionaries)
        transitions = ['A-B', 'A-C', 'A-D', 'B-A', 'B-C', 'B-D', 'C-A', 'C-B', 'C-D', 'D-A', 'D-B', 'D-C']

        # Handle headers for 4 dictionaries (Microstates)
        if num_dicts == 4:
            print(f"High Outliers for {segment}:\n")
            if scores_:
                headers = [f"{'A':<26}", f"{'B':<26}", f"{'C':<26}", f"{'D':<26}"]
                print("".join(headers))
                print("=" * (25 * num_dicts))
            else:
                headers = [f"{'A':<13}", f"{'B':<13}", f"{'C':<13}", f"{'D':<13}"]
                print("".join(headers))
                print("=" * (12 * num_dicts))
        
        # Handle headers for 12 dictionaries (Transitions)
        elif num_dicts == 12:
            print(f"{o} Outliers for {segment}:\n")
            headers = [f"{transitions[i]:<13}" for i in range(num_dicts)]
            print("".join(headers))
            print("=" * (12 * num_dicts + 5))

        # Print each element in the lists side by side
        for row in zip(*outliers_with_scores):
            row_str = ""
            for outlier, score in row:
                if scores_:
                    if score != '':
                        row_str += f"{outlier:<10} {score:<15.4f}"
                    else:
                        row_str += f"{outlier:<26}"
                else:
                    row_str += f"{outlier:<13}"
            print(row_str)



def roi_std_comparison(*dfs):
    """
    Compares the standard deviation between corresponding ROIs from multiple DataFrames.
    
    Parameters:
    - dfs (list): List of DataFrames containing ROI values. All DataFrames must have the same columns.
    
    Output:
    - std_comparison (dict): Contains the standard deviations for each ROI in all DataFrames, as well as the absolute difference in standard deviation between each pair of DataFrames.
    """

    diff_dict = {}
    
    # Ensure all dataframes have the same columns
    if not all(dfs[0].columns.equals(df.columns) for df in dfs):
        raise ValueError("All DataFrames must have the same columns (ROIs).")
    
    # Initialize dictionary to store results
    std_comparison = {}
    
    # Get the names for the dataframes to use as keys
    if len(dfs) == 4:
        microstates = ['A', 'B', 'C', 'D']
        dfs_names = [m for m in microstates]
        
    elif len(dfs) == 12:
        transitions = ['A-B', 'A-C', 'A-D', 'B-A', 'B-C', 'B-D', 'C-A', 'C-B', 'C-D', 'D-A', 'D-B', 'D-C']
        dfs_names = [t for t in transitions]

    else:
        dataframe_names = [f"df_{i}" for i in range(len(dataframes))]  # e.g., df_0, df_1, ...
    
    # Loop over each ROI (column)
    for roi in dfs[0].columns:
        # Calculate standard deviations for the ROI in each dataframe
        stds = [df[roi].std() for df in dfs]
        
        # Store the standard deviations
        std_comparison[roi] = {name: std for name, std in zip(dfs_names, stds)}
        
        # Calculate the absolute differences between each pair of dataframes
        for (i, j) in combinations(range(len(dfs)), 2):
            key = f"{dfs_names[i]}-{dfs_names[j]}"
            std_comparison[roi][key] = abs(stds[i] - stds[j])

    # Loop through each ROI in the std_dict
    for roi, diff_data in std_comparison.items():
        
        # Extract the difference pairs and use them as keys
        pairs = [key for key in diff_data if '-' in key]  # e.g., 'A-B', 'A-C', etc.
        
        # Extract the differences for the current ROI
        diffs = [diff_data[pair] for pair in pairs]
        
        # Store the differences in the dictionary, with ROI as the key
        diff_dict[roi] = diffs
    
    # Convert the dictionary into a DataFrame. Rows will be the pairs ('A-B', 'A-C', etc.), and columns will be ROIs
    diff_df = pd.DataFrame(diff_dict, index=pairs)
    
    # Obtain the std for each microstate
    std_m = {roi: {key: value for key, value in sub_dict.items() if key not in pairs} for roi, sub_dict in std_comparison.items()}

    return diff_df, std_m



def rois_std(df, df_std, std_m, column, num_rois=1):
    """
    Finds the top N maximum values for standard deviation and corresponding ROIs for each row in the DataFrame with the corresponding activation.

    Parameters:
    - df (DataFrame): Contains microstate or transition data for each ROI.
    - df_std (DataFrame): Contains the ROI names as columns and the microstate transitions as rows.
    - std_m (dict): Dictionary with ROI standard deviations for each microstate transition.
    - column (str): The column name to filter on. Valid choices are "MicrostateTransition", "MicrostateFrom", and "MicrostateTo".
    - num_rois (int, optional): Number of top ROIs to return per row.

    Output:
    - max_roi_dict (dict): A dictionary where the keys are the row indices and the values are lists of tuples for the top N ROIs.
    """
    
    max_roi_dict = {}

    # Iterate over each row in the DataFrame
    for idx, (index, row) in enumerate(df_std.iterrows()):

        # Obtain the respective microstates that are being dealt with
        m1 = index[0]
        m2 = index[-1]
        
        # Filter the dataframe with respect to the microstates that are being dealt with
        df_m1 = u.microstate_filtering(df, m1, column)
        df_m2 = u.microstate_filtering(df, m2, column)
        
        if num_rois == 1:
            # Find the maximum value in the row
            max_value = row.max()
    
            # Find the corresponding ROI (column name) for the maximum value
            roi = row.idxmax()
            
            max_roi_dict[index] = (
                                   {'ROI': roi},
                                   {'Difference': max_value},
                                   {m1: {'activation': df_m1[roi].mean(), 'std': std_m[roi][m1]}},
                                   {m2: {'activation': df_m2[roi].mean(), 'std': std_m[roi][m2]}}
                                  )
            
        else:
            # Get the top N ROIs and values
            top_rois = row.nlargest(num_rois)
            
            # Create a list to store the top ROI data for the current row
            roi_data = []
        
            for roi, max_value in top_rois.items():

                roi_data.append((
                    {'ROI': roi}, 
                    {'Difference': max_value}, 
                    {m1: {'activation': df_m1[roi].mean(), 'std': std_m[roi][m1]}},
                    {m2: {'activation': df_m2[roi].mean(), 'std': std_m[roi][m2]}}
                ))

            # Store the ROI data in the dictionary
            max_roi_dict[index] = roi_data

    return max_roi_dict



def print_info(comparison_dict, outliers_dict=None, outlier_type=None):
    """
    Prints a summary of the comparison data from the given dictionary in a tabular format, with ROI names in bold if they are outliers. 
    Additionally, any ROI outliers from outliers_dict that are not present in comparison_dict will be added to the table.

    Parameters:
    - comparison_dict (dict): A dictionary where keys are comparison labels and values are lists of tuples or single tuples containing details for each comparison.
    - outliers_dict (dict): A dictionary of outliers where keys are comparison labels, and values are tuples of lists of ROI names.
    - outlier_type (str): The type of outliers which are selected to be shown: high, low
    
    Output:
    - Prints the summary of comparisons in a tabular format with bolded ROI names if they are outliers.
    """
    
    # ANSI escape codes for bold text
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Print the table header
    print(f"{'Comparison':<15} {'ROI':<24} {'Activations':<28} {'STDs':<19} {'Difference':<15}")
    print("="*100)
    
    # Iterate over each comparison in the dictionary
    for comparison, values in comparison_dict.items():
        # Print the comparison label only once at the beginning of each section
        first_item = True

        # Retrieve the appropriate outliers for this comparison if outliers_dict is provided
        if outliers_dict is not None:
            # Based on the selected outlier_type, either 'high' or 'low', retrieve the respective outliers
            if outlier_type == 'high':
                outliers, _ = outliers_dict.get(comparison, ([], []))
            elif outlier_type == 'low':
                _, outliers = outliers_dict.get(comparison, ([], []))
        
        # Initialize a set to track which ROIs have been printed to avoid duplication
        printed_rois = set()

        # Check if the comparison data contains a list of tuples (multiple comparisons) or a single tuple
        if isinstance(values, list):
            for item in values:
                roi = item[0]['ROI']
                diff = item[1]['Difference']
                m1 = comparison[0]
                m2 = comparison[-1]
                activation_1 = item[2][m1]['activation']
                std_1 = item[2][m1]['std']
                activation_2 = item[3][m2]['activation']
                std_2 = item[3][m2]['std']

                # Add the ROI to the printed set
                printed_rois.add(roi)

                # Check if the ROI should be highlighted as an outlier
                if outliers_dict is None:
                    roi_display = roi
                else:
                    # If the ROI is an outlier, display it in bold
                    if roi in outliers:
                        roi_bold = f"{BOLD}{roi}{RESET}"
                        roi_display = roi_bold.ljust(19 + len(BOLD) + len(RESET))
                    else:
                        roi_display = roi.ljust(15)

                # Print the row, with the comparison label on the first row of each section
                if first_item:
                    print(f"{comparison:<15} {roi_display:<19} {activation_1:<10.4f} {activation_2:<15.4f} {std_1:<10.4f} {std_2:<15.4f} {diff:<15.4f}")
                    first_item = False
                else:
                    print(f"{'':<15} {roi_display:<19} {activation_1:<10.4f} {activation_2:<15.4f} {std_1:<10.4f} {std_2:<15.4f} {diff:<15.4f}")

        # Handle single comparison case
        else:
            item = values
            roi = item[0]['ROI']
            diff = item[1]['Difference']
            m1 = comparison[0]
            m2 = comparison[-1]
            activation_1 = item[2][m1]['activation']
            std_1 = item[2][m1]['std']
            activation_2 = item[3][m2]['activation']
            std_2 = item[3][m2]['std']

            # Add the ROI to the printed set
            printed_rois.add(roi)

            # Check if the ROI should be highlighted as an outlier
            if outliers_dict is None:
                roi_display = roi
            else:
                # If the ROI is an outlier, display it in bold
                if roi in outliers:
                    roi_bold = f"{BOLD}{roi}{RESET}"
                    roi_display = roi_bold.ljust(19 + len(BOLD) + len(RESET))
                else:
                    roi_display = roi.ljust(15)

            # Print the row for the single comparison
            print(f"{comparison:<15} {roi_display:<19} {activation_1:<10.4f} {activation_2:<15.4f} {std_1:<10.4f} {std_2:<15.4f} {diff:<15.4f}")

        # Check for and print any additional outliers that were not in the comparison data
        if outliers_dict is not None:
            additional_outliers = (set(outliers) - printed_rois)
            for roi in additional_outliers:
                roi_bold = f"{BOLD}{roi}{RESET}"
                roi_display = roi_bold.ljust(19 + len(BOLD) + len(RESET))
                # Print the outlier row with 'N/A' for activations, STDs, and differences
                print(f"{'':<15} {roi_display:<19} {'N/A':<10} {'N/A':<15} {'N/A':<10} {'N/A':<15} {'N/A':<15}")
        
        # Print a blank line after each comparison section for readability
        print()



def find_outlier_std_comparison(std_df, method='iqr', threshold=1.5):
    """
    Finds outliers in the standard deviation differences between pairs of DataFrames for each ROI.

    Parameters:
    - std_df (Dataframe): Contains the standard deviation differences for each ROI.
    - method (str): Method for detecting outliers, either 'zscore' or 'iqr'. Default is 'zscore'.
    - threshold (float): Threshold for identifying outliers. 
                         - For 'zscore', this represents the number of standard deviations from the mean. 
                         - For 'iqr', this represents the multiplier for the IQR method.

    Output:
    - outliers_dict (dict): Dictionary containing the detected outliers for each ROI.
    """

    outliers_dict = {}

    # Loop through each pair
    for pair in list(std_df.index):
        # Detect outliers based on the selected method
        if method == 'zscore':
            # Calculate the mean and standard deviation
            high, low = u.zscore_outliers(std_df.loc[pair], threshold=threshold)

        elif method == 'iqr':
            # Calculate the IQR (Interquartile Range)
            high, low = u.iqr_outliers(std_df.loc[pair], threshold=threshold)

        else:
            raise ValueError("Method must be 'zscore' or 'iqr'.")

        # Store the ROI names that are outliers
        high_outliers = list(high.keys())
        low_outliers = list(low.keys())

        # Only add to the dictionary if there are outliers
        if high_outliers or low_outliers:
            outliers_dict[pair] = (high_outliers, low_outliers)

    return outliers_dict



def print_outlier_std_comparison(outlier_dict):
    """
    Print outlier comparisons in a tabular format.

    Parameters:
    - outlier_dict (dict): Dictionary containing outlier comparisons and differences for each ROI.

    Output:
    - Prints the ROI names, their corresponding microstate comparisons, and the high differences side by side in a tabular format.
    """
    
    # Print header
    print(f"{'ROI':<15}{'Comparisons':<30}{'Differences':<15}")
    print("="*60)

    # Iterate over the dictionary to extract and print the values
    for comparison, (high_outliers, low_outliers) in outlier_dict.items():
        # Process high outliers
        for roi, diff in high_outliers.items():
            print(f"{roi:<15}{comparison:<30}{diff:<15.4f}")



def numerical_normality_test(x, return_=False):
    """
    Performs a normality test on the given dataset `x` to assess whether it follows a normal distribution. 
    The function automatically selects the appropriate test based on the sample size.

    Parameters:
    - x (array or DataFrame): The dataset to be tested for normality. 
      If the sample size is greater than 50, the Kolmogorov-Smirnov test is used; otherwise, the Shapiro-Wilk test is applied.
    - return_ (bool): If True, returns the test name, test statistic, and p-value; otherwise, results are printed.

    Outputs:
    - If `return_` is True, returns a tuple containing:
        - test (str): The name of the test performed.
        - statistic (float): The test statistic from the normality test.
        - p_value (float): The p-value from the test.
    - If `return_` is False, prints the p-value.
    """
    
    # Check if the dataset size is greater than 50 for using KS test,
    # otherwise use Shapiro-Wilk test
    if x.shape[0] > 50:
        # Perform Kolmogorov-Smirnov test
        test = 'Kolmogorov-Smirnov'
        statistic, p_value = kstest(x, 'norm', args=(x.mean(), x.std()))
        #print('Kolmogorov-Smirnov statistic:', ks_statistic)
        #print('p-value:', p_value)
    else:
        # Perform Shapiro-Wilk test
        test = 'Shapiro-Wilk'
        statistic, p_value = shapiro(x)
        #print('Shapiro-Wilk Statistic:', sw_statistic)
        print('p-value:', p_value)
        
    # Interpret the p-value: if it's greater than 0.05, the data is likely normal
    """if p_value > 0.05:
        print('The data follows a normal distribution.')
    else:
        print('The data does not follow a normal distribution.')"""

    if return_:
        return test, statistic, p_value



def print_normality_results(normality_results_ce, normality_results_oe, microstate_name):
    """
    Print normality test results for features in two datasets with alignment.

    Parameters:
    - normality_results_ce (dict): Dictionary containing normality test results for the "Closed Eyes" dataset.
      Each entry is a tuple (test_name, statistic_value, p_value).
    - normality_results_oe (dict): Dictionary containing normality test results for the "Open Eyes" dataset.
      Each entry is a tuple (test_name, statistic_value, p_value).
    - microstate_name (str): Name of the microstate for which the results are being reported (e.g., 'A', 'B', 'C', 'D').

    Output:
    - Prints a table of normality test results with alignment for easy comparison between the two datasets.
    """

    print(f"\nNormality Test Results for Microstate {microstate_name}:\n")
    print(f"{'Feature':<15} {'Test (CE)':<20} {'Statistic (CE)':<10} {'Normality (CE)':<16} {'p-Value (CE)':<10} {'Test (OE)':<20} {'Statistic (OE)':<15} {'Normality (OE)':<15} {'p-Value (OE)'}")

    # Get the set of all features from both datasets
    all_features = set(normality_results_ce.keys()).union(set(normality_results_oe.keys()))
    
    # Sort features based on the numerical part (assuming feature names follow a specific format)
    sorted_features = sorted(all_features, key=lambda x: int(x.split('_')[1]))

    for feature in sorted_features:
        # Normality test results for Closed Eyes
        result_ce = normality_results_ce.get(feature, (None, None, None))
        test_ce, statistic_ce, p_value_ce = result_ce
        normality_ce = 'Normal' if (p_value_ce is not None and p_value_ce > 0.05) else 'Not Normal'
        
        # Normality test results for Open Eyes
        result_oe = normality_results_oe.get(feature, (None, None, None))
        test_oe, statistic_oe, p_value_oe = result_oe
        normality_oe = 'Normal' if (p_value_oe is not None and p_value_oe > 0.05) else 'Not Normal'
        
        # Print aligned results
        print(f"{feature:<15} {test_ce:<20} {statistic_ce:<14.3e} {normality_ce:<16} {p_value_ce:<12.3e} {test_oe:<20} {statistic_oe:<15.3e} {normality_oe:<15} {p_value_oe:<10.3e}")



def roi_comparison_wilcoxon_cVSo(df_ce, df_oe, m=None):
    """
    Compares the ROIs between Closed Eyes (CE) and Open Eyes (OE) for a given microstate using the Wilcoxon signed-rank test. 
    It has also the possibility to compare both entire dataframes.

    Parameters:
    - df_ce (DataFrame): DataFrame containing ROI data for Closed Eyes.
    - df_oe (DataFrame): DataFrame containing ROI data for Open Eyes.
    - m (str, optional): The name of the microstate to specify which dataset to compare. If None, compares all datasets.

    Output:
    - Prints the results of the Wilcoxon signed-rank tests comparing each ROI between CE and OE, including the test statistic, p-value, and significance.
    """
    
    if m is not None:
        print(f"\nStatistical Comparison for Microstate {m}:\n") 
    else:
        print(f"\nStatistical Comparison for all the datasets together")

    print(f"{'Feature':<20} {'Test Used':<30} {'Statistic':<15} {'p-Value':<10} {'Significant Difference?'}")

    # List of all features (ROIs) common to both datasets
    all_features = set(df_ce.columns).intersection(set(df_oe.columns))

    # Sort features based on the numerical part (assuming feature names follow a specific format)
    sorted_features = sorted(all_features, key=lambda x: int(x.split('_')[1]))
    
    for feature in sorted_features:
        # Perform Wilcoxon signed-rank test for the feature data
        statistic, p_value = wilcoxon(df_ce[feature], df_oe[feature])
        test_name = 'Wilcoxon Signed-Rank'

        # Determine if the p-value indicates a significant difference
        significant = 'Yes' if p_value < 0.05 else 'No'
    
        # Print the results in a formatted way
        print(f"{feature:<20} {test_name:<30} {statistic:<15.3f} {p_value:<10.3f} {significant}")



def generate_comparisons(*dfs):
    """
    Generates a list of comparison tuples between the provided DataFrames.

    Parameters:
    - dfs (list): List of DataFrames to compare.
    - labels (list, optional): List of labels for the DataFrames. If None, labels will be generated.

    Output:
    - comparisons (list of tuples): A list of comparisons in the format (label1, label2, group1, group2).
    """
    num_dfs = len(dfs)

    # Default labels based on the number of dataframes
    if num_dfs == 4:
        labels = ['A', 'B', 'C', 'D']
    elif num_dfs == 12:
        labels = ['A-B', 'A-C', 'A-D', 'B-A', 'B-C', 'B-D', 'C-A', 'C-B', 'C-D', 'D-A', 'D-B', 'D-C']
    else:
        labels = [f"Group {i+1}" for i in range(num_dfs)]
    
    # Create pairwise comparisons
    comparisons = [
        (labels[i], labels[j], dfs[i], dfs[j])
        for i, j in combinations(range(num_dfs), 2)
    ]
    
    return comparisons



def microstate_roi_analysis(*dfs, feature):
    """
    Analyzes the ROI feature across multiple microstates (n microstates) using the Friedman test and performs pairwise comparisons if significant.

    Parameters:
    - *dfs (DataFrame): Variable number of DataFrames for microstates.
    - feature (str): The name of the feature to analyze.

    Output:
    - Prints the result of the Friedman test for the feature across the datasets.
    - If the Friedman test is significant, prints pairwise Wilcoxon signed-rank test results with Bonferroni correction.
    """
    
    # Ensure that at least two DataFrames are provided
    num_dfs = len(dfs)
    if num_dfs < 2:
        raise ValueError("At least two DataFrames must be provided for comparison.")
    
    # Perform the Friedman test
    friedman_result = friedmanchisquare(*[df[feature] for df in dfs])

    # Print the result of the Friedman test
    print(f"Friedman test result:")
    print(f"Statistic: {friedman_result.statistic:.4f}, p-value: {friedman_result.pvalue:.4e}\n")

    # If the Friedman test is significant, perform pairwise comparisons
    if friedman_result.pvalue < 0.05:
        # Generate default comparisons if not provided
        comparisons = generate_comparisons(*dfs)

        # Calculate the number of comparisons
        num_tests = len(comparisons)
        
        # Adjust the significance level using Bonferroni correction
        alpha_original = 0.05
        alpha_adjusted = alpha_original / num_tests
        
        # Perform pairwise comparisons using the Wilcoxon signed-rank test
        pairwise_results = []
        for label1, label2, group1, group2 in comparisons:
            result = wilcoxon(group1[feature], group2[feature])
            pairwise_results.append((label1, label2, result.statistic, result.pvalue))
        
        # Print pairwise comparison results with Bonferroni correction
        print(f"\nPairwise Wilcoxon signed-rank test results with Bonferroni correction:\n")
        print(f"{'Comparison':<25} {'Statistic':<14} {'Adjusted p-value':<20} {'Original p-value':<20} {'Significant?'}")
        print("-" * 90)
        
        for label1, label2, stat, pval in pairwise_results:
            adjusted_pval = min(pval * num_tests, 1.0)  # Ensure p-values don't exceed 1.0 after adjustment
            significant = 'Yes' if adjusted_pval < alpha_original else 'No'
            
            print(f"{label1} vs {label2:<18} {stat:<14.3e} {adjusted_pval:<20.3e} {pval:<20.3e} {significant}")
    else:
        print("No significant difference found with Friedman test.\n")





# =============================================================================================================================================================
# CLASSIFIERS ROI PREPROCESSING 

def data_split(df, rs):
    """
    Splits a DataFrame into training and testing sets based on unique patient IDs.

    Parameters:
    - df (DataFrame): The input DataFrame containing patient records.
    - rs (int): The random state seed for reproducibility of the split.

    Outputs:
    - train (DataFrame): The training set containing patient records without the 'PatientID' column.
    - test (DataFrame): The testing set containing patient records without the 'PatientID' column.
    """

    # Get a list of unique patient IDs
    patient_ids = df['PatientID'].unique()
    
    # Split the data between train and test with respect to the patient IDs list
    train_patients, test_patients = train_test_split(patient_ids, test_size=37/187, random_state=rs)
    
    # Select all records for patients in the train and test groups and sort them by patient
    df_train = df[df['PatientID'].isin(train_patients)].sort_values(by='PatientID')
    df_test = df[df['PatientID'].isin(test_patients)].sort_values(by='PatientID')
    
    # Shuffle the records within each patient group
    df_train = df_train.groupby('PatientID').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    df_test = df_test.groupby('PatientID').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    
    # Drop Patient ID column
    df_train = df_train.drop(columns=['PatientID'])
    df_test = df_test.drop(columns=['PatientID'])

    # Check if there's any overlap (patients in both train and test subsets)
    assert len(set(train_patients).intersection(set(test_patients))) == 0

    return df_train, df_test, train_patients, test_patients, rs



def choose_classifier():
    """
    Provides a menu for selecting the type of classifier.
    Specifies which classifier to prepare for:
    - 1: Drops 'MicrostateTransition', encodes 'MicrostateTo' (target variable) and 'MicrostateFrom'.
    - 2: Drops 'MicrostateFrom' and 'MicrostateTo', encodes 'MicrostateTransition' (target variable).

    Parameters:
      None
    
    Output:
    - choice (int): The selected classifier option.
    """
    
    print("\nSelect the classifier type:")
    print("1. Classifier - Microstate Endpoint: Uses 'MicrostateTo' as the target variable.")
    print("2. Classifier - Microstate Transition: Uses 'MicrostateTransition' as the target variable.")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in [1, 2]:
                return choice
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")



def choose_data_format():
    """
    Provides a menu for selecting the type of data format.
    - tp (int): Determines the format of the output:
        - 1: Returns data as DataFrames.
        - 2: Returns data as NumPy arrays.

    Parameters:
      None

    Output:
    - choice (int): The selected data format option.
    """
    
    print("\nSelect the data format:")
    print("1: Returns data as DataFrames.")
    print("2: Returns data as NumPy arrays.")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in [1, 2]:
                return choice
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")



def pre_requirements(df, c=None, rs=None):
    """
    Prepares the DataFrame for training a classifier by dropping columns and encoding target variables.

    Parameters:
    - df (DataFrame): The input DataFrame containing patient records and target variables.

    Outputs: Calls the 'data_split' function, having as outputs the following.
    - train (DataFrame): The training set containing patient records without the 'PatientID' column.
    - test (DataFrame): The testing set containing patient records without the 'PatientID' column.
    """
    
    if c == None:
        classifier = choose_classifier()
    else:
        classifier = c
    
    # Drop unnecessary columns
    if classifier == 1:
        df = df.drop(columns=['MicrostateTransition'])
    elif classifier == 2:
        df = df.drop(columns=['MicrostateFrom', 'MicrostateTo'])
    else: 
        print('Please, choose between 1 and 2')
        return 

    # Encode categorical target variables
    encoder = LabelEncoder()
    if classifier == 1:
        df['MicrostateTo'] = encoder.fit_transform(df['MicrostateTo'])
        df['MicrostateFrom'] = encoder.transform(df['MicrostateFrom'])
    elif classifier == 2:
        df['MicrostateTransition'] = encoder.fit_transform(df['MicrostateTransition'])
    else: 
        print('Please, choose between 1 and 2')

    # Generate a random state for reproducibility if none was introduced as parameter
    if rs == None:
        rs = random.randint(1, 100)
        
    print('Random State: ', rs)
    
    # Split the data into training and testing sets
    return u.data_split(df, rs)



def extraction(df_train, df_test, tp=None, c=None):
    """
    Extracts features and target variables from training and testing datasets based on the specified type and classifier.

    Parameters:
    - df_train (DataFrame): The training dataset, including both features and target variables.
    - df_test (DataFrame): The testing dataset, including both features and target variables.
    
    Outputs:
    - If `tp` is 1:
        - df_X_train (DataFrame): Training feature set.
        - df_y_train (Series): Training target variable.
        - df_X_test (DataFrame): Testing feature set.
        - df_y_test (Series): Testing target variable.
    - If `tp` is 2:
        - X_train (array-like): Training feature set in NumPy array format.
        - y_train (array-like): Training target variable in NumPy array format.
        - X_test (array-like): Testing feature set in NumPy array format.
        - y_test (array-like): Testing target variable in NumPy array format.
    """
    
    if tp == None:
        tp = choose_data_format()
    else:
        tp = tp
    
    if c == None:
        classifier = choose_classifier()
    else:
        classifier = c

    
    # Extract data as DataFrames
    if tp == 1:
        
        if classifier == 1:
            df_X_train, df_y_train = df_train.drop(columns=['MicrostateTo']), df_train['MicrostateTo']
            df_X_test, df_y_test = df_test.drop(columns=['MicrostateTo']), df_test['MicrostateTo']
            
        elif classifier == 2:
            df_X_train, df_y_train = df_train.drop(columns=['MicrostateTransition']), df_train['MicrostateTransition']
            df_X_test, df_y_test = df_test.drop(columns=['MicrostateTransition']), df_test['MicrostateTransition']

        else:
            print('Please, choose between 1 and 2')
            return 
            
        return df_X_train, df_y_train, df_X_test, df_y_test

    
    # Extract data as NumPy arrays
    elif tp == 2:
        
        if classifier == 1:
            X_train, y_train = df_train.drop(columns=['MicrostateTo']).values, df_train['MicrostateTo'].values
            X_test, y_test = df_test.drop(columns=['MicrostateTo']).values, df_test['MicrostateTo'].values
            
        elif classifier == 2:
            X_train, y_train = df_train.drop(columns=['MicrostateTransition']).values, df_train['MicrostateTransition'].values
            X_test, y_test = df_test.drop(columns=['MicrostateTransition']).values, df_test['MicrostateTransition'].values

        else:
            print('Please, choose between 1 and 2')
            return

        return X_train, y_train, X_test, y_test

    else:
        print('Please, choose a type between 1 and 2')



def normalizing(tp, X_train, X_test):
    """
    Normalizes the input datasets using the specified scaler type.

    Parameters:
    - tp (str): Type of scaler to use. Options: "ss" for StandardScaler, "mm" for MinMaxScaler, "rs" for RobustScaler.
    - X_train (array-like): Training feature dataset.
    - X_test (array-like): Testing feature dataset.
    - y_train (array-like): Training target variable.
    - y_test (array-like): Testing target variable.

    Output:
    - X_train_norm (array-like): normalized feature for training.
    - X_test_norm (array-like): normalized feature for test.
    """
    
    if tp == "ss":
        # Dataset Normalization using StandardScaler
        scaler = StandardScaler()
    elif tp == "mm":
        # Dataset Normalization using MinMaxScaler
        scaler = MinMaxScaler()
    elif tp == "rs":
        # Dataset Normalization using RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type. Choose from 'ss', 'mm', or 'rs'.")
    
    if X_train.shape[0] == 211: # Separate the first column and the rest of the columns as we are in the first classifier
        X_train_first_col = X_train[:, 0:1]
        X_train_rest = X_train[:, 1:]
        X_test_first_col = X_test[:, 0:1]
        X_test_rest = X_test[:, 1:]

        # Normalize the rest of the columns
        X_train_rest_norm = scaler.fit_transform(X_train_rest)
        X_test_rest_norm = scaler.transform(X_test_rest)

        # Combine the first column with the normalized rest
        X_train_norm = np.hstack([X_train_first_col, X_train_rest_norm])
        X_test_norm = np.hstack([X_test_first_col, X_test_rest_norm])
        
    else: # If shape[0] is 210 we are in the second classifier, thus normalize the entire dataset
        # Fitting the scaler with the X_train subset and normalizing it
        X_train_norm = scaler.fit_transform(X_train)
        # Normalizing the X_test subset with respect to the values taken from X_train, as the scaler was trained with it
        X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_test_norm
    




# =============================================================================================================================================================
# FEATURE SELECTION

def log_reg(X_train, y_train, X_test, feature=None, penalty=None, param_grid=None):
    """
    Perform logistic regression with optional regularization using L1, L2, or ElasticNet penalties.

    Parameters:
    - X_train (array-like): Training feature dataset.
    - y_train (array-like): Training target variable.
    - X_test (array-like): Testing feature dataset.
    - feature (str): If specified, train and predict using only the selected feature.
    - penalty (str): If specified, apply regularization with options 'l1' for L1 penalty, 'l2' for L2 penalty, or 'elasticnet' for ElasticNet penalty.
    - param_grid (dict): If specified, perform grid search with cross-validation using the given parameter grid.

    Output: Depending on the provided parameters the following will be returned.
    - If feature is None and param_grid is None: Tuple containing the intercept, coefficients, predicted labels, and predicted probabilities.
    - If feature is None and param_grid is specified: Tuple containing the best parameters, best score, predicted labels, and predicted probabilities.
    - If feature is specified: Tuple containing the intercept, coefficient, predicted labels, and predicted probabilities for the selected feature.
    """

    # Logistic regression with optional regularization
    if penalty == None:
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    elif penalty == 'l1':
        model = LogisticRegression(solver='saga', penalty='l1', multi_class='multinomial')
    elif penalty == 'l2':
        model = LogisticRegression(solver='saga', penalty='l2', multi_class='multinomial')
    elif penalty == 'elasticnet':
        model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5, multi_class='multinomial')
    else:
        print('Please choose an accepeted penalty [l1, l2 or elasticnet].')
    
    if feature == None:
        if param_grid == None:
            # Train the model using X_train as the input
            model = model.fit(X_train, y_train)

            # Predict the values using the test set. Obtain both the hard and the soft output
            y_pred = model.predict(X_test)
            y_prob_pred = model.predict_proba(X_test)
            
            # Show the intercept
            intercept = model.intercept_
            #print(intercept)

            # Show the coefficients
            coefficients = model.coef_
            #print(coefficients)

            return coefficients[0], intercept, y_pred, y_prob_pred
            
        else:
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

            # Fit the grid search to your training data
            grid_search.fit(X_train, y_train)  # Replace X_train and y_train with your training data
            
            # Predict using the model with the best parameters
            y_pred = grid_search.best_estimator_.predict(X_test)
            y_prob_pred = grid_search.best_estimator_.predict_proba(X_test)

            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            return best_params, best_score, y_pred, y_prob_pred
            
    else:
        # Train the model using X_train using only the feature that best estimates the target variable as the input
        model = model.fit(np.array(X_train[feature]).reshape(-1, 1), y_train)
        
        # Predict the values using the test set. Obtain both the hard and the soft output
        y_pred = model.predict(np.array(X_test[feature]).reshape(-1,1))
        y_prob_pred = model.predict_proba(np.array(X_test[feature]).reshape(-1,1))
        
        # Show the intercept
        intercept = model.intercept_
        #print(intercept)

        # Show the coefficients
        coefficients = model.coef_
        #print(coefficients)

        return coefficients[0][0], intercept[0], y_pred, y_prob_pred



def wrappers(model, tp, n, X_train, y_train, features):
    """
    Performs the chosen wrapper method between Recursive Feature Elimination (RFE), Forward Selection (FS) or Backward Elimination (BE) to select the top n features using the provided model.
    
    Parameters:
    - model: The machine learning model used for feature selection (e.g., LogisticRegression, Tree Classifiers, Random Forest, Gradient Boosting, XGBoost...).
    - tp (str): The corresponding wrapper method (Recursive Feature Elimination, Forward Selection or Backward Elimination).
    - n (int): The number of top features to select.
    - X_train (DataFrame or array-like): The training dataset features.
    - y_train (Series or array-like): The target labels corresponding to X_train.
    - features (list): List or array of feature names corresponding to the columns of X_train.
    
    Output:
    - selected_features (list): List of the top n selected features.
    """
    
    # Create and fit the RFE model with the corresponding classifier
    if tp == 'RFE':
        wrapper = RFE(model, n_features_to_select=n)
        
    elif tp == 'FS':
        wrapper = SequentialFeatureSelector(model, n_features_to_select=n, direction='forward')

    elif tp == 'BE':
        wrapper = SequentialFeatureSelector(model, n_features_to_select=n, direction='backward')

    else:
        print('Please choose a correct wrapper method between Recursive Feature Elimination (RFE), Forward Selection (FS) or Backward Elimination (BE).')
        
    fit = wrapper.fit(X_train, y_train)
    
    # Take the selected features and their corresponding ROIs
    selected_features = list(features[fit.support_])

    return selected_features



def wrappers_print(selected_features, feature_positions, importances_sorted):
    """
    Print the selected features from the chosen wrapper method with their importance and position, sorted by importance.
    Filter and print features from the list with their corresponding importance and position

    Parameters:
    - selected_features (list): List of features selected by the wrapper method.
    - feature_positions (dict): Dictionary mapping features to their position in the sorted importance dictionary.
    - importances_sorted (dict): Dictionary of features sorted by their importance or coefficient values.
    
    Output:
    - Prints the summary of the wrapper coefficients foe each feature by their respective position.
    """
    
    # Filter and sort features from the list by their position in the sorted dictionary
    sorted_features = sorted(selected_features, key=lambda x: feature_positions.get(x, float('inf')))
    
    # Filter and print features from the list with their corresponding importance and position
    print("Features from the list with their importance and position (sorted by importance):")
    for feature in sorted_features:
        if feature in importances_sorted:
            position = feature_positions[feature]
            importance = importances_sorted[feature]
            print(f"{position}. {feature}, Coefficient: {importance}")



def tree_model(tree, X_train, y_train, features):
    """
    Fits the given tree-based model to the training data and computes feature importances.

    Parameters:
    - tree (tree-base model): The decision tree classifier to be trained.
    - X_train (DataFrame or array-like): The training input samples.
    - y_train (Series or array-like): The target values for training.

    Outputs:
    - importances_sorted (dict): A dictionary of feature importances sorted in descending order.
    - feature_positions (dict): A dictionary of feature positions sorted in ascending order.
    """

   # Fit the corresponding classifier
    tree = tree.fit(X_train, y_train)

    return u.importances_model(features, tree.feature_importances_)



def importances_model(features, importances):
    """
    Generates a sorted dictionary of feature importances and their respective rankings based on the absolute value of their importance scores.

    Parameters:
    - features (list): A list of feature names corresponding to the model's features.
    - importances (list): A list of importance scores corresponding to each feature.

    Outputs:
    - importances_sorted (dict): A dictionary where keys are feature names and values are their importance scores, sorted in descending order of absolute importance.
    - feature_positions (dict): A dictionary mapping each feature to its rank based on the sorted importance values (1 for the highest importance, 2 for the second-highest, and so on).
    """
    
    # Store feature importances in a dictionary
    importances_dict = dict(zip(features, importances))
    
    # Sort by the importance value in descending order
    importances_sorted = dict(sorted(importances_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    # Create a mapping from feature to its position in the sorted dictionary
    feature_positions = {feature: idx + 1 for idx, feature in enumerate(importances_sorted)}

    return importances_sorted, feature_positions


    
def print_results_model(importances):
    """
    Prints the feature importances in a readable format.

    Parameters:
    - importances (dict): A dictionary of feature importances sorted in descending order.

    Output:
    - Prints the feature names and their corresponding importance values.
    """

    # Printing results
    print("Feature - Importance")
    for i, (key, value) in enumerate(importances.items()):
        print(f"{i+1}. {key}: {value}")



def plot_results_model(importances, features):
    """
    Plots the feature importances of a decision tree model using bar and scatter plots.

    Parameters:
    - importances (list): List containing the importance scores of the corresponding model.
    - features (list of str): A list of feature names corresponding to the training data.

    Output:
    - Displays bar and scatter plots showing the feature importances.
    """

    # Create a bar plot or scatter plot
    plt.figure(figsize=(15, 6))  # Optional: Adjust the figure size
    
    # Bar plot
    plt.bar(features, importances, color='blue', alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance of Features')
    plt.xticks(rotation=90)  # Optional: Rotate feature names for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.5)  # Optional: Add a grid
    
    # Scatter plot (alternative)
    plt.scatter(features, importances, color='g', s=100)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance of Features')
    
    plt.tight_layout()
    plt.show()


def corr_groups(df_X_train, threshold, print_=False):
    """
    Groups features based on their correlation with each other.

    Parameters:
    - df_X_train (DataFrame): The DataFrame containing the features to be analyzed.
    - threshold (float): The correlation threshold above which features are grouped together.
    - print_ (bool): If True, prints the resulting groups of correlated features.

    Output:
    - correlation_groups (dict): A dictionary where keys are group identifiers (or feature names) and values are lists of features that are correlated with each other.
    """
    
    # Create a dictionary to store groups of highly correlated columns
    correlation_groups = {}
    
    # Iterate over the columns of the DataFrame
    for feature in df_X_train.columns:
        
        # Flag to indicate if the column is assigned to an existing group
        assigned_to_group = False
        
        # Iterate over the existing groups
        for group, group_features in correlation_groups.items():
            
            # Calculate the average correlation between the current column and the columns in the group
            avg_correlation = sum(df_X_train[feature].corr(df_X_train[fea]) for fea in group_features) / len(group_features)
            
            # If the average correlation is greater than the threshold, assign the column to the group
            if avg_correlation > threshold:
                correlation_groups[group].append(feature)
                assigned_to_group = True
                break
        
        # If the column is not assigned to an existing group, create a new group
        if not assigned_to_group:
            correlation_groups[feature] = [feature]

    if print_:
        # Showing the resulting groups
        for i, (group, group_columns) in enumerate(correlation_groups.items()):
            print(f"\n Group {i+1}: {group_columns}")

    return correlation_groups



def best_features_corr_groups(corr_groups, importances, print_=False):
    """
    Selects the most important feature from each group of correlated features.

    Parameters:
    - corr_groups (dict): A dictionary where keys are group identifiers (e.g., group numbers or names) and values are lists of features that are correlated.
    - importances (dict): A dictionary mapping features to their importance scores. Example: {'feature1': 0.95, 'feature2': 0.80, ...}
    - print_ (bool): If True, prints the best feature for each group.

    Output:
    - best_features (list): A list of the most important features from each group.
    """

    # Initialize a list to store the best features
    best_features = []
    
    # Iterate over the correlation groups
    for group, group_columns in corr_groups.items():
        # Find the feature with the highest importance in the current group
        best_feature = max(group_columns, key=lambda x: importances.get(x, float('-inf')))
        best_features.append(best_feature)

    # Print the selected features if print_ is True
    if print_:
        for i, feature in enumerate(best_features):
            print(f"Best feature for group {i+1}: {feature}")
    
    return best_features



def compare_best_features(*lists):
    """
    Compare unique features across multiple lists and print a table showing unique features for each list, along with a column for features common across all lists.

    Parameters:
    - *lists: Variable number of lists containing features.

    Outputs:
    - unique_features: A list of lists, each containing the unique features for each input list.
    - common_features: A list containing features common to all input lists.
    """
    
    # Convert lists to sets
    sets = [set(lst) for lst in lists]
    
    # Find unique elements for each list
    unique_features = []
    for current_set in sets:
        unique = current_set - set.union(*(s for s in sets if s is not current_set))
        unique_features.append(sorted(unique))  # Sort for consistent ordering
    
    # Find common features across all lists
    common_features = set.intersection(*sets)
    
    # Determine the maximum number of unique features and common features to align rows
    max_length = max(len(uniq) for uniq in unique_features)
    max_common_length = len(common_features)
    column_width = 30  # Set the width of each column for alignment
    common_column_width = column_width  # Same width for the common column
    
    # Print headers
    headers = [f"List {i + 1}" for i in range(len(unique_features))]
    print(f"{'Rank':<10}", end='')
    for header in headers:
        print(f"{header:<{column_width}}", end='')
    print(f"{'Common Features':<{common_column_width}}")
    
    print("=" * (column_width * len(unique_features) + common_column_width - 5))
    
    # Print unique features and common features
    for i in range(max(max_length, max_common_length)):
        print(f"{i + 1:<10}", end='')  # Print the rank/group number
        for unique in unique_features:
            if i < len(unique):
                print(f"{unique[i]:<{column_width}}", end='')
            else:
                print(f"{'':<{column_width}}", end='')  # Empty space for alignment
        
        if i < max_common_length:
            print(f"{sorted(common_features)[i]:<{common_column_width}}", end='')
        else:
            print(f"{'':<{common_column_width}}", end='')  # Empty space for alignment
        
        print()  # Newline after each row
    
    return unique_features, sorted(common_features)


def calculate_feature_scores(feature_dict):
    """
    Calculate average and weighted scores for each feature based on their importances and positions.
    
    Parameters:
    - feature_dict: Dictionary containing feature importance and position data for models and filters.
    
    Output:
    - feature_dict: Dictionary with added 'scores' key containing 'avg' and 'weighted' scores.
    """
    
    # Define weights for models and filters
    model_weights = {
        'LR': 3,  # Higher weight for LR
        'GB': 5,  # Higher weight for GB
        'TC': 1,  # Lower weight for TC
        'RF': 1,  # Lower weight for RF
        'XG': 1   # Lower weight for XG
    }
    
    filter_weights = {
        'mi': 4,  # Higher weight for mi
        'cc': 2,  # Higher weight for cc
        'v': 1    # Lower weight for v
    }

    for feature, data in feature_dict.items():
        model_scores = []
        filter_scores = []
        
        # Calculate model scores
        for model, (importance, position) in data['models'].items():
            if importance is not None and position is not None:
                model_score = importance / (position + 1)  # Normalize by position
                weighted_score = model_score * model_weights.get(model, 1)  # Apply weight
                model_scores.append(model_score)
                filter_scores.append(weighted_score)

        # Calculate filter scores
        for filter_, (importance, position) in data['filters'].items():
            if importance is not None and position is not None:
                filter_score = importance / (position + 1)  # Normalize by position
                weighted_score = filter_score * filter_weights.get(filter_, 1)  # Apply weight
                filter_scores.append(weighted_score)
        
        # Calculate average score
        avg_score = (sum(model_scores) + sum(filter_scores)) / (len(model_scores) + len(filter_scores))
        
        # Calculate weighted score
        weighted_score = sum(filter_scores) + sum(model_scores)
        
        # Update feature dict with scores
        feature_dict[feature] = {
            'models': data['models'],
            'filters': data['filters'],
            'scores': {
                'avg': avg_score,
                'weighted': weighted_score
            }
        }
    
    return feature_dict



def extract_scores(feature_dict):
    """
    Extract average and weighted scores from the updated feature_dict.

    Parameters:
    - feature_dict: Dictionary containing feature importance and position data with scores.
    
    Outputs:
    - avg_scores_list: List of tuples with feature names and their average scores.
    - weighted_scores_list: List of tuples with feature names and their weighted scores.
    """
    
    avg_scores_list = []
    weighted_scores_list = []

    for feature, data in feature_dict.items():
        avg_score = data['scores'].get('avg', 0)
        weighted_score = data['scores'].get('weighted', 0)
        
        avg_scores_list.append(avg_score)
        weighted_scores_list.append(weighted_score)
    
    return avg_scores_list, weighted_scores_list



def print_model_comparison(model_positions, model_importances):
    """
    Prints a comparison of feature rankings and scores across different models.

    Parameters:
    - model_positions (dict): Dictionary where keys are model names ('LR', 'TC', 'RF', 'GB', 'XG'), and values are dictionaries 
      mapping features to their ranks (positions). Example: {'LR': {'feature1': 1, 'feature2': 2}, ...}
    - model_importances (dict): Dictionary where keys are model names ('LR', 'TC', 'RF', 'GB', 'XG'), and values are dictionaries 
      mapping features to their importance scores. Example: {'LR': {'feature1': 0.95, 'feature2': 0.80}, ...}

    Output:
    - Prints a table comparing the ranks and scores of features for each model.
    """

    # Print table headers
    print(f"{'':<10} {'Logistic Regression':<30} {'Tree Classifiers':<30} {'Random Forest':<25} {'Gradient Boosting':<33} {'XGBoost':<30}")
    print(f"{'Rank':<10} {'Feature':<12} {'Score':<15} {'Feature':<12} {'Score':<15} {'Feature':<12} {'Score':<15} {'Feature':<12} {'Score':<15} {'Feature':<12} {'Score':<15}")
    print("="*150)
    
    # Sort features for each model by their ranks in descending order
    sorted_features = {key: sorted(val.items(), key=lambda item: item[1], reverse=True) for key, val in model_positions.items()}
    
    # Get all unique features across all models
    all_features = set()
    for pos_dict in model_positions.values():
        all_features.update(pos_dict.keys())
    
    # Create a dictionary to store feature ranks and scores for each model
    feature_rank_scores = {}
    for feature in all_features:
        feature_rank_scores[feature] = {
            'LR': (model_positions['LR'].get(feature, float('inf')), model_importances['LR'].get(feature, 'N/A')),
            'TC': (model_positions['TC'].get(feature, float('inf')), model_importances['TC'].get(feature, 'N/A')),
            'RF': (model_positions['RF'].get(feature, float('inf')), model_importances['RF'].get(feature, 'N/A')),
            'GB': (model_positions['GB'].get(feature, float('inf')), model_importances['GB'].get(feature, 'N/A')),
            'XG': (model_positions['XG'].get(feature, float('inf')), model_importances['XG'].get(feature, 'N/A'))
        }
    
    # Determine the maximum rank to display based on the largest list of ranked features
    max_rank = max(len(sorted_features[key]) for key in sorted_features)

    # Print each rank with the corresponding feature and score for each model
    for rank in range(1, max_rank + 1):
        # Get feature and score for Logistic Regression (LR) at the current rank
        feature_LR = next((f for f, r in sorted_features['LR'] if r == rank), 'N/A')
        score_LR = f"{model_importances['LR'].get(feature_LR, 'N/A'):.3f}" if feature_LR != 'N/A' else 'N/A'
        
        # Get feature and score for Tree Classifiers (TC) at the current rank
        feature_TC = next((f for f, r in sorted_features['TC'] if r == rank), 'N/A')
        score_TC = f"{model_importances['TC'].get(feature_TC, 'N/A'):.3f}" if feature_TC != 'N/A' else 'N/A'
        
        # Get feature and score for Random Forest (RF) at the current rank
        feature_RF = next((f for f, r in sorted_features['RF'] if r == rank), 'N/A')
        score_RF = f"{model_importances['RF'].get(feature_RF, 'N/A'):.3f}" if feature_RF != 'N/A' else 'N/A'
        
        # Get feature and score for Gradient Boosting (GB) at the current rank
        feature_GB = next((f for f, r in sorted_features['GB'] if r == rank), 'N/A')
        score_GB = f"{model_importances['GB'].get(feature_GB, 'N/A'):.3f}" if feature_GB != 'N/A' else 'N/A'
        
        # Get feature and score for XGBoost (XG) at the current rank
        feature_XG = next((f for f, r in sorted_features['XG'] if r == rank), 'N/A')
        score_XG = f"{model_importances['XG'].get(feature_XG, 'N/A'):.3f}" if feature_XG != 'N/A' else 'N/A'
        
        # Print the row for the current rank
        print(f"{rank:<10} {feature_LR:<12} {score_LR:<15} {feature_TC:<12} {score_TC:<15} {feature_RF:<12} {score_RF:<15} {feature_GB:<12} {score_GB:<15} {feature_XG:<12} {score_XG:<15}")



def print_filter_comparison(filter_positions, filter_importances):
    """
    Sorts and prints feature rankings and scores across different filter methods.

    Parameters:
    - filter_positions (dict): Dictionary where keys are filter method names ('v', 'mi', 'cc'), and values are dictionaries 
      mapping features to their ranks (positions). Example: {'v': {'feature1': 1, 'feature2': 2}, ...}
    - filter_importances (dict): Dictionary where keys are filter method names ('v', 'mi', 'cc'), and values are dictionaries 
      mapping features to their importance scores. Example: {'v': {'feature1': 0.95, 'feature2': 0.80}, ...}

    Output:
    - Prints a table comparing the ranks and scores of features for each filter method.
    """

    # Print table headers
    print(f"{'':<23} {'Variance':<30} {'Mutual Information':<33} {'Correlation Coefficient':<40}")
    print(f"{'Rank':<15} {'Feature':<15} {'Score':<20} {'Feature':<15} {'Score':<20} {'Feature':<15} {'Score':<20}")
    print("="*115)
    
    # Create a sorted list of features for each filter method, ordered by their ranks
    sorted_features = {key: sorted(val.items(), key=lambda x: x[1]) for key, val in filter_positions.items()}
    
    # Get all unique features across all filter methods
    all_features = set()
    for pos_dict in filter_positions.values():
        all_features.update(pos_dict.keys())
    
    # Create a dictionary to store feature ranks and scores for each method
    feature_rank_scores = {}
    for feature in all_features:
        feature_rank_scores[feature] = {
            'v': (filter_positions['v'].get(feature, float('inf')), filter_importances['v'].get(feature, 'N/A')),
            'mi': (filter_positions['mi'].get(feature, float('inf')), filter_importances['mi'].get(feature, 'N/A')),
            'cc': (filter_positions['cc'].get(feature, float('inf')), filter_importances['cc'].get(feature, 'N/A'))
        }
    
    # Determine the maximum rank to display (based on the largest list of ranked features)
    max_rank = max(len(sorted_features[key]) for key in sorted_features)

    # Print each rank with corresponding feature and score for each filter method
    for rank in range(1, max_rank + 1):
        # Get feature and score for 'v' method at the current rank
        feature_v = next((f for f, r in sorted_features['v'] if r == rank), 'N/A')
        score_v = f"{filter_importances['v'].get(feature_v, 'N/A'):.3f}" if feature_v != 'N/A' else 'N/A'
        
        # Get feature and score for 'mi' method at the current rank
        feature_mi = next((f for f, r in sorted_features['mi'] if r == rank), 'N/A')
        score_mi = f"{filter_importances['mi'].get(feature_mi, 'N/A'):.3f}" if feature_mi != 'N/A' else 'N/A'
        
        # Get feature and score for 'cc' method at the current rank
        feature_cc = next((f for f, r in sorted_features['cc'] if r == rank), 'N/A')
        score_cc = f"{filter_importances['cc'].get(feature_cc, 'N/A'):.3f}" if feature_cc != 'N/A' else 'N/A'

        # Print the rank, feature names, and scores for each filter method
        print(f"{rank:<15} {feature_v:<15} {score_v:<20} {feature_mi:<15} {score_mi:<20} {feature_cc:<15} {score_cc:<20}")



def print_model_comparison_corr_groups(best_features, importances):
    """
    Prints a comparison of the top features and their importance scores across different models.

    Parameters:
    - best_features (dict): Dictionary where keys are model names ('LR', 'GB', 'TC', 'RF', 'XG') and values are lists of top features 
      for each model. Example: {'LR': ['feature1', 'feature2'], 'GB': ['featureA', 'featureB'], ...}
    - importances (dict): Dictionary where keys are model names ('LR', 'GB', 'TC', 'RF', 'XG') and values are dictionaries 
      mapping features to their importance scores. Example: {'LR': {'feature1': 0.95, 'feature2': 0.80}, ...}

    Output:
    - Prints a table comparing the top features and their scores for each model.
    """

    # Print table headers
    print(f"{'':<14} {'LR':<23} {'GB':<23} {'TC':<22} {'RF':<24} {'XG':<23} {'VE':<30}")
    print(f"{'Group':<6} {'Feature':<10} {'Score':<12} {'Feature':<10} {'Score':<12} {'Feature':<10} {'Score':<12} {'Feature':<10} {'Score':<12} {'Feature':<10} {'Score':<12} {'Feature':<10} {'Score':<12}")
    print("="*150)

    # Determine the number of top features (assuming all lists are of equal length)
    num_features = len(best_features['LR'])

    # Iterate through the list of top features for each model
    for group in range(num_features):
        # Extract features and scores for each model
        feature_LR = best_features['LR'][group]
        score_LR = f"{importances['LR'].get(feature_LR, 'N/A'):.3f}"

        feature_GB = best_features['GB'][group]
        score_GB = f"{importances['GB'].get(feature_GB, 'N/A'):.3f}"

        feature_TC = best_features['TC'][group]
        score_TC = f"{importances['TC'].get(feature_TC, 'N/A'):.3f}"

        feature_RF = best_features['RF'][group]
        score_RF = f"{importances['RF'].get(feature_RF, 'N/A'):.3f}"

        feature_XG = best_features['XG'][group]
        score_XG = f"{importances['XG'].get(feature_XG, 'N/A'):.3f}"

        feature_V = best_features['V'][group]
        score_V = f"{importances['V'].get(feature_V, 'N/A'):.3f}"

        # Print the row for the current group of top features
        print(f"{group + 1:<6} {feature_LR:<10} {score_LR:<12} {feature_GB:<10} {score_GB:<12} {feature_TC:<10} {score_TC:<12} {feature_RF:<10} {score_RF:<12} {feature_XG:<10} {score_XG:<12} {feature_V:<10} {score_V:<12}")



def print_scores_comparison(feature_positions, feature_importances):
    """
    Sorts and prints feature rankings and scores across different metrics (average and weighted).

    Parameters:
    - feature_positions (dict): Dictionary where keys are metrics ('Average', 'Weighted'), and values are dictionaries 
      mapping features to their ranks (positions). Example: {'Average': {'feature1': 1, 'feature2': 2}, ...}
    - feature_importances (dict): Dictionary where keys are metrics ('Average', 'Weighted'), and values are dictionaries 
      mapping features to their importance scores. Example: {'Average': {'feature1': 0.95, 'feature2': 0.80}, ...}

    Output:
    - Prints a table comparing the ranks and scores of features for each metric.
    """

    # Print table headers with subcolumn names
    print(f"{'':<15} {'Average':<30} {'Weighted':<30}")
    print(f"{'Rank':<10} {'ROI':<10} {'Score':<20} {'ROI':<10} {'Score':<20}")
    print("=" * 62)
    
    # Create a sorted list of features for each metric, ordered by their ranks
    sorted_features_avg = sorted(feature_positions['Average'].items(), key=lambda x: x[1])
    sorted_features_weighted = sorted(feature_positions['Weighted'].items(), key=lambda x: x[1])
    
    # Get all unique features across both metrics
    all_features = set(feature_positions['Average'].keys()).union(feature_positions['Weighted'].keys())
    
    # Determine the maximum rank to display (based on the largest list of ranked features)
    max_rank = max(len(sorted_features_avg), len(sorted_features_weighted))

    # Print each rank with corresponding feature and score for each metric
    for rank in range(1, max_rank + 1):
        # Get feature and score for 'Average' metric at the current rank
        feature_avg = next((f for f, r in sorted_features_avg if r == rank), 'N/A')
        score_avg = f"{feature_importances['Average'].get(feature_avg, 'N/A'):.3f}" if feature_avg != 'N/A' else 'N/A'
        
        # Get feature and score for 'Weighted' metric at the current rank
        feature_weighted = next((f for f, r in sorted_features_weighted if r == rank), 'N/A')
        score_weighted = f"{feature_importances['Weighted'].get(feature_weighted, 'N/A'):.3f}" if feature_weighted != 'N/A' else 'N/A'

        # Print the rank, feature names, and scores for each metric
        print(f"{rank:<10} {feature_avg:<10} {score_avg:<20} {feature_weighted:<10} {score_weighted:<20}")



def get_top_ROIs(dicts, top_n=10, print_=False):
    """
    Extracts the top N features from each provided dictionary based on their positions.
    
    Parameters:
    - dicts (list of tuples): List of tuples where each tuple contains a string identifier and a dictionary with feature positions.
    - top_n (int): The number of top features to extract.
    
    Output:
    - dict: A dictionary with top N features for each provided dictionary.
    """
    
    top_features = {}

    for identifier, d in dicts.items():
        # Sort features based on their positions
        sorted_features = sorted(d.items(), key=lambda item: item[1])
        
        # Extract top N features
        top_features[identifier] = [feature for feature, position in sorted_features[:top_n]]
    
    # Print the top features
    """print(f"\nTop {top_n} Features for Each Model:")
    for identifier, features in top_features.items():
        print(f"{identifier}:")
        for i, feature in enumerate(features):
            print(f"  Rank {i + 1}: {feature}")"""

    if print_:
        u.print_top_ROIs(top_features)
    
    return top_features



def print_top_ROIs(top_features):
    """
    Prints the top features from each provided dictionary in a structured columns format.
    
    Parameters:
    - top_features (dict): A dictionary with top features for each dictionary.
    
    Output:
    - Prints a table showing the top features for comparison.
    """
    
    model_keys = list(top_features.keys())
    
    # Print headers
    header_format = "{:<20}"  # Format string for column headers
    headers = ['Rank'] + model_keys
    header_line = '  '.join([header_format.format(header) for header in headers])
    print("\nFeature Comparison Across Models:")
    print(header_line)
    print("=" * len(header_line))
    
    # Prepare column format strings
    column_formats = [header_format] + [header_format for _ in model_keys]
    
    # Print each row
    for rank in range(len(model_keys)):
        row = [f"{rank + 1:2}"]  # Rank column
        for key in model_keys:
            features = top_features[key]
            row.append(f"{features[rank]:<20}")  # Adjust column width as needed
        row_line = '  '.join([fmt.format(item) for fmt, item in zip(column_formats, row)])
        print(row_line)



def choose_dict():
    """
    Provides a menu for selecting the dictionary to create, save and return.
    - tp (int): Determines the dictionary to create and save:
        - 1: Creates and saves the info dictionary.
        - 2: Returns data as NumPy arrays.

    Parameters:
      None

    Output:
    - int: The selected data format option.
    """
    
    print("\nSelect the data format:")
    print("1: Create and save the info dictionary: Train and test patients with the random state.")
    print("2: Create, sace and returns the results dictionary: Importances and positions with respect each model")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in [1, 2]:
                return choice
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")



def save_json(path, tp=None, d=None, l1=None, l2=None, rs=None):
    """
    Saves the provided data into a JSON file.

    Parameters:
    - l1 (list): List of training patient identifiers.
    - l2 (list): List of testing patient identifiers.
    - rs (any): Additional data to include in the JSON file.
    - path (str): Directory path where the JSON file will be saved.

    Outputs:
    - Saves a JSON file named 'info_dict.json' in the specified directory.
    """
    
    if tp == None:
        tp = u.choose_dict()
        
    else:
        # Create dictionary
        if tp == 1:
            if l1 is None or l2 is None or rs is None:
                raise ValueError("Parameters 'l1', 'l2', and 'rs' must be provided for info_dict.")
            
            name = 'info_dict.json'
            data_dict = {
                'train_patients': list(l1),
                'test_patients': list(l2),
                'rs': rs
            }

        elif tp == 2:
            if d is None:
                raise ValueError("Parameter 'feature_dict' must be provided for results_dict.")
            
            name = 'result_dict.json'
            data_dict = d

        else:
            raise ValueError("Invalid value for 'tp'. Use 1 for info_dict or 2 for results_dict.")
        
    # Save the dictionary to a JSON file
    try:
        with open(os.path.join(path, name), 'w') as f:
            json.dump(data_dict, f, indent=4)  # Added indent for better readability
        print(f"Data successfully saved to {os.path.join(path, name)}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")



def load_json(file_path):
    """
    Loads JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file to be loaded.

    Output:
    - dict: Data loaded from the JSON file.
    """
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def display_train_test_rs(data):
    """
    Displays a comparison of train patients, test patients, and random states for each partition in a tabular format.

    Parameters:
    - data (dict): Dictionary where keys are partition names (e.g., 'P1', 'P2') and values are dictionaries containing
      'train_patients', 'test_patients', and 'rs' keys.

    Output:
    - Prints a table showing train patients, test patients, and random states for each partition.
    """

    # Print the table headers
    partitions = ['P1', 'P2', 'P3', 'P4', 'P5']
    print(f"{'':<11} {' '.join([f'{p:<30}' for p in partitions])}")
    print(f"{'Train':<10} {'Test':<10} {'RS':<9}" * len(partitions))
    print("=" * (len(partitions) * 30))

    # Extract the maximum length of the train or test patient lists across partitions to standardize rows
    max_length = max(max(len(content['train_patients']), len(content['test_patients'])) for content in data.values())

    # Loop over each row index up to the maximum length
    for i in range(max_length):
        row_data = []

        # For each partition, print the train, test, and rs data, handling cases where the list might be shorter than max_length
        for partition in partitions:
            train_patients = sorted(data[partition]['train_patients'], key=lambda x: int(x[-3:]) if len(x) >= 3 else 0)
            test_patients = sorted(data[partition]['test_patients'], key=lambda x: int(x[-3:]) if len(x) >= 3 else 0)
            rs = data[partition]['rs']

            # Add train patient if exists, otherwise add empty string
            train_value = train_patients[i] if i < len(train_patients) else ''
            test_value = test_patients[i] if i < len(test_patients) else ''
            rs_value = rs if i == 0 else ''  # Only show RS once for each partition

            # Add the values to the row data
            row_data.append(f"{train_value:<10} {test_value:<10} {rs_value:<8}")

        # Join all columns and print them
        print(f"{' '.join(row_data)}")



def extract_and_sort_importances_positions(data, selected_key):
    """
    Extracts and sorts importances and positions for a specific model/filter from the provided data.
    If `selected_key='all'`, it extracts and sorts importances and positions for all models/filters.

    Parameters:
    - data (dict): Dictionary where keys are partition names (e.g., 'P1', 'P2') and values are dictionaries containing 'models' and 'filters' with associated importance and position values.
    - selected_key (str): Key to specify which model or filter to extract importances and positions for, or 'all' to extract all available models/filters.

    Outputs:
    - importances (dict): A dictionary where keys are partition names, and values are dictionaries of sorted importances. If `selected_key='all'`, values are dictionaries with model/filter names as keys and another dictionary of ROIs as values.
    - positions (dict): A dictionary where keys are partition names, and values are dictionaries of sorted positions. If `selected_key='all'`, values are dictionaries with model/filter names as keys and another dictionary of ROIs as values.
    """
    
    # Initialize dictionaries to store importances and positions for each partition
    importances = {}
    positions = {}

    # Iterate over each partition in the provided data
    for partition, content in data.items():
        
        # Case 1: Extract and sort data for all models and filters
        if selected_key == 'all':
            importance_dict = {}  # Stores importances by model/filter for the current partition
            position_dict = {}  # Stores positions by model/filter for the current partition
            
            # Iterate over each ROI in the current partition
            for roi, roi_content in content.items():
                
                # Extract importances and positions for each model in the current ROI
                for model_key, (importance, position) in roi_content['models'].items():
                    if model_key not in importance_dict:
                        importance_dict[model_key] = {}  # Initialize dictionary for new model
                    importance_dict[model_key][roi] = importance  # Store importance

                    if model_key not in position_dict:
                        position_dict[model_key] = {}  # Initialize dictionary for new model
                    position_dict[model_key][roi] = position  # Store position
                
                # Extract importances and positions for each filter in the current ROI
                for filter_key, (importance, position) in roi_content['filters'].items():
                    if filter_key not in importance_dict:
                        importance_dict[filter_key] = {}  # Initialize dictionary for new filter
                    importance_dict[filter_key][roi] = importance  # Store importance

                    if filter_key not in position_dict:
                        position_dict[filter_key] = {}  # Initialize dictionary for new filter
                    position_dict[filter_key][roi] = position  # Store position
            
            # Sort importances and positions for each model/filter by absolute value
            importances[partition] = {
                model_key: dict(sorted(roi_dict.items(), key=lambda item: abs(item[1]), reverse=True))
                for model_key, roi_dict in importance_dict.items()
            }
            positions[partition] = {
                model_key: dict(sorted(roi_dict.items(), key=lambda item: abs(item[1])))
                for model_key, roi_dict in position_dict.items()
            }
        
        # Case 2: Extract and sort data for a specific model or filter
        else:
            importance_dict = {}  # Stores importances for the selected model/filter
            position_dict = {}  # Stores positions for the selected model/filter
            
            # Iterate over each ROI in the current partition
            for roi, roi_content in content.items():
                # Check if the selected key exists in models or filters and extract data
                if selected_key in roi_content['models']:
                    importance, position = roi_content['models'][selected_key]
                elif selected_key in roi_content['filters']:
                    importance, position = roi_content['filters'][selected_key]
                else:
                    continue  # Skip ROI if the selected key is not found
                
                # Store the extracted importance and position
                importance_dict[roi] = importance
                position_dict[roi] = position
            
            # Sort the extracted importances and positions by absolute value
            importances[partition] = dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))
            positions[partition] = dict(sorted(position_dict.items(), key=lambda item: abs(item[1])))

    # Print the sorted importances and positions for verification
    print(f"\nExtracted and Sorted Importances for '{selected_key}':")
    for partition, importance_dict in importances.items():
        print(f"\n{partition} Partition Importances:")
        if selected_key == 'all':
            for model_key, roi_dict in importance_dict.items():
                print(f"Model: {model_key}")
                for roi, importance in roi_dict.items():
                    print(f"  ROI: {roi}, Importance: {importance:.3f}")
        else:
            for roi, importance in importance_dict.items():
                print(f"ROI: {roi}, Importance: {importance:.3f}")
    
    print(f"\nExtracted and Sorted Positions for '{selected_key}':")
    for partition, position_dict in positions.items():
        print(f"\n{partition} Partition Positions:")
        if selected_key == 'all':
            for model_key, roi_dict in position_dict.items():
                print(f"Model: {model_key}")
                for roi, position in roi_dict.items():
                    print(f"  ROI: {roi}, Position: {position:.3f}")
        else:
            for roi, position in position_dict.items():
                print(f"ROI: {roi}, Position: {position:.3f}")
    
    # Return the sorted importances and positions
    return importances, positions



def print_partition_comparison(global_positions, global_importances):
    """
    Prints a comparison of feature rankings and scores across different models/filters within each partition.

    Parameters:
    - global_positions (dict): Dictionary where keys are partition names (e.g., 'P1', 'P2') and values are either:
        a) Dictionaries mapping features to their ranks (positions) for a specific model/filter.
        b) Dictionaries where keys are model/filter names, and values are dictionaries of feature ranks.
    - global_importances (dict): Dictionary where keys are partition names (e.g., 'P1', 'P2') and values are either:
        a) Dictionaries mapping features to their importance scores for a specific model/filter.
        b) Dictionaries where keys are model/filter names, and values are dictionaries of feature importance scores.

    Output:
    - Prints a table comparing the ranks and scores of features for each partition.
    """

    # Check if the data contains multiple models/filters
    # This checks whether the values are dictionaries of dictionaries (indicating multiple models/filters)
    is_multiple_models = isinstance(next(iter(global_positions.values())), dict) and \
                         isinstance(next(iter(global_positions.values())).get(next(iter(next(iter(global_positions.values())))), {}), dict)

    if is_multiple_models:
        # Case 1: Handling multiple models/filters within each partition
        # Extract the list of all model/filter names to be used in headers and data extraction
        all_models_filters = list(next(iter(global_positions.values())).keys())

        # Iterate over each partition
        for partition in global_positions.keys():
            # Print the partition name as a header
            print(f"\nPartition: {partition}")
            
            # Prepare and print the sub-headers for each model/filter
            sub_headers = "".join([f"{'Feature (' + mf + ')':<15}" for mf in all_models_filters])
            print(f"{'Rank':<10} {sub_headers}")
            print("="*150)
    
            # Determine the maximum rank across all models/filters for the current partition
            max_rank = max(len(global_positions[partition].get(model_filter, {})) for model_filter in all_models_filters)
    
            # Iterate over each rank from 1 to max_rank
            for rank in range(1, max_rank + 1):
                row_data = []
                
                # Collect data for each model/filter at the current rank
                for model_filter in all_models_filters:
                    positions = global_positions[partition].get(model_filter, {})
                    importances = global_importances[partition].get(model_filter, {})
                    
                    # Identify the feature corresponding to the current rank
                    feature = next((f for f, r in sorted(positions.items(), key=lambda x: x[1]) if r == rank), 'N/A')
                    
                    # Append the feature name to the row data (feature may be 'N/A' if not present at this rank)
                    row_data.append(f"{feature:<15}")
                
                # Print the entire row for the current rank
                print(f"{rank:<10} {''.join(row_data)}")
    
            print("="*150)
            
    else:
        # Case 2: Handling a single model/filter across all partitions
        # Print the headers for ranks, features, and scores
        print(f"{'':<18} {'P1':<26} {'P2':<26} {'P3':<26} {'P4':<26} {'P5':<30}")
        print(f"{'Rank':<10} {'Feature':<11} {'Score':<14} {'Feature':<11} {'Score':<14} {'Feature':<11} {'Score':<14} {'Feature':<11} {'Score':<14} {'Feature':<11} {'Score':<14}")
        print("="*150)
        
        # Determine the maximum rank across all partitions
        max_rank = max(len(global_positions[partition]) for partition in global_positions)
    
        # Iterate over each rank from 1 to max_rank
        for rank in range(1, max_rank + 1):
            row_data = []
            
            # Collect data for each partition at the current rank
            for partition in global_positions.keys():
                positions = global_positions[partition]
                importances = global_importances[partition]
    
                # Identify the feature corresponding to the current rank
                feature = next((f for f, r in sorted(positions.items(), key=lambda x: x[1]) if r == rank), 'N/A')
                
                # Get the importance score for the identified feature (or 'N/A' if the feature is not found)
                score = f"{importances.get(feature, 'N/A'):.3f}" if feature != 'N/A' else 'N/A'
    
                # Append the feature name and score to the row data
                row_data.append(f"{feature:<12}{score:<15}")
            
            # Print the entire row for the current rank
            print(f"{rank:<10} {''.join(row_data)}")



def show_top_n_features(dicts, top_n=10):
    """
    Extracts the top N features from each provided dictionary based on their positions.

    Parameters:
    - dicts (dict): Dictionary where keys are partition names (e.g., 'P1', 'P2') and values are either:
        a) Dictionaries mapping features to their ranks (positions) for a specific model/filter.
        b) Dictionaries where keys are model/filter names, and values are dictionaries of feature ranks.
    - top_n (int): The number of top features to extract.

    Output:
    - Prints a comparison table of the top N features for each partition.
    """
    
    # Determine if we are dealing with multiple models/filters or just one
    is_multiple_models = isinstance(next(iter(dicts.values())), dict) and isinstance(next(iter(dicts.values())).get(next(iter(next(iter(dicts.values())))), {}), dict)
    
    if is_multiple_models:
        for partition, models_dict in dicts.items():
            print(f"\nPartition: {partition}")
            model_keys = list(models_dict.keys())
            
            # Prepare top N features for each model/filter
            top_features = {model: [] for model in model_keys}
            
            for model in model_keys:
                features_with_positions = sorted(models_dict[model].items(), key=lambda item: item[1])
                top_features[model] = [feature for feature, position in features_with_positions[:top_n]]
            
            # Print headers
            header_format = "{:<12}"  # Format string for column headers
            headers = ['Rank'] + model_keys
            header_line = '  '.join([header_format.format(header) for header in headers])
            print(header_line)
            print("=" * len(header_line))
            
            # Print each row
            for rank in range(top_n):
                row = [f"{rank + 1:2}"]  # Rank column
                for model in model_keys:
                    row.append(f"{top_features[model][rank]:<12}")  # Adjust column width as needed
                row_line = '  '.join([header_format.format(item) for item in row])
                print(row_line)
            
            print("=" * len(header_line))
            
    else:
        # Single model/filter case
        model_keys = list(dicts.keys())
        
        # Prepare top N features for the single model/filter
        top_features = {key: [] for key in model_keys}
        
        for key in model_keys:
            features_with_positions = sorted(dicts[key].items(), key=lambda item: item[1])
            top_features[key] = [feature for feature, position in features_with_positions[:top_n]]
        
        # Print headers
        header_format = "{:<20}"  # Format string for column headers
        headers = ['Rank'] + model_keys
        header_line = '  '.join([header_format.format(header) for header in headers])
        print(header_line)
        print("=" * len(header_line))
        
        # Print each row
        for rank in range(top_n):
            row = [f"{rank + 1:2}"]  # Rank column
            for key in model_keys:
                row.append(f"{top_features[key][rank]:<20}")  # Adjust column width as needed
            row_line = '  '.join([header_format.format(item) for item in row])
            print(row_line)





# CLASSIFIERS







# VISUALIZATION

def setup_3d_axes():
    """
    Function to set up a 3D axes for plotting.
    
    Parameters: None
        
    Outputs:
    - ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axes object.
    """
    
    ax = plt.axes(projection="3d")
    ax.view_init(azim=-105, elev=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_zlim(0, 5)
    return ax





# DEBUG ERRORS

def move_error_files(patient_id, source_dir='H:\\CSIC\\DATA_PREPROCESSED_MICROSTATES', target_dir='H:\\CSIC\\ERRORS'):
    """
    Move error files for a given patient ID from source directory to target directory, maintaining the segment structure.

    Parameters:
    patient_id (int): The ID of the patient.
    source_dir (str): The source directory containing the data.
    target_dir (str): The target directory to move the error files to.

    Output:
    None
    """

    for segment in range(1, 17):
        # Define the source and target segment directories
        source_segment_dir = os.path.join(source_dir, f'segment_{segment}')
        target_segment_dir = os.path.join(target_dir, f'segment_{segment}')
        
        # Ensure the target segment directory exists
        os.makedirs(target_segment_dir, exist_ok=True)
        
        # Iterate through files in the source segment directory
        for file_name in os.listdir(source_segment_dir):
            if patient_id in file_name and file_name.endswith('.csv'):
                # Construct the full file paths
                source_file = os.path.join(source_segment_dir, file_name)
                target_file = os.path.join(target_segment_dir, file_name)
                print(f"Moved {patient_id} to {target_segment_dir}")
                # Move the file to the target directory
                shutil.move(source_file, target_file)
        


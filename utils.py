"""
Utility functions for EEG Preprocessing

Enrique Almazán (2024)
"""

import os
import requests
import shutil
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

import csv

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.linalg import svd
from scipy.signal import find_peaks

import mne
from mne.preprocessing import ICA, create_eog_epochs




def download_files_from_url(url, destination_directory):
    """
    Download files recursively from a given URL.

    Parameters:
    url (str): The base URL to start downloading from.
    destination_directory (str): The local directory where the files will be saved.

    Output:
    Downloads files and directories to the specified local directory.
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
    url (str): The URL of the file to download.
    destination (str): The local path where the file will be saved.

    Output:
    Saves the downloaded file to the specified local path.
    """
    
    response = requests.get(url)
    with open(destination, "wb") as file:
        file.write(response.content)




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
                    #'time': (item[0] - start_time) / sf
                }
                c1 += 1
                
            elif prev_event == 200:
                d['Oe' + str(c2 + 1)] = {
                    #'events': list_events,
                    'start_sample': start_time,
                    'end_sample': item[0],
                    #'start_time': start_time / sf,
                    #'end_time': item[0] / sf,
                    #'time': (item[0] - start_time) / sf
                }
                c2 += 1
                
            elif prev_event == 210 or prev_event == 208:
                d['Ce' + str(c3 + 1)] = {
                    #'events': list_events,
                    'start_sample': start_time,
                    'end_sample': item[0],
                    #'start_time': start_time / sf,
                    #'end_time': item[0] / sf,#
                    #'time': (item[0] - start_time) / sf
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
            #'time': (events[-1][0] - start_time) / sf
        }
    elif last_event == 200:
        d['Oe' + str(c2 + 1)] = {
            #'events': list_events,
            'start_sample': start_time,
            'end_sample': item[0],
            #'start_time': start_time / sf,
            #'end_time': events[-1][0] / sf,
            #'time': (events[-1][0] - start_time) / sf
        }
    elif last_event == 210 or prev_event == 208:
        d['Ce' + str(c3 + 1)] = {
            #'events': list_events,
            'start_sample': start_time,
            'end_sample': item[0],
            #'start_time': start_time / sf,
            #'end_time': events[-1][0] / sf,
            #'time': (events[-1][0] - start_time) / sf
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
    Function to extract segment start and end samples for different event types from a dictionary.

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
    Matches the lists of start and end samples, joins the corresponding pairs,
    and sorts them in ascending order.

    Parameters:
    ss_ce (list): List of start samples for condition 'ce'.
    ss_oe (list): List of start samples for condition 'oe'.
    es_ce (list): List of end samples for condition 'ce'.
    es_oe (list): List of end samples for condition 'oe'.

    Outputs:
    pairs (list): A sorted list of tuples, each containing a pair of start and end samples.
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

    Returns:
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

    Returns:
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



def gfp_data(eeg):
    """
    Computes the Global Field Power (GFP) from EEG data, identifies GFP peaks and valleys.

    Parameters:
    eeg (mne.Epochs or mne.Evoked): The EEG data structure from which to compute GFP.
                                     It should be an instance of mne.Epochs or mne.Evoked.

    Returns:
    tuple: A tuple containing:
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
    ModK (object): The model object used to predict microstates.
    eeg (mne.Epochs or mne.Evoked): The EEG data structure to segment.
    pairs (list of tuples): A list of tuples representing the start and end samples.
    subject (str): The subject identifier used for naming the output CSV file.
    peaks (ndarray): Indices of the GFP peaks.
    valleys (ndarray): Indices of the GFP valleys.

    Returns:
    None: The function writes the results to a CSV file.
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
    headers = ['microstate', 'index', 'event']
    
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

    This function sets the display precision for pandas DataFrame to 14 decimal places and then formats each 
    float value in the DataFrame to ensure it has up to 14 decimal places without scientific notation.

    Parameters:
    df (pd.DataFrame): The input DataFrame to format.

    Returns:
    pd.DataFrame: The formatted DataFrame with float values represented up to 14 decimal places.
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
    folder_path (str): The base path of the folder containing the microstates data.
    segment_number (int): The segment number to process.

    Output:
    Saves the processed data as a CSV file in the specified folder.
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

    # Save the DataFrame as a CSV file
    csv_path = os.path.join(target_path, f'segment_{segment_number}.csv')
    mean_df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")




def move_error_files(patient_id, source_dir='H:\\CSIC\\DATA_PREPROCESSED_MICROSTATES', target_dir='H:\\CSIC\\ERRORS'):
    """
    Move error files for a given patient ID from source directory to target directory, maintaining the segment structure.

    Parameters:
    patient_id (int): The ID of the patient.
    source_dir (str): The source directory containing the data.
    target_dir (str): The target directory to move the error files to.

    Returns:
    None
    """
    # Convert patient_id to a string with leading zeros to match file naming format
    patient_id_str = str(patient_id)

    for segment in range(1, 17):
        # Define the source and target segment directories
        source_segment_dir = os.path.join(source_dir, f'segment_{segment}')
        target_segment_dir = os.path.join(target_dir, f'segment_{segment}')
        
        # Ensure the target segment directory exists
        os.makedirs(target_segment_dir, exist_ok=True)
        
        # Iterate through files in the source segment directory
        for file_name in os.listdir(source_segment_dir):
            if patient_id_str in file_name and file_name.endswith('.csv'):
                # Construct the full file paths
                source_file = os.path.join(source_segment_dir, file_name)
                target_file = os.path.join(target_segment_dir, file_name)
                
                # Move the file to the target directory
                shutil.move(source_file, target_file)
                print(f"Moved {file_name} to {target_segment_dir}")


import numpy as np
import obspy
from obspy.core.utcdatetime import UTCDateTime
import os
import pandas as pd
from scipy.signal import spectrogram
from skimage.transform import resize
from matplotlib.image import imread
from datetime import datetime


def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()


class data_getter:
    """Gets data"""

    def __init__(
        self,
        shumba_path="/.../SHUMBA/MSEED/",
        seismic_path="/.../seis_processed/",
        camera_path="/.../DATA_PROCESSED/CAMERA/",
    ):
        """Initialitizes paths to data. Defaults to paths on blackburn"""

        self.seismic_path = seismic_path
        self.shumba_path = shumba_path
        self.camera_path = camera_path

    def get_seismic(self, station_name, start_window, end_window, components):
        """Returns list of obspy traces of seismic data from station_name, from start_window
        to end_window, for selected components"""

        station_dir = os.path.join(self.seismic_path, station_name)
        # Get csv with name of file, start and end times
        df = pd.read_csv(os.path.join(station_dir, "files_times.csv"))
        df["start"] = [UTCDateTime(t) for t in df["start"]]
        df["end"] = [UTCDateTime(t) for t in df["end"]]
        inds = df[
            (df["start"] <= start_window) & (df["end"] >= end_window)
        ].index.tolist()
        if len(inds) == 0:
            raise ValueError(
                "The window you requested is out of range, or is spread over two files, which not implemented yet."
            )
        # print(inds) # suppressed output (michael)

        traces = []
        for ind in inds:
            file_name = df["name"].iloc[ind]
            file_path = os.path.join(station_dir, file_name)
            # Check if right component
            if any(comp in file_name for comp in components):
                # Add trace to list
                # print("reading %i"%(ind)) # suppressed output (michael)
                traces.append(
                    obspy.read(
                        file_path,
                        starttime=start_window,
                        endtime=end_window,
                        format="MSEED",
                        dtype=np.float32,
                    )
                )

        return traces

    def trim_trace(self, trace, start_window, end_window):
        trcp = trace.copy()
        trace = trcp.trim(start_window, end_window)
        del trcp
        return trace

    def get_shumba(self, station_name, start_window, end_window):
        """Returns list of obspy traces of shumba data from station_name, from start_window
        to end_window"""

        station_dir = os.path.join(self.shumba_path, station_name)
        # Get csv with name of file, start and end times
        df = pd.read_csv(os.path.join(station_dir, "files_times.csv"))
        df["start"] = [UTCDateTime(t) for t in df["start"]]
        df["end"] = [UTCDateTime(t) for t in df["end"]]
        inds = df[
            (df["start"] <= start_window) & (df["end"] >= end_window)
        ].index.tolist()
        if len(inds) == 0:
            raise ValueError(
                "The window you requested is out of range, or is spread over two files, which not implemented yet."
            )
        if len(inds) > 1:
            raise ValueError(
                "Shumba should only return one trace, there is a problem somewhere. "
            )
        # print(inds)

        traces = []
        for ind in inds:
            file_name = df["name"].iloc[ind]
            file_path = os.path.join(station_dir, file_name)
            # Add trace to list
            print("reading %i" % (ind))
            traces.append(
                obspy.read(
                    file_path,
                    starttime=start_window,
                    endtime=end_window,
                    format="MSEED",
                )
            )

        return traces

    def get_camera(self, camera_number, start_window, end_window):
        """
        Returns numpy array with pictures from selcted camera and time window.
        """
        camera_number = str(camera_number)
        camera_dir = os.path.join(self.camera_path, camera_number)
        excel_file = "Camera" + camera_number + ".xlsx"
        df = pd.read_excel(os.path.join(camera_dir, excel_file))

        df["UTCDateTime"] = [UTCDateTime(t) for t in df["UTCDateTime"]]

        inds_start = df[(df["UTCDateTime"] <= start_window)].index.tolist()
        inds_end = df[(df["UTCDateTime"] >= end_window)].index.tolist()

        if len(inds_start) == 0 or len(inds_end) == 0:
            raise ValueError("Time window selected outside range")

        # We want the latest start and earliest end
        ind_start = inds_start[-1]
        ind_end = inds_end[0]

        file_names = df["Raw Name"][ind_start:ind_end]
        # Hardcoded intermediate directory
        file_paths = [
            os.path.join(camera_dir, "processedimages", file_name)
            for file_name in file_names
        ]
        images = np.asarray([imread(file_path) for file_path in file_paths])

        return images


def generate_image(
    station, time, timewindow=10, fmin=10, fmax=50, resize=False, size=(128, 128)
):
    ##some settings for generating seismic spectrograms
    spect_seis = {}
    spect_seis["fs"] = 200
    spect_seis["NFFT"] = 400
    spect_seis["noverlap"] = 198
    spect_seis["nperseg"] = 199
    spect_seis["fft_resolution"] = spect_seis["NFFT"]
    spect_seis["fft_stride"] = spect_seis["NFFT"] - spect_seis["noverlap"]
    return _generate_image(
        spect_seis=spect_seis,
        station=station,
        time=time,
        timewindow=timewindow,
        fmin=fmin,
        fmax=fmax,
        resize=resize,
        size=size,
    )


def _generate_image(
    spect_seis,
    station,
    time,
    timewindow=10,
    fmin=10,
    fmax=50,
    resize=False,
    size=(128, 128),
):
    """
    Generates spectrogram np array"
    ...

    Attributes
    ----------
    station: str
        the name of the station
    time : obspy.UTCDateTime
        the time of event
    timewindow : int/float
        the duration (seconds) of the signal to be converted into an image
    fmin/fmax : int
        min/max frequency of the signal to be converted into an image
    resize: boolean
        if True, spectrograms are being "resized"

    """

    # define start and end time of the signal
    timestamp = obspy.UTCDateTime(time)
    start = timestamp - (timewindow / 2)
    end = timestamp + (timewindow / 2)
    # load all three seismic components

    dg = data_getter(camera_path=None)
    seis = dg.get_seismic(
        station_name=station, start_window=start, end_window=end, components=["_e_"]
    )[0]
    seis.append(
        dg.get_seismic(
            station_name=station, start_window=start, end_window=end, components=["_n_"]
        )[0].traces[0]
    )
    seis.append(
        dg.get_seismic(
            station_name=station, start_window=start, end_window=end, components=["_z_"]
        )[0].traces[0]
    )
    seis.filter("highpass", freq=fmin)
    seis.filter("lowpass", freq=fmax)
    data_seis = np.array([tr.data - np.mean(tr.data) for tr in seis]).transpose()
    # convert all three components into spectrograms and sum the spectrograms
    _, _, powerSpectrum1 = spectrogram(
        data_seis[:, 0],
        fs=spect_seis["fs"],
        nperseg=spect_seis["nperseg"],
        noverlap=spect_seis["noverlap"],
        nfft=spect_seis["NFFT"],
    )
    _, _, powerSpectrum2 = spectrogram(
        data_seis[:, 1],
        fs=spect_seis["fs"],
        nperseg=spect_seis["nperseg"],
        noverlap=spect_seis["noverlap"],
        nfft=spect_seis["NFFT"],
    )
    _, _, powerSpectrum3 = spectrogram(
        data_seis[:, 2],
        fs=spect_seis["fs"],
        nperseg=spect_seis["nperseg"],
        noverlap=spect_seis["noverlap"],
        nfft=spect_seis["NFFT"],
    )
    sum_of_spectra = 10 * np.log10(powerSpectrum1 + powerSpectrum2 + powerSpectrum3)

    # crop by selected frequencies
    fullsignal = sum_of_spectra[int(fmin * 2) : int(fmax * 2), :]

    # resize image to size
    if resize:
        fullsignal = resize(fullsignal, size, anti_aliasing=True)

    return fullsignal

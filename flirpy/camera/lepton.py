from flirpy.camera.core import Core
import sys
import pkg_resources
import subprocess
import cv2
import struct
import logging
import os

class Lepton(Core):

    def __init__(self, loglevel=logging.WARNING):
        self.cap = None
        self.conn = None

        logging.basicConfig(level=loglevel)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def find_video_device(self):
        """
        Attempts to automatically detect which video device corresponds to the PureThermal Lepton by searching for the PID and VID.

        Returns
        -------
            int
                device number
        """

        res = None

        if sys.platform.startswith('win32'):
            device_check_path = pkg_resources.resource_filename('flirpy', 'bin/find_cameras.exe')
            device_id = int(subprocess.check_output([device_check_path, "PureThermal"]).decode())

            if device_id >= 0:
                return device_id

        elif sys.platform == "darwin":
            output = subprocess.check_output(["system_profiler", "SPCameraDataType"]).decode()
            devices = [line.strip() for line in output.split("\n") if line.strip().startswith("Model")]

            device_id = 0

            for device in devices:
                if device.contains("VendorID_1E4E") and device.contains("ProductID_0100"):
                    return device_id
            
        else:
            import pyudev

            context = pyudev.Context()
            devices = pyudev.Enumerator(context)

            path = "/sys/class/video4linux/"
            video_devices = [os.path.join(path, device) for device in os.listdir(path)]
            dev = []
            for device in video_devices:
                udev = pyudev.Devices.from_path(context, device)

                try:
                    vid= udev.properties['ID_VENDOR_ID']
                    pid = udev.properties['ID_MODEL_ID']

                    if vid.lower() == "1e4e" and pid.lower() == "0100":
                        dev.append(int(device.split('video')[-1]))
                except KeyError:
                    pass
            
            # For some reason multiple devices can show up
            if len(dev) > 1:
                for d in dev:
                    cam = cv2.VideoCapture(d + cv2.CAP_V4L2)
                    data = cam.read()
                    cam.release()

                    if data[0] == True and data[1] is not None:
                        res = d
                        break
            elif len(dev) == 1:
                res = dev[0]

        return res

    def setup_video(self, device_id=None):
        """
        Setup the camera for video/frame capture.

        Attempts to automatically locate the camera, then opens an OpenCV VideoCapture object. The
        capture object is setup to capture raw video.
        """

        if device_id is None:
            device_id = self.find_video_device()
        
        if device_id is None:
            raise ValueError("Lepton not connected.")

        if sys.platform.startswith('linux'):
            self.cap = cv2.VideoCapture(device_id + cv2.CAP_V4L2)
        elif sys.platform.startswith('win32'):
            self.cap = cv2.VideoCapture(device_id + cv2.CAP_DSHOW)
        else:
            # Catch anything else, e.g. Mac?
            self.cap = cv2.VideoCapture(device_id)

        if not self.cap.isOpened():
           raise IOError("Failed to open capture device {}".format(device_id))
        
        # The order of these calls matters!
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def decode_status(self, data):
        status = { "FCC Desired": True if data & (1 << 3) else False }

        ret = (0b00, "FFC never commanded")
        if data & (0b11 << 4):
            ret = (0b11, "FFC complete")
        elif data & (0b10 << 4):
            ret = (0b10, "FFC in progress")
        elif data & (0b01 << 4):
            ret = (0b01, "FFC imminent")
        status["FFC State"] = ret

        status["AGC State"] = True if data & (1 << 12) else False
        status["Shutter lockout"] = True if data & (1 << 15) else False
        status["Overtemp shutdown imminent"] = True if data & (1 << 20) else False

        return status

    def decode_telemetry(self, image, mode="footer"):
        """
        Extracts telemetry from an image
        """
        if image.shape[1] == 160:   # Lepton 3.5
            # 3 rows of telemetry is packed in 2 rows in image
            row_A = image[-2, :80]
            row_B = image[-2, 80:]
            row_C = image[-1, :80]
        elif image.shape[1] == 80:
            row_A = image[-3]
            row_B = image[-2]
            row_C = image[-1]
        else:
            self.logger.warn("Failed to decode telemetry. Non standard image size")
            return

        self.telemetry = {}

        res = struct.unpack("<2BII16s4h6xI5h4xhIh2x6H64xIH10x", row_A)
        self.telemetry["major_version"] = res[1]
        self.telemetry["minor_version"] = res[0]
        self.telemetry["uptime_ms"] = res[2]
        self.telemetry["status"] = self.decode_status(res[3])
        self.telemetry["serial"] = res[4].hex()
        self.telemetry["software_rev"] = res[5:9]
        self.telemetry["frame_counter"] = res[9]
        self.telemetry["frame_mean"] = res[10]
        self.telemetry["FPA_temp"] = res[11]
        self.telemetry["FPA_temp_K"] = res[12]
        self.telemetry["housing_temp"] = res[13]
        self.telemetry["housing_temp_K"] = res[14]
        self.telemetry["FPA_temp_at_last_FFC_K"] = res[15] / 100.0
        self.telemetry["time_counter_at_last_FFC"] = res[16]
        self.telemetry["housing_temp_at_last_FFC_K"] = res[17] / 100.0
        self.telemetry["AGC_ROI"] = res[18:22]
        self.telemetry["AGC_clip_limit_high"] = res[22]
        self.telemetry["AGC_clip_limit_low"] = res[23]
        self.telemetry["video_output_format"] = res[24]
        self.telemetry["log2_of_FFC"] = res[25]

        if self.telemetry["video_output_format"] == 7:
            self.telemetry["video_output_format"] = "RAW14"

        res = struct.unpack("<38x8H106x", row_B)
        self.telemetry["emissivity"] = res[0] / 8192.0
        self.telemetry["background_temp_K"] = res[1] / 100.0
        self.telemetry["atmospheric_transmission"] = res[2] / 8192.0
        self.telemetry["atmospheric_temp_K"] = res[3] / 100.0
        self.telemetry["window_transmission"] = res[4] / 8192.0
        self.telemetry["window_reflection"] = res[5] / 8192.0
        self.telemetry["window_temp_K"] = res[6] / 100.0
        self.telemetry["window_reflection_temp_K"] = res[7] / 100.0

        res = struct.unpack("<10x5h2H4x2h12x4h44x10H44x", row_C)
        self.telemetry["gain_mode"] = (res[0], "High") if res[0] == 0 \
            else (res[0], "Low") if res[0] == 1 else (res[0], "Auto") if res[0] == 2 else (res[0], "unknown")
        self.telemetry["effective_gain_mode"] = (res[1], "High") if res[1] == 0 \
            else (res[1], "Low") if res[1] == 1 else (res[1], "unknown")
        self.telemetry["gain_mode_desired_flag"] = (0, "current gain mode is desired") if res[2] == 0 \
                else (1, "gain mode switch desired") if res[2] == 1 else (res[2], "unknown")
        self.telemetry["temp_gain_mode_thr_high_to_low_C"] = res[3]
        self.telemetry["temp_gain_mode_thr_low_to_high_C"] = res[4]
        self.telemetry["temp_gain_mode_thr_high_to_low_K"] = res[5]
        self.telemetry["temp_gain_mode_thr_low_to_high_K"] = res[6]
        self.telemetry["population_gain_mode_thr_high_to_low"] = res[7]
        self.telemetry["population_gain_mode_thr_low_to_high"] = res[8]
        self.telemetry["gain_mode_ROI"] = res[9:13]
        self.telemetry["TLinear_enable"] = True if res[13] > 0 else False
        self.telemetry["TLinear_resolution"] = (0, 0.1) if res[14] == 0 else (1, 0.01) if res[14] == 1 \
            else (res[14], "unknown")
        # spotmeter - SPM
        self.telemetry["SPM_mean_K"] = res[15] / 100.0
        self.telemetry["SPM_max_K"] = res[16] / 100.0
        self.telemetry["SPM_min_K"] = res[17] / 100.0
        self.telemetry["SPM_population_px"] = res[18]
        self.telemetry["SPM_ROI_start_row"] = res[19]
        self.telemetry["SPM_ROI_start_col"] = res[20]
        self.telemetry["SPM_ROI_end_row"] = res[21]
        self.telemetry["SPM_ROI_end_col"] = res[22]

        # res = struct.unpack("<2cII16x4h6xIh2xh8xhI4xhhhhhh64xI172x", image[-2, :])
        # self.major_version = res[0]
        # self.minor_version = res[1]
        # self.uptime_ms = res[2]
        # self.status = res[3]
        # self.revision = res[4:8]
        # self.frame_count = res[8]
        # self.frame_mean = res[9]
        # self.fpa_temp_k = res[10]/100.
        # self.ffc_temp_k = res[11]/100.
        # self.ffc_elapsed_ms = res[12]
        # self.agc_roi = res[13:17]
        # self.agc_clip_high = res[17]
        # self.agc_clip_low = res[18]
        # self.video_format = res[19]
        
    def grab(self, device_id=None, telemetry_mode="footer", strip_telemetry=True):
        """
        Captures and returns an image.

        Parameters
        ----------

            int
                the device ID for the camera. On a laptop, this is likely to be 1 if
                you have an internal webcam.

        Returns
        -------
            
            np.array, or None if an error occurred
                captured image
        """

        if self.cap is None:
            self.setup_video(device_id)

        res, image = self.cap.read()

        if res:
            self.decode_telemetry(image, telemetry_mode)

            if strip_telemetry:
                image = image[:-2,:]
        else:
            self.logger.warn("Failed to capture image")

        return image

    
    def release(self):
        if self.cap:
            self.cap.release()

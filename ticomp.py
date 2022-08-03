# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np
import copy
import argparse
from seg.utils.configer import Configer

from utils.flircamera import CameraManager as tcam
from utils.signal_processing_lib import lFilter
from seg.inference import ThermSeg

from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene
from PySide6.QtCore import QFile, QObject, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage
import pyqtgraph as pg


global camera_connect_status, acquisition_status, live_streaming_status, keep_acquisition_thread, perform_seg_flag, extract_breathing_signal, nose_label
acquisition_status = False
live_streaming_status = False
camera_connect_status = False
keep_acquisition_thread = True
perform_seg_flag = False
extract_breathing_signal = False
nose_label = 5

class TIComp(QWidget):
    def __init__(self, configer):
        super(TIComp, self).__init__()
        self.load_ui(configer)

    def load_ui(self, configer):
        self.configer = configer

        ckpt_root = self.configer.get('checkpoints', 'checkpoints_dir')
        ckpt_name = self.configer.get('checkpoints', 'checkpoints_name')
        self.configer.update(['network', 'resume'], os.path.join(ckpt_root, ckpt_name + '_max_performance.pth' ))

        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)

        self.tcamObj = tcam()
        self.segObj = ThermSeg(self.configer)
        input_size = self.configer.get('test', 'data_transformer')['input_size']
        self.seg_img_width = input_size[0]
        self.seg_img_height = input_size[1]

        self.ui.connectButton.pressed.connect(self.scan_and_connect_camera)
        self.ui.acquireButton.pressed.connect(self.control_acquisition)
        self.ui.segButton.pressed.connect(self.perform_segmentation_control)
        self.ui.signalExtractionButton.pressed.connect(self.extract_signal_control)
        self.resp_plot_initialized = False

        # self.fps = 50.0
        self.fps = 30.0
        self.lFilterObj = lFilter(0.05, 1.0, sample_rate=self.fps)

        self.imgAcqLoop = threading.Thread(name='imgAcqLoop', target=capture_frame_thread, daemon=True, args=(
            self.tcamObj, self.segObj, self.updatePixmap, self.updateLog, self.addRespData))
        self.imgAcqLoop.start()
        ui_file.close()

    def closeEvent(self, event):
        global camera_connect_status, acquisition_status, keep_acquisition_thread
        keep_acquisition_thread = False
        print("Please wait while camera is released...")
        time.sleep(0.5)
        if camera_connect_status and acquisition_status:
            self.tcamObj.release_camera(acquisition_status)

    def scan_and_connect_camera(self):
        global acquisition_status, camera_connect_status

        if camera_connect_status == False:
            if self.tcamObj.get_camera():
                self.cam_serial_number, self.cam_img_width, self.cam_img_height = self.tcamObj.setup_camera()
                if "error" not in self.cam_serial_number.lower():
                    self.ui.connectButton.setText("Disconnect Camera")
                    self.ui.label_2.setText("Camera Serial Number: " + self.cam_serial_number)
                    camera_connect_status = True
                    self.img_width = self.cam_img_width
                    self.img_height = self.cam_img_height
                else:
                    self.ui.label_2.setText("Error Setting Up Camera: " + self.cam_serial_number)

        if camera_connect_status:
            if acquisition_status == False:
                acquisition_status = True
                self.ui.connectButton.setText("Disconnect Camera")
                self.tcamObj.begin_acquisition()
            else:
                self.tcamObj.end_acquisition()
                acquisition_status = False
                self.ui.connectButton.setText("Scan and Connect \nThermal Camera")

    def control_acquisition(self):
        global live_streaming_status
        if live_streaming_status == False:
            self.ui.acquireButton.setText('Stop Live Streaming')
            live_streaming_status = True
            self.updateLog("Acquisition started")

        else:
            live_streaming_status = False
            self.ui.acquireButton.setText('Start Live Streaming')
            self.updateLog("Acquisition stopped")

    def updatePixmap(self, thermal_matrix):
        qimg1 = QImage(thermal_matrix.data, self.img_width, self.img_height, QImage.Format_Indexed8)
        self.ui.pix_label.setPixmap(QPixmap.fromImage(qimg1))

    def updateLog(self, message):
        self.ui.log_label.setText(message)

    def perform_segmentation_control(self):
        global perform_seg_flag
        if perform_seg_flag == False:
            self.img_width = self.seg_img_width
            self.img_height = self.seg_img_height
            perform_seg_flag = True
            self.ui.segButton.setText("Stop Segmentation")
        else:
            perform_seg_flag = False
            self.ui.segButton.setText("Perform Segmentation")
            self.img_width = self.cam_img_width
            self.img_height = self.cam_img_height

    def extract_signal_control(self):
        global extract_breathing_signal

        if extract_breathing_signal == False:

            if self.resp_plot_initialized == False:
                scene = QGraphicsScene()
                self.ui.graphicsView_resp.setScene(scene)
                self.plotWidget = pg.PlotWidget()
                self.x_ax = self.plotWidget.getAxis('bottom')
                proxy_widget = scene.addWidget(self.plotWidget)
                # self.setCentralWidget(self.plotWidget)
                self.plotWidget.setBackground('w')

            self.resp_time_axis = list(range(300))  # 300 time points
            self.resp_plot_data = np.zeros([300, ], dtype=float)  # 300 data points
            self.resp_signal = np.array([], dtype=float)
            pen = pg.mkPen(color=(0, 0, 0))
            self.data_line = self.plotWidget.plot(self.resp_time_axis, self.resp_plot_data, pen=pen)
            # self.plotWidget.setRange(xRange=(0, 300), yRange=(-1, 1))
            self.plotWidget.enableAutoRange()
            # self.plotWidget.
            extract_breathing_signal = True
            self.ui.signalExtractionButton.setText("Stop Extracting Signal")
        else:
            self.resp_plot_data = None
            extract_breathing_signal = False
            self.ui.signalExtractionButton.setText("Extract and Plot Breathing Signal")

    def addRespData(self, respVal):
        global extract_breathing_signal
        if extract_breathing_signal and respVal != 0:
            filtered_respVal = self.lFilterObj.lfilt(respVal)
            # filtered_respVal = respVal
            self.resp_plot_data = np.append(self.resp_plot_data, filtered_respVal)
            self.resp_plot_data = np.delete(self.resp_plot_data, [0])
            # self.resp_signal = np.append(self.resp_signal, filtered_respVal)

            # self.resp_time_axis = self.resp_time_axis[1:]  # Remove the first y element.
            # # Add a new value 1 higher than the last.
            # self.resp_time_axis.append(self.resp_time_axis[-1] + 1)

            self.data_line.setData(self.resp_time_axis, self.resp_plot_data)  # Update the data.

def perform_seg(thermal_matrix):
    thermal_matrix, pred_seg_mask, time_taken = segObj.run_inference(thermal_matrix)
    pred_seg_mask_org = copy.deepcopy(pred_seg_mask)
    pred_seg_mask = ((pred_seg_mask/ 3.0) + 1.0)
    img_array = copy.deepcopy(thermal_matrix)
    img_array = img_array * pred_seg_mask
    time_taken = np.round(time_taken, 3)
    info_str = info_str + "[Min Temp, Max Temp, Inference Time] = " + str([min_temp, max_temp, time_taken])

    if extract_breathing_signal:
        respVal = 0
        bbox_corners = np.argwhere(pred_seg_mask_org == nose_label)
        if bbox_corners.size > 0:
            nose_pix_min_y, nose_pix_min_x = bbox_corners.min(0)
            nose_pix_max_y, nose_pix_max_x = bbox_corners.max(0)
            nostril_box = thermal_matrix[nose_pix_max_y-20:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]
            nostril_box_label = pred_seg_mask_org[nose_pix_max_y-20:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]

            nostril_seg_matrix = nostril_box[nostril_box_label == nose_label]
            respVal = np.mean(nostril_seg_matrix)
            info_str = info_str + "; " + str([bbox_corners.min(0), bbox_corners.max(0)])

            # Highlight the nostril segmentation
            nostril_seg_mask = copy.deepcopy(pred_seg_mask_org)
            nostril_seg_mask[nostril_seg_mask != nose_label] = 0
            nostril_seg_mask[nostril_seg_mask == nose_label] = 1

            nostril_box_mask = copy.deepcopy(pred_seg_mask_org)
            nostril_box_mask[nostril_box_mask != nose_label] = 0
            nostril_box_mask[nose_pix_max_y-30:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x] = 1

            nostril_seg_mask = nostril_seg_mask * nostril_box_mask

            img_array[nostril_seg_mask == 1] = img_array[nostril_seg_mask == 1] * 1.4
        else:
            nose_mask = thermal_matrix[pred_seg_mask_org == nose_label]
            if nose_mask.size > 0:
                respVal = np.mean(nose_mask)
                info_str = info_str + "; Nostril extraction failed, using whole nose mask"
            else:
                respVal = max_temp
                info_str = info_str + "; Nose not detected!!"
        mySrc.resp_signal.emit(respVal)



# Setup a signal slot mechanism, to send data to GUI in a thread-safe way.
class Communicate(QObject):
    data_signal = Signal(np.ndarray)
    status_signal = Signal(str)
    resp_signal = Signal(float)


def capture_frame_thread(tcamObj, segObj, updatePixmap, updateLog, addRespData):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(updatePixmap)
    mySrc.status_signal.connect(updateLog)
    mySrc.resp_signal.connect(addRespData)

    global live_streaming_status, acquisition_status, camera_connect_status, keep_acquisition_thread, perform_seg_flag, extract_breathing_signal, nose_label
    pred_seg_mask = np.array([])

    while True:
        if keep_acquisition_thread:
            if camera_connect_status and acquisition_status and live_streaming_status:
                t1 = time.time()
                info_str = ""
                thermal_matrix, frame_status = tcamObj.capture_frame()

                if frame_status == "valid" and thermal_matrix.size > 0:
                    min_temp = np.round(np.min(thermal_matrix), 2)
                    max_temp = np.round(np.max(thermal_matrix), 2)

                    if perform_seg_flag:
                        thermal_matrix, pred_seg_mask, time_taken = segObj.run_inference(thermal_matrix)
                        pred_seg_mask_org = copy.deepcopy(pred_seg_mask)
                        pred_seg_mask = ((pred_seg_mask/ 3.0) + 1.0)
                        img_array = copy.deepcopy(thermal_matrix)
                        img_array = img_array * pred_seg_mask
                        time_taken = np.round(time_taken, 3)
                        info_str = info_str + "[Min Temp, Max Temp, Inference Time] = " + str([min_temp, max_temp, time_taken])

                        if extract_breathing_signal:
                            respVal = 0
                            bbox_corners = np.argwhere(pred_seg_mask_org == nose_label)
                            if bbox_corners.size > 0:
                                nose_pix_min_y, nose_pix_min_x = bbox_corners.min(0)
                                nose_pix_max_y, nose_pix_max_x = bbox_corners.max(0)
                                nostril_box = thermal_matrix[nose_pix_max_y-20:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]
                                nostril_box_label = pred_seg_mask_org[nose_pix_max_y-20:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x]

                                nostril_seg_matrix = nostril_box[nostril_box_label == nose_label]
                                respVal = np.mean(nostril_seg_matrix)
                                info_str = info_str + "; " + str([bbox_corners.min(0), bbox_corners.max(0)])

                                # Highlight the nostril segmentation
                                nostril_seg_mask = copy.deepcopy(pred_seg_mask_org)
                                nostril_seg_mask[nostril_seg_mask != nose_label] = 0
                                nostril_seg_mask[nostril_seg_mask == nose_label] = 1

                                nostril_box_mask = copy.deepcopy(pred_seg_mask_org)
                                nostril_box_mask[nostril_box_mask != nose_label] = 0
                                nostril_box_mask[nose_pix_max_y-30:nose_pix_max_y, nose_pix_min_x:nose_pix_max_x] = 1

                                nostril_seg_mask = nostril_seg_mask * nostril_box_mask

                                img_array[nostril_seg_mask == 1] = img_array[nostril_seg_mask == 1] * 1.4
                            else:
                                nose_mask = thermal_matrix[pred_seg_mask_org == nose_label]
                                if nose_mask.size > 0:
                                    respVal = np.mean(nose_mask)
                                    info_str = info_str + "; Nostril extraction failed, using whole nose mask"
                                else:
                                    respVal = max_temp
                                    info_str = info_str + "; Nose not detected!!"
                            mySrc.resp_signal.emit(respVal)

                    else:
                        img_array = thermal_matrix
                        info_str = "[Min Temp, Max Temp] = " + str([min_temp, max_temp])

                    mn_val = np.min(img_array)
                    mx_val = np.max(img_array)
                    img_array = np.round((img_array - mn_val) / (mx_val - mn_val) * 255.0)
                    img_array = img_array.astype(np.uint8)

                    mySrc.data_signal.emit(img_array)
                
                info_str = "Frame acquisition status: " + frame_status + "; " + info_str                
                # time.sleep(0.05)
                t2 = time.time()
                t_elapsed = str(t2 - t1)
                info_str = info_str + "; total_time_per_frame: " + t_elapsed
                mySrc.status_signal.emit(info_str)

            else:
                time.sleep(0.25)
        else:
            mySrc.status_signal.emit("Acquisition thread termination. Please restart the application...")
            break


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')

    parser.add_argument('REMAIN', nargs='*')

    args_parser = parser.parse_args()
    configer = Configer(args_parser=args_parser)

    app = QApplication([])
    widget = TIComp(configer=configer)
    widget.show()
    sys.exit(app.exec())

import PySpin
import os
os.system("sudo sysctl -p")

class CameraManager():
    def __init__(self):
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        self.version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' %
              (self.version.major, self.version.minor, self.version.type, self.version.build))

    def get_camera(self):
        result = True

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            print('No camera detected!')
            return False

        else:
            self.cam = self.cam_list[0]
            return True


    def setup_camera(self):

        img_height = 0
        img_width = 0
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        # Initialize camera
        self.cam.Init()
        # Retrieve GenICam nodemap
        self.nodemap = self.cam.GetNodeMap()
        self.sNodemap = self.cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(self.sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            return 'Error: Unable to set stream buffer handling mode.. Aborting...'

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            return 'Error: Unable to set stream buffer handling mode.. Aborting...'

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print(
                    'Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
                'Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print(
                    'Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            # self.cam.BeginAcquisition()
            # print('Acquiring images...')

            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
            
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
            if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
                width_to_set = node_width.GetMax()
                node_width.SetValue(width_to_set)
                print('Width set to %i...' % node_width.GetValue())
            else:
                print('Width not available...')

            node_height = PySpin.CIntegerPtr(self.nodemap.GetNode('Height'))
            if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
                width_to_set = node_height.GetMax()
                node_height.SetValue(width_to_set)
                print('Height set to %i...' % node_height.GetValue())
            else:
                print('Height not available...')

            img_width = int(node_width.GetValue())
            img_height = int(node_height.GetValue())


            # Apply mono 14 pixel format
            node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
            if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
                node_pixel_format_mono14 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono14'))
                if PySpin.IsAvailable(node_pixel_format_mono14) and PySpin.IsReadable(node_pixel_format_mono14):
                    pixel_format_mono14 = node_pixel_format_mono14.GetValue()
                    node_pixel_format.SetIntValue(pixel_format_mono14)
                    print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())
                else:
                    print('Pixel format mono 14 not available...')
            else:
                print('Pixel format not available...')

            # Enable TemperatureLinearMode
            node_TemperatureLinearMode = PySpin.CEnumerationPtr(self.nodemap.GetNode('TemperatureLinearMode'))
            if PySpin.IsAvailable(node_TemperatureLinearMode) and PySpin.IsWritable(node_TemperatureLinearMode):
                node_TemperatureLinearMode_On = PySpin.CEnumEntryPtr(node_TemperatureLinearMode.GetEntryByName('On'))
                if PySpin.IsAvailable(node_TemperatureLinearMode_On) and PySpin.IsReadable(node_TemperatureLinearMode_On):
                    TemperatureLinearMode_ON = node_TemperatureLinearMode_On.GetValue()
                    node_TemperatureLinearMode.SetIntValue(TemperatureLinearMode_ON)
                    print('TemperatureLinearMode set to %s...' % node_TemperatureLinearMode.GetCurrentEntry().GetSymbolic())
                else:
                    print('TemperatureLinearMode On not available...')
            else:
                print('TemperatureLinearMode On not available...')


            # Set TemperatureLinearResolution High
            node_TemperatureLinearResolution = PySpin.CEnumerationPtr(self.nodemap.GetNode('TemperatureLinearResolution'))
            if PySpin.IsAvailable(node_TemperatureLinearResolution) and PySpin.IsWritable(node_TemperatureLinearResolution):
                node_TemperatureLinearResolution_High = PySpin.CEnumEntryPtr(node_TemperatureLinearResolution.GetEntryByName('High'))
                if PySpin.IsAvailable(node_TemperatureLinearMode_On) and PySpin.IsReadable(node_TemperatureLinearMode_On):
                    TemperatureLinearResolution_High = node_TemperatureLinearMode_On.GetValue()
                    node_TemperatureLinearResolution.SetIntValue(TemperatureLinearResolution_High)
                    print('TemperatureLinearResolution set to %s...' % node_TemperatureLinearResolution.GetCurrentEntry().GetSymbolic())
                else:
                    print('TemperatureLinearResolution High not available...')
            else:
                print('TemperatureLinearResolution High not available...')

            return device_serial_number, img_width, img_height

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return 'Error setting up camera: ' + str(ex)

    def begin_acquisition(self):
        self.cam.BeginAcquisition()

    def end_acquisition(self):
        self.cam.EndAcquisition()

    def capture_frame(self):
        thermal_matrix = None
        frame_status = "valid"

        image_result = self.cam.GetNextImage(1000)

        #  Ensure image completion
        if image_result.IsIncomplete():
            frame_status = str(image_result.GetImageStatus())
            print("Invalid frame:", frame_status)
        else:
            # Getting the image data as a numpy array
            thermal_matrix = image_result.GetNDArray()
            thermal_matrix = (thermal_matrix * 0.04) - 273.15

        image_result.Release()
        return thermal_matrix, frame_status

    def release_camera(self, acquisition_status):

        if acquisition_status == True:
            self.cam.EndAcquisition()

        self.cam.DeInit()

        del self.cam
        # Clear camera list before releasing system
        self.cam_list.Clear()
        # Release system instance
        self.system.ReleaseInstance()

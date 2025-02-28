import json
import time
import threading
import torch
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SpeedMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender

class CameraCalibration:
    def __init__(self, calibration_folder, nx, ny):
        """Initialize camera calibration with chessboard dimensions."""
        self.calibration_folder = calibration_folder
        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None
        self.calibrate_camera()
    
    def calibrate_camera(self):
        """Calibrate camera using chessboard images."""
        import glob
        import os
        
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        images = glob.glob(os.path.join(self.calibration_folder, 'calibration*.jpg'))
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        if len(objpoints) > 0:
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
    
    def undistort(self, img):
        """Undistort an image using the camera calibration."""
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img

class Thresholding:
    def __init__(self):
        """Initialize thresholding parameters."""
        pass
    
    def forward(self, img):
        """Apply color and gradient thresholds to isolate lane lines."""
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        
        # Grayscale for Sobel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        return combined_binary

class PerspectiveTransformation:
    def __init__(self):
        """Initialize perspective transformation matrices."""
        # Source and destination points for perspective transform
        self.src = np.float32(
            [[580, 460],
             [700, 460],
             [1040, 680],
             [260, 680]])
        
        self.dst = np.float32(
            [[320, 0],
             [960, 0],
             [960, 720],
             [320, 720]])
        
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
    
    def forward(self, img):
        """Apply a perspective transform to the image."""
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warped
    
    def backward(self, img):
        """Apply an inverse perspective transform to the image."""
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)
        return unwarped

class LaneLines:
    def __init__(self):
        """Initialize lane line detection parameters."""
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.left_fit = None
        self.right_fit = None
        # x values of the last n fits
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_left_fit = None
        self.best_right_fit = None
        # radius of curvature of the line in meters
        self.left_curverad = None
        self.right_curverad = None
        # distance in meters of vehicle center from the line
        self.center_dist = None
        # difference in fit coefficients between last and new fits
        self.left_fit_delta = np.array([0, 0, 0], dtype='float')
        self.right_fit_delta = np.array([0, 0, 0], dtype='float')
    
    def forward(self, binary_warped):
        """Find and fit lane lines in a binary warped image."""
        # If lines were previously detected, use sliding windows to find them again
        if self.detected:
            return self.find_lines_given_previous(binary_warped)
        else:
            return self.find_lines_sliding_window(binary_warped)
    
    def find_lines_sliding_window(self, binary_warped):
        """Find lane lines using the sliding window approach."""
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # Identifying the x and y coordinates of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int_(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Number of sliding windows
        nwindows = 9
        # Height of windows - based on nwindows above and image shape
        window_height = np.int_(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            # Find the four boundaries of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                         (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
                         (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # If there are no indices, return the original image
            return binary_warped
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        if len(leftx) > 0 and len(rightx) > 0:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            self.detected = True
        else:
            self.detected = False
            return binary_warped
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except TypeError:
            # If no curve was found, just return the original image
            return binary_warped
        
        # Create an image to draw on and an image to show the selection window
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Color the lane area
        lane_img = np.zeros_like(out_img)
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        line_pts = np.hstack((left_line_window, right_line_window))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(lane_img, np.int_([line_pts]), (0,255, 0))
        
        return lane_img
    
    def find_lines_given_previous(self, binary_warped):
        """Find lane lines based on previous detection."""
        # Create an output image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Identify the x and y coordinates of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the width of the windows +/- margin
        margin = 100
        
        # Set minimum number of pixels found to continue using previous line fit
        minpix = 50
        
        if self.left_fit is not None and self.right_fit is not None:
            # Use the previous fit coefficients
            left_fit = self.left_fit
            right_fit = self.right_fit
            
            # Find pixels within the window defined by the polynomial
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + 
                                          left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                             (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                                          left_fit[1]*nonzeroy + left_fit[2] + margin)))
            
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + 
                                           right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                              (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))
            
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # If we don't find enough pixels, use sliding window method
            if len(leftx) < minpix or len(rightx) < minpix:
                return self.find_lines_sliding_window(binary_warped)
            
            # Fit a second order polynomial to each
            new_left_fit = np.polyfit(lefty, leftx, 2)
            new_right_fit = np.polyfit(righty, rightx, 2)
            
            # Calculate the deviation from the previous fit
            self.left_fit_delta = np.abs(new_left_fit - self.left_fit)
            self.right_fit_delta = np.abs(new_right_fit - self.right_fit)
            
            # Check if the new fit is reasonable (not too different from the previous one)
            if np.max(self.left_fit_delta) > 100 or np.max(self.right_fit_delta) > 100:
                return self.find_lines_sliding_window(binary_warped)
            
            # Update the fit
            self.left_fit = new_left_fit
            self.right_fit = new_right_fit
            
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            
            # Create an image to draw on
            lane_img = np.zeros_like(out_img)
            
            # Generate a polygon to illustrate the lane area
            left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            line_pts = np.hstack((left_line_window, right_line_window))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(lane_img, np.int_([line_pts]), (0,255, 0))
            
            return lane_img
        else:
            return self.find_lines_sliding_window(binary_warped)
    
    def plot(self, img):
        """Plot detected lane information on the image."""
        if self.left_fit is not None and self.right_fit is not None:
            # Calculate curvature and vehicle position
            self.calculate_curvature_and_position(img)
            
            # Add text displaying curvature and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Curve Radius: {int((self.left_curverad + self.right_curverad)/2)}m"
            cv2.putText(img, text, (50, 50), font, 1, (255, 255, 255), 2)
            
            direction = "left" if self.center_dist < 0 else "right"
            text = f"Vehicle is {abs(self.center_dist):.2f}m {direction} of center"
            cv2.putText(img, text, (50, 100), font, 1, (255, 255, 255), 2)
        
        return img
    
    def calculate_curvature_and_position(self, img):
        """Calculate radius of curvature and position of the car with respect to lane."""
        if self.left_fit is not None and self.right_fit is not None:
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
            
            # Calculate radius of curvature
            ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
            y_eval = np.max(ploty)
            
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty*ym_per_pix, 
                                    (self.left_fit[0]*(ploty**2) + 
                                     self.left_fit[1]*ploty + self.left_fit[2])*xm_per_pix, 2)
            
            right_fit_cr = np.polyfit(ploty*ym_per_pix, 
                                     (self.right_fit[0]*(ploty**2) + 
                                      self.right_fit[1]*ploty + self.right_fit[2])*xm_per_pix, 2)
            
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            
            self.left_curverad = left_curverad
            self.right_curverad = right_curverad
            
            # Calculate vehicle position with respect to lane center
            # Assuming camera is mounted at the center of the car
            # and the lane center is the midpoint between the detected lane lines
            lane_width = (self.right_fit[2] - self.left_fit[2]) * xm_per_pix
            vehicle_center = img.shape[1]/2
            lane_center = (self.right_fit[2] + self.left_fit[2])/2
            self.center_dist = (lane_center - vehicle_center) * xm_per_pix

class SmartDrivingSystem(ThreadWithStop):
    def __init__(self, queues, serialCom, logFile, logger, debugger=False):
        super(SmartDrivingSystem, self).__init__()
        self.queuesList = queues
        self.serialCom = serialCom
        self.logFile = logFile
        self.logger = logger
        self.debugger = debugger

        self.running = False
        self.engineEnabled = False
        self.messageConverter = MessageConverter()
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)

        # Load YOLOv8 model for traffic light and sign detection
        self.model = YOLO('/home/raspi/Downloads/traffic_sign_final.pt')
        self.class_names = self.model.names

        # Initialize PiCamera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration())
        self.picam2.start()

        # Default speed
        self.constant_speed = 200
        
        # Initialize lane detection components
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def sendToSerial(self, msg):
        """Convert message to serial command and send."""
        command_msg = self.messageConverter.get_command(**msg)
        if command_msg != "error":
            self.serialCom.write(command_msg.encode("ascii"))
            self.logFile.write(command_msg)
            if self.debugger:
                self.logger.info(f"Sent to serial: {command_msg}")

    def set_motor_speed(self, speed_value):
        """Set the motor speed."""
        command = {"action": "speed", "speed": speed_value}
        self.sendToSerial(command)
        if self.debugger:
            self.logger.info(f"Motor speed set to: {speed_value}")

    def process_frame(self, frame):
        """Process a frame through the lane detection pipeline."""
        # Apply camera calibration
        undistorted = self.calibration.undistort(frame)
        
        # Transform to bird's eye view
        warped = self.transform.forward(undistorted)
        
        # Apply thresholding to isolate lane lines
        binary = self.thresholding.forward(warped)
        
        # Detect lane lines
        lane_overlay = self.lanelines.forward(binary)
        
        # Transform back to original perspective
        lane_overlay = self.transform.backward(lane_overlay)
        
        # Combine original image with lane overlay
        result = cv2.addWeighted(frame, 1, lane_overlay, 0.6, 0)
        
        # Add lane information (curvature, position)
        result = self.lanelines.plot(result)
        
        return result

    def run(self):
        """Run loop to process camera feed and control motor based on lane detection."""
        self.running = True
        self.engineEnabled = True

        self.sendToSerial({"action": "kl", "mode": 30})
        self.logger.info("Engine enabled in mode 30.")
        time.sleep(1)

        self.set_motor_speed(self.constant_speed)
        self.logger.info(f"Initial speed set to {self.constant_speed}")

        while self._running:
            # Capture frame from camera
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (640, 480))
            
            # Process frame through lane detection pipeline
            processed_frame = self.process_frame(frame_resized)
            
            # Run traffic sign detection
            results = self.model(frame_resized)
            result_frame = results[0].plot()
            
            # Combine lane detection with sign detection visualization
            if results[0].boxes.cls.numel() > 0:
                # If signs are detected, show detection results
                display_frame = result_frame
                
                # Extract detected classes
                detected_classes = results[0].boxes.cls.cpu().numpy()
                detected_names = [self.class_names[int(cls_id)] for cls_id in detected_classes]
                
                # Adjust speed based on detected signs
                if 'stop' in detected_names:
                    self.set_motor_speed(0)
                    self.logger.info("Stop sign detected - stopping vehicle")
                elif 'speed_limit_30' in detected_names:
                    self.set_motor_speed(150)  # Reduced speed
                    self.logger.info("Speed limit sign detected - reducing speed")
                elif 'speed_limit_60' in detected_names:
                    self.set_motor_speed(250)  # Normal speed
                    self.logger.info("Speed limit sign detected - setting normal speed")
            else:
                # No signs detected, show lane detection
                display_frame = processed_frame
                
                # Adjust steering based on lane position if we have valid lane detection
                if self.lanelines.center_dist is not None:
                    # Center adjustment logic would go here
                    # For example, send steering commands based on center_dist
                    pass
            
            cv2.imshow("Smart Driving System", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.picam2.stop()

    def stop(self):
        """Stop the vehicle and clean up resources."""
        self.set_motor_speed(0)
        self.sendToSerial({"action": "kl", "mode": 0})
        time.sleep(2)
        super(SmartDrivingSystem, self).stop()
        self.logger.info("Vehicle stopped and thread terminated.")

if __name__ == "__main__":
    from multiprocessing import Queue
    import serial
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SmartDrivingSystem")

    devFile = "/dev/ttyACM0"
    serialCom = serial.Serial(devFile, 115200, timeout=0.1)
    serialCom.flushInput()
    serialCom.flushOutput()

    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }

    logFile = open("historyFile.txt", "w")

    smart_drive_thread = SmartDrivingSystem(queueList, serialCom, logFile, logger, debugger=True)
    smart_drive_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        smart_drive_thread.stop()

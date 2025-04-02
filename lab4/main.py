import cv2
import numpy as np

import time
import string
import logging
from queue import Queue
from threading import Thread, Event
import argparse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S',
                        filename='logs/log.txt')

cv2.setLogLevel(0)
open("logs/log.txt", "w").close()


def list_cameras(max_cameras=10):
    available_cameras = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            if cap.read()[0]:
                available_cameras.append(str(index))
            cap.release()

    return available_cameras

def parse():
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument("camera_name")
    parser.add_argument("resolution")
    parser.add_argument("framerate")

    try:
        args = parser.parse_args()
    except SystemExit as e:
        logger.fatal("Error parsing arguments")
        exit(1)

    width, height = args.resolution.split('x')

    if args.camera_name not in list_cameras():
        logger.fatal("Camera not found")
        logger.fatal("Fatal error")
        exit(1)

    if int(width) <= 0 or int(height) <= 0:
        logger.fatal("Width and/or height must be positive")
        logger.fatal("Fatal error")
        exit(1)

    if float(args.framerate) <= 0:
        logger.fatal("Framerate must be positive")
        logger.fatal("Fatal error")
        exit(1)

    return args.camera_name, int(width), int(height), float(args.framerate)


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    @staticmethod
    def sensor_x_work(stop_event: Event, delay: float, queue: Queue):
        sensor = SensorX(delay)
        while not stop_event.is_set():
            queue.put(sensor.get())

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1

        return self._data


class SensorCam(Sensor):
    def __init__(self, name: string, width: int, height: int):
        self._name = int(name)
        self._width = width
        self._height = height
        self._camera = cv2.VideoCapture(self._name)

    @staticmethod
    def sensor_cam_work(stop_event: Event, name: string, width: int, height: int, queue: Queue):
        sensor = SensorCam(name, width, height)
        while not stop_event.is_set():
            queue.put(sensor.get())

    def get(self) -> np.ndarray:
        ret, frame = self._camera.read()
        if ret:
            return cv2.resize(frame, (self._width, self._height))
        else:
            logger.error("Frame not caught")
            logger.error("Trying to access cv2.VideoCapture(0)")

            self._name = 0
            self._camera = cv2.VideoCapture(self._name)

            ret, frame = self._camera.read()
            if ret:
                logger.info("Access gained")
                return cv2.resize(frame, (self._width, self._height))
            else:
                logger.fatal("Frame still not caught")
                logger.fatal("Fatal error")
                exit(1)

    def __del__(self):
        self._camera.release()
        logger.info("Camera released")


class WindowImage:
    def __init__(self, frames_per_second: float):
        self._frames_per_second = frames_per_second

    def show(self, img: np.ndarray):
        cv2.imshow("Image", img)
        time.sleep(1 / self._frames_per_second)

    def __del__(self):
        cv2.destroyWindow("Image")


class ImageProcessor:
    def __init__(self, sensor_x_queues: list[Queue], sensor_cam_queue: Queue):
        self._sensor_x_data = [0, 0, 0]
        self._sensor_x_queues = sensor_x_queues
        self._sensor_cam_data = np.zeros((640, 360, 3))
        self._sensor_cam_queue = sensor_cam_queue

    @staticmethod
    def get_last_data(prev_data, queue: Queue):
        data = prev_data
        while not queue.empty():
            data = queue.get()

        return data

    def get_frame(self):
        self._sensor_cam_data = ImageProcessor.get_last_data(self._sensor_cam_data, self._sensor_cam_queue)
        for i in range(3):
            self._sensor_x_data[i] = ImageProcessor.get_last_data(self._sensor_x_data[i], self._sensor_x_queues[i])
        text = (f"SensorX 1: {self._sensor_x_data[0]} "
                f"SensorX 2: {self._sensor_x_data[1]} "
                f"SensorX 3: {self._sensor_x_data[2]}")

        return cv2.putText(img=self._sensor_cam_data,
                           text=text, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    logger.info("Starting up...")

    try:
        camera_name, cam_width, cam_height, framerate = parse()
    except Exception as exception:
        logger.exception("Exception during parsing", exc_info=exception)
        exit(-1)

    logger.info("Arguments successfully parsed")

    stop_event = Event()
    sensor_x_queues = [Queue(), Queue(), Queue()]
    sensor_cam_queue = Queue()

    sensor0_thread = Thread(target=SensorX.sensor_x_work,
                            args=(stop_event, 0.01, sensor_x_queues[0]))
    logger.info("Starting SensorX 1 thread")
    sensor0_thread.start()

    sensor1_thread = Thread(target=SensorX.sensor_x_work,
                            args=(stop_event, 0.1, sensor_x_queues[1]))
    logger.info("Starting SensorX 2 thread")
    sensor1_thread.start()

    sensor2_thread = Thread(target=SensorX.sensor_x_work,
                            args=(stop_event, 1, sensor_x_queues[2]))
    logger.info("Starting SensorX 3 thread")
    sensor2_thread.start()

    sensor_cam_thread = Thread(target=SensorCam.sensor_cam_work,
                               args=(stop_event, camera_name, cam_width, cam_height, sensor_cam_queue))
    logger.info("Starting Sensor Cam thread")
    sensor_cam_thread.start()

    logger.info("Starting Window Image")
    window_image = WindowImage(framerate)

    logger.info("Starting Frame Assembly...")
    image_processor = ImageProcessor(sensor_x_queues, sensor_cam_queue)

    while True:
        window_image.show(image_processor.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("exiting by q button")

            stop_event.set()

            sensor0_thread.join()
            sensor1_thread.join()
            sensor2_thread.join()
            sensor_cam_thread.join()

            logger.info("Window destroy")
            exit(0)

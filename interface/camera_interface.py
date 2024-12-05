import cv2
from queue import Queue
import threading
from config import *

class CameraManager:
    def __init__(self):
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.cap = None
        self._camera_window_name = CAMERA_WINDOW_NAME

    def cleanup(self):
        """清理资源"""
        print("开始清理摄像头资源...")
        self.running = False
        
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        
        print("摄像头资源清理完成")

    def camera_thread_func(self, camera_id=0):
        """摄像头捕获线程"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print("无法打开摄像头!")
                self.running = False
                return

            print("摄像头已启动,按'q'键退出")
            
            while self.running:
                if not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                
                if not self.running:
                    break
                    
                cv2.imshow(self._camera_window_name, frame)
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break

        except Exception as e:
            print(f"摄像头线程发生错误: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyWindow(self._camera_window_name)

    def get_current_frame(self):
        """获取当前帧"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def list_cameras(self):
        """列出所有可用摄像头"""
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            available_cameras.append(f"Camera {index}")
            cap.release()
            index += 1
        return available_cameras

    def select_camera(self, camera_id):
        """选择摄像头"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {camera_id}!")
            self.running = False
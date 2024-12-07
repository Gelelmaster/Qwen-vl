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
        self.camera_enabled = False
        self.selected_camera_id = 0
        self.init_camera()

    def init_camera(self):
        """初始化摄像头：检测并选择"""
        available_cameras = self.detect_cameras()
        if available_cameras:
            print("\n检测到以下摄像头:")
            for cam in available_cameras:
                print(f"[{cam['id']}] 分辨率: {cam['resolution']}")
            
            while True:
                choice = input("\n请选择要使用的摄像头编号 (或直接按回车跳过): ").strip()
                if not choice:  # 直接回车
                    print("跳过摄像头功能")
                    self.disable_camera()
                    break
                try:
                    camera_id = int(choice)
                    if any(cam['id'] == camera_id for cam in available_cameras):
                        if self.select_camera(camera_id):
                            print(f"已选择摄像头 {camera_id}")
                            break
                    else:
                        print("无效的摄像头编号，请重新选择")
                except ValueError:
                    print("请输入有效的数字")
        else:
            print("未检测到可用的摄像头，摄像头功能将被禁用")
            self.disable_camera()

    def detect_cameras(self):
        """检测可用的摄像头"""
        available_cameras = []
        for i in range(10):  # 检查前10个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 尝试读取一帧
                ret, _ = cap.read()
                if ret:
                    # 获取摄像头信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available_cameras.append({
                        'id': i,
                        'resolution': f"{width}x{height}"
                    })
                cap.release()
        return available_cameras

    def select_camera(self, camera_id):
        """选择指定的摄像头"""
        self.camera_enabled = True
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            self.disable_camera()
            return False
        return True

    def disable_camera(self):
        """禁用摄像头功能"""
        self.camera_enabled = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

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
        if not self.camera_enabled:
            return

        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                print("无法打开摄像头!")
                self.running = False
                return

            print("摄像头已启动,按'q'键退出")
            
            while self.running and self.camera_enabled:
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
        if not self.camera_enabled:
            return None
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
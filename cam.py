import cv2
import time
import numpy as np
from playsound import playsound

# 定义摄像头捕获器
cap = cv2.VideoCapture(0)

# 定义变量
last_frame = None
alarm_time = None
has_changed = False

# 持续监控画面
while True:
    # 读取当前画面
    ret, frame = cap.read()

    # 如果读取失败，跳过
    if not ret:
        continue

    # 调整画面大小
    frame = cv2.resize(frame, (640, 480))

    # 将画面转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 如果是第一帧，记录下来
    if last_frame is None:
        last_frame = gray
        continue

    # 计算当前画面和上一帧画面的差异
    frame_delta = cv2.absdiff(last_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果画面有变化，更新上一帧画面和警报时间，并标记为已有变化
    if len(contours) > 0:
        last_frame = gray
        alarm_time = None
        has_changed = True
    # 如果画面无变化，检查警报时间是否超过3分钟，如果是则触发警报，并标记为无变化
    elif alarm_time is None:
        alarm_time = time.time()
        has_changed = False
    elif time.time() - alarm_time > 180:
        playsound('alarm.wav')
        alarm_time = time.time()
        has_changed = False

    # 输出对比结果
    if has_changed:
        print('画面有变化')
    else:
        print('画面无变化')
    
    # 显示画面
    cv2.imshow('frame', frame)

    # 处理键盘输入
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

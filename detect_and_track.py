import cv2
from ultralytics import YOLO

# 1. تحميل الموديل الجديد (تأكد من المسار اللي طلع لك بعد التدريب)
model_path = r'C:\Users\Administrator\Desktop\Fish_Project_New 1\runs\train_final_7_classes\weights\best.pt'
model = YOLO(model_path)

# 2. تشغيل الكاميرا (0 هي كاميرا اللاب توب أو الـ USB)
cap = cv2.VideoCapture(0)

print("🚀 الموديل شغال.. اضغط 'q' للخروج")

while True:
    ret, frame = cap.read()
    if not ret: break

    # تشغيل التتبع (Tracking) - بيستخدم الـ GPU تلقائياً لو متوفر
    results = model.track(frame, persist=True, conf=0.4, iou=0.5)

    # رسم النتائج
    if results[0].boxes is not None:
        annotated_frame = results[0].plot() # هذه ترسم المربعات والأسماء تلقائياً لكل الأنواع الـ 7
        cv2.imshow("Sayda Project - PC Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
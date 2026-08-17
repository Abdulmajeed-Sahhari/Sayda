import cv2
from ultralytics import YOLO

# 1. تحميل الموديل
model_path = r'runs\detect\runs\train_final_7_classes\weights\best.pt'
model = YOLO(model_path)

# 2. تشغيل الكاميرا
cap = cv2.VideoCapture(0)

print("⚙️ جاري تسخين الموديل (Warm-up GPU)... لحظات بس")

# 3. قراءة 10 إطارات (9 للتسخين، والأخير للطباعة)
for i in range(10):
    ret, frame = cap.read()
    if ret:
        if i == 9:  # هذا الإطار العاشر اللي بنطبعه
            print("\n" + "="*70)
            print("🚀 --- جاري اختبار سرعة الموديل الفائقة (Inference Speed Proof) ---")
            # device=0 تجبره يستخدم كرت الشاشة
            results = model.predict(frame, verbose=True, device=0)
            print("="*70 + "\n")
        else:
            # الإطارات الأولى يحللها بصمت عشان يسخن
            model.predict(frame, verbose=False, device=0)
    else:
        print("❌ تعذر قراءة الكاميرا.")
        break

cap.release()
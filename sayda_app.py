import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# 1. تحميل الموديل (تأكدنا من المسار مسبقاً)
model_path = r'runs\detect\runs\train_final_v2\weights\best.pt'
model = YOLO(model_path)

# 2. تشغيل الكاميرا
cap = cv2.VideoCapture(1)

# 3. إعداد نافذة التطبيق (الواجهة)
root = tk.Tk()
root.title("مشروع صيدة - الذكاء الاصطناعي للتعرف على الأسماك")
root.geometry("1000x650")
root.configure(bg="#2c3e50") # خلفية داكنة فخمة (Dark Mode)

# متغير يحفظ السمكة اللي اخترتها (الافتراضي: الكل)
selected_fish = tk.StringVar(value="All")

# قائمة الأسماك + خيار الكل
fish_classes = ["All", "Tuna", "Agam", "Hamour", "Hamra", "Parrotfish", "Salmon", "Tilapia"]

# --- تصميم الواجهة ---

# القسم الأيسر: القائمة والأزرار
left_frame = tk.Frame(root, bg="#34495e", width=250)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

tk.Label(left_frame, text="🎯 لوحة التحكم", fg="white", bg="#34495e", font=("Arial", 16, "bold")).pack(pady=20)
tk.Label(left_frame, text="اختر الهدف:", fg="#bdc3c7", bg="#34495e", font=("Arial", 12)).pack(pady=5)

# إنشاء الأزرار (Radio Buttons) لكل سمكة
for fish in fish_classes:
    btn = tk.Radiobutton(left_frame, text=fish, variable=selected_fish, value=fish,
                         indicatoron=0, width=15, height=2, font=("Arial", 12, "bold"),
                         bg="#ecf0f1", selectcolor="#e74c3c", fg="#2c3e50", activebackground="#c0392b")
    btn.pack(pady=5)

# القسم الأيمن: شاشة الكاميرا
video_label = tk.Label(root, bg="black", bd=5, relief="ridge")
video_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

# 4. دالة تحديث الكاميرا والذكاء الاصطناعي
def update_frame():
    ret, frame = cap.read()
    if ret:
        # قلب الصورة زي المراية (عشان تكون أريح للعين)
        frame = cv2.flip(frame, 1)
        
        # تشغيل التتبع
        results = model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)
        target = selected_fish.get()

        if results[0].boxes is not None:
            if target == "All":
                # ارسم كل الأسماك
                frame = results[0].plot()
            else:
                # ارسم السمكة المختارة فقط
                for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                    class_name = model.names[int(cls)]
                    if class_name == target:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # رسم خلفية للاسم عشان يكون واضح
                        cv2.rectangle(frame, (x1, y1 - 35), (x1 + 150, y1), (0, 255, 0), -1)
                        cv2.putText(frame, class_name, (x1 + 5, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # تحويل الصورة من نظام OpenCV إلى نظام الواجهة (Tkinter)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # تحديث الشاشة
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # كرر العملية كل 15 ملي ثانية (لضمان سلاسة الفيديو)
    video_label.after(15, update_frame)

# 5. تشغيل الدورة الرئيسية للتطبيق
update_frame()
root.mainloop()

# إغلاق الكاميرا عند إغلاق البرنامج (علامة X)
cap.release()
cv2.destroyAllWindows()
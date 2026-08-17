import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import serial
import time

# ==============================
# إعداد الأردوينو
# ==============================
try:
    arduino = serial.Serial('COM3', 9600, timeout=0.1)
    time.sleep(2)
    arduino_connected = True
    print("✅ ممتاز! الأردوينو متصل في COM6 (سيرفو + حساس جاهزين).")
except:
    arduino_connected = False
    print("⚠️ خطأ: الأردوينو غير متصل في COM6! تأكد من الكيبل.")

# ==============================
# تحميل الموديل
# ==============================
model_path = r'runs\detect\runs\train_final_v2\weights\best.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

# ==============================
# متغيرات التحكم
# ==============================
trap_closed = False

# ==============================
# الواجهة (GUI)
# ==============================
root = tk.Tk()
root.title("مشروع صيدة - الذكاء الاصطناعي للتعرف على الأسماك")
root.geometry("1200x700")
root.configure(bg="#2c3e50")

selected_fish = tk.StringVar(value="All")
fish_classes = ["All", "Tuna", "Agam", "Hamour", "Hamra", "Parrotfish", "Salmon", "Tilapia"]

left_frame = tk.Frame(root, bg="#34495e", width=250)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

tk.Label(left_frame, text="🎯 لوحة التحكم", fg="white", bg="#34495e", font=("Arial", 16, "bold")).pack(pady=20)
tk.Label(left_frame, text="اختر الهدف:", fg="#bdc3c7", bg="#34495e", font=("Arial", 12)).pack(pady=5)

for fish in fish_classes:
    btn = tk.Radiobutton(left_frame, text=fish, variable=selected_fish, value=fish,
                         indicatoron=0, width=15, height=2, font=("Arial", 12, "bold"),
                         bg="#ecf0f1", selectcolor="#e74c3c", fg="#2c3e50", activebackground="#c0392b")
    btn.pack(pady=5)

status_label = tk.Label(left_frame, text="🔍 المصيدة مفتوحة، جاري البحث...", fg="white",
                         bg="#34495e", font=("Arial", 11, "bold"))
status_label.pack(pady=20)

distance_label = tk.Label(left_frame, text="📏 المسافة: جاري القراءة...", fg="#f1c40f",
                         bg="#34495e", font=("Arial", 14, "bold"))
distance_label.pack(pady=10)

video_label = tk.Label(root, bg="black", bd=5, relief="ridge")
video_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

# ==============================
# دالة إرسال أمر السيرفو
# ==============================
def send_to_arduino(signal: str):
    if arduino_connected:
        arduino.write(signal.encode())
        arduino.flush()

# ==============================
# دالة تحديث الكاميرا
# ==============================
def update_frame():
    global trap_closed
    
    # 1. قراءة المسافة من الحساس
    if arduino_connected and arduino.in_waiting > 0:
        try:
            arduino_data = arduino.readline().decode('utf-8').strip()
            if arduino_data.isdigit():
                dist = int(arduino_data)
                distance_label.config(text=f"📏 المسافة: {dist} سم")
        except:
            pass 

    # 2. معالجة الكاميرا والموديل
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    
    # ==============================
    # تقسيم الشاشة إلى 3 مناطق
    # ==============================
    h, w, _ = frame.shape
    line1_y = int(h * 0.33)  # الخط الأول (الثلث العلوي)
    line2_y = int(h * 0.66)  # الخط الثاني (الثلث السفلي)

    # رسم الخطوط الوهمية
    cv2.line(frame, (0, line1_y), (w, line1_y), (0, 255, 255), 2)
    cv2.line(frame, (0, line2_y), (w, line2_y), (0, 255, 255), 2)
    
    # كتابة أسماء المناطق
    cv2.putText(frame, "Open Zone (TOP)", (10, line1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "Catch Zone (MIDDLE)", (10, line1_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Open Zone (BOTTOM)", (10, line2_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    results = model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)
    target = selected_fish.get()
    
    fish_detected_now = False
    fish_cy = None 

    if results[0].boxes is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            class_name = model.names[int(cls)]
            if target == "All" or class_name == target:
                fish_detected_now = True
                x1, y1, x2, y2 = map(int, box)
                
                # حساب نقطة المنتصف للسمكة
                fish_cy = (y1 + y2) // 2 
                fish_cx = (x1 + x2) // 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (fish_cx, fish_cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, class_name, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                break 

    # ==============================
    # 3. حركة السيرفو بناءً على المناطق الثلاث
    # ==============================
    if fish_detected_now and fish_cy is not None:
        if fish_cy < line1_y:
            # المنطقة العلوية -> فتح السيرفو
            if trap_closed:
                send_to_arduino('0')
                trap_closed = False
                status_label.config(text="🔓 السمكة فوق! المصيدة انفتحت", fg="#2ecc71")
                
        elif line1_y <= fish_cy < line2_y:
            # المنطقة الوسطى (بين الخطين) -> قفل السيرفو (الصيد)
            if not trap_closed:
                send_to_arduino('1')
                trap_closed = True
                status_label.config(text="🔒 السمكة بالمنتصف! المصيدة مقفلة", fg="#e74c3c")
                
        else:
            # المنطقة السفلية (تحت الخط الثاني) -> فتح السيرفو
            if trap_closed:
                send_to_arduino('0')
                trap_closed = False
                status_label.config(text="🔓 السمكة نزلت! المصيدة انفتحت", fg="#2ecc71")

    # تحديث واجهة العرض
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(15, update_frame)

# ==============================
# إغلاق البرنامج
# ==============================
def on_close():
    cap.release()
    if arduino_connected:
        arduino.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# بداية التشغيل (فتح المصيدة)
send_to_arduino('0')
update_frame()
root.mainloop()
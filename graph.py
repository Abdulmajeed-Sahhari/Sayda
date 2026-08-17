import pandas as pd
import matplotlib.pyplot as plt
import os

# المسار الأكيد اللي طلعته أنت
results_csv = os.path.join('runs', 'detect', 'runs', 'train_final_7_classes', 'results.csv')
output_dir = 'Final_Presentation_Graphs'

print(f"🔍 أحاول قراءة الملف من: {results_csv}")

if not os.path.exists(results_csv):
    print("❌ الملف لسه غير موجود!")
    exit()

os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    # رسم منحنى الدقة
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], color='#2ecc71', linewidth=2, label='mAP@0.5')
    plt.fill_between(df['epoch'], df['metrics/mAP50(B)'], color='#2ecc71', alpha=0.1)
    plt.title('Model Accuracy Progress (mAP50)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '01_accuracy.png'))
    plt.close()

    # رسم الـ Pie Chart
    classes = ['Tuna', 'Agam', 'Hamour', 'Hamra', 'Parrotfish', 'Salmon', 'Tilapia']
    counts = [183, 177, 115, 43, 200, 200, 200] 
    
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=140)
    plt.title('Dataset Composition (7 Species)')
    plt.savefig(os.path.join(output_dir, '02_distribution.png'))
    plt.close()

    print(f"✅ الصور صارت جاهزة في مجلد: Final_Presentation_Graphs")

except Exception as e:
    print(f"❌ حدث خطأ غير متوقع: {e}")
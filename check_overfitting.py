import pandas as pd
import matplotlib.pyplot as plt
import os

# المسار الأكيد اللي طلعته أنت
results_csv = os.path.join('runs', 'detect', 'runs', 'train_final_7_classes', 'results.csv')
output_dir = 'Final_Presentation_Graphs'

print(f"🔍 جاري القراءة من: {results_csv}")

if not os.path.exists(results_csv):
    print("❌ الملف لسه غير موجود! تأكد من المسار.")
    exit()

os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # رسم منحنى خسارة الصندوق
    ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#e74c3c', linewidth=2)
    ax1.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss', color='#3498db', linewidth=2)
    ax1.set_title('Bounding Box Loss (Overfitting Check)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # رسم منحنى خسارة التصنيف
    ax2.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='#e74c3c', linewidth=2)
    ax2.plot(df['epoch'], df['val/cls_loss'], label='Validation Class Loss', color='#3498db', linewidth=2)
    ax2.set_title('Classification Loss (Overfitting Check)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, '03_overfitting_check.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 مبروك! الرسمة جاهزة في: {save_path}")

except Exception as e:
    print(f"❌ حدث خطأ أثناء الرسم: {e}")
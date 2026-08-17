import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 1. البحث عن أحدث ملف نتائج
search_path = os.path.join('runs', '**', 'results.csv')
all_results = glob.glob(search_path, recursive=True)

if not all_results:
    print("❌ لم أتمكن من العثور على ملف results.csv!")
    exit()

latest_results_csv = max(all_results, key=os.path.getmtime)
print(f"🔍 جاري قراءة البيانات من: {latest_results_csv}")

output_dir = 'Final_Presentation_Graphs'
os.makedirs(output_dir, exist_ok=True)

try:
    # 2. قراءة البيانات وتنظيف أسماء الأعمدة
    df = pd.read_csv(latest_results_csv)
    df.columns = df.columns.str.strip()

    # 3. قص البيانات عشان توقف عند 90 Epoch بس!
    df_90 = df[df['epoch'] <= 90]

    # 4. إعداد الرسمة (رسمة واحدة كبيرة ومفصلة)
    plt.figure(figsize=(10, 6))

    # رسم منحنى التدريب والاختبار
    plt.plot(df_90['epoch'], df_90['train/cls_loss'], label='Train Class Loss', color='#e74c3c', linewidth=2.5)
    plt.plot(df_90['epoch'], df_90['val/cls_loss'], label='Validation Class Loss', color='#3498db', linewidth=2.5)
    
    # تحسين شكل الرسمة
    plt.title('Classification Loss (Epochs 1 - 90)', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # إجبار محور السينات (X) إنه ينتهي عند 90
    plt.xlim(left=0, right=90)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4, linestyle='--')

    # 5. حفظ الصورة
    save_path = os.path.join(output_dir, '04_cls_loss_90.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 مبروك! تم رسم المنحنى إلى الدورة 90 وحفظه في: {save_path}")

except Exception as e:
    print(f"❌ حدث خطأ أثناء الرسم: {e}")
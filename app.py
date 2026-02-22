import streamlit as st
import pickle
import numpy as np

# تحميل الموديل الذكي
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ خطأ: لم يتم العثور على ملف الموديل. تأكد من رفعه مع الكود.")

# إعدادات الصفحة
st.set_page_config(page_title="فحص السكري الذكي", page_icon="🏥", layout="wide")

# تصميم الواجهة
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🏥 نظام التشخيص الذكي لمرض السكري</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>هذا النظام يعتمد على خوارزميات تعلم الآلة للتنبؤ باحتمالية الإصابة</p>", unsafe_allow_html=True)
st.markdown("---")

# تقسيم المدخلات في أعمدة
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("🤰 عدد مرات الحمل", min_value=0, max_value=20, value=0)
    glu = st.number_input("🩸 مستوى الجلوكوز (بعد الصيام)", min_value=0, max_value=300, value=100)
    bp = st.number_input("💓 ضغط الدم الانبساطي", min_value=0, max_value=150, value=70)
    skin = st.number_input("📏 سمك طبقة الجلد (mm)", min_value=0, max_value=100, value=20)

with col2:
    ins = st.number_input("💉 مستوى الأنسولين", min_value=0, max_value=900, value=80)
    bmi = st.number_input("⚖️ مؤشر كتلة الجسم (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("🧬 عامل وراثة السكري", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("📅 العمر", min_value=1, max_value=120, value=30)

st.markdown("---")

# تنفيذ التنبؤ
if st.button("🔍 إجراء الفحص الآن"):
    # ترتيب البيانات للموديل
    features = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1] * 100 # نسبة الاحتمالية

    if prediction[0] == 1:
        st.error(f"⚠️ النتيجة: المريض قد يكون مصاباً بالسكر بنسبة {prob:.1f}%")
        st.info("نصيحة: يرجى استشارة الطبيب للقيام بفحص السكر التراكمي.")
    else:
        st.success(f"✅ النتيجة: المريض سليم غالباً بنسبة {100-prob:.1f}%")
        st.balloons()
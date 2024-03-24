from flask import Flask, request, render_template
import joblib

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('iris_model.joblib')

app = Flask(__name__)

# สร้าง route หลักสำหรับหน้าเว็บ
@app.route('/')
def home():
    return render_template('index.html')

# สร้าง route สำหรับการทำนาย
@app.route('/predict', methods=['POST'])
def predict():
    # รับค่า features จากฟอร์ม
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    
    # ทำนายประเภทดอก IRIS
    prediction = model.predict(final_features)
    
    # แปลงผลลัพธ์เป็นชื่อประเภทดอก IRIS
    output = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}[prediction[0]]
    
    # ส่งผลลัพธ์กลับไปยังหน้าเว็บ
    return render_template('index.html', prediction_text=f'ดอก IRIS นี้เป็นประเภท: {output}')

if __name__ == "__main__":
    app.run(debug=True)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# โหลดชุดข้อมูล IRIS
iris = load_iris()
X, y = iris.data, iris.target

# แบ่งชุดข้อมูลเป็นชุดฝึกหัดและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# บันทึกโมเดลเป็นไฟล์
joblib.dump(clf, 'iris_model.joblib')

print("โมเดลได้ถูกสร้างและบันทึกเป็นไฟล์ 'iris_model.joblib' แล้ว")
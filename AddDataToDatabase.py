import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-9fe8d-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "222631141":
        {
            "name": "Nguyen Mai Thanh",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 6,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2024-01-11 13:54:34"
        },
    "222611080":
        {
            "name": "Tran Tien Dung",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 16,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2024-01-12 09:25:24"
        }
}

for key, value in data.items():
    ref.child(key).set(value)

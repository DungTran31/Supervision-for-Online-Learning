import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-9fe8d-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "222611080":
        {
            "name": "Tran Tien Dung",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 16,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2024-01-12 09:25:24"
        },
    "222631105":
        {
            "name": "Vu Quang Huy",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 2,
            "standing": "N",
            "year": 2,
            "last_attendance_time": "2024-03-11 13:04:24"
        },
    "222631124":
        {
            "name": "Nguyen Minh",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 6,
            "standing": "B",
            "year": 2,
            "last_attendance_time": "2024-01-12 15:25:34"
        },
    "222631132":
        {
            "name": "Nguyen Minh Quan",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 10,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2024-02-12 09:25:44"
        },
    "222631141":
        {
            "name": "Nguyen Mai Thanh",
            "major": "Information Technology",
            "starting_year": 2022,
            "total_attendance": 12,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2024-01-12 09:35:22"
        }
}

for key, value in data.items():
    ref.child(key).set(value)

import csv
from sklearn.preprocessing import LabelEncoder
from flask import Flask,render_template,request,redirect,url_for,flash
import numpy as np
import mysql.connector
import cv2,os
import pandas as pd
from PIL import Image
import datetime
import time
import math
import pickle
app=Flask(__name__)
app.config['SECRET_KEY']='attendance system'
mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="smart_attendance")
cursor = mydb.cursor()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/collection')
def collection():
    return render_template('collection.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/blog')
def blog():
    return render_template('blog.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/single')
def single():
    return render_template('single.html')

@app.route("/addback", methods=['POST','GET'])
def addback():
    if request.method=='POST':
        Id=request.form['rno']
        name=request.form['name']
        print(type(Id))
        print(type(name))
        if not Id:
           flash("Please enter roll number properly ","warning")
           return render_template('collection.html')


        elif not name:
            flash("Please enter your name properly ", "warning")
            return render_template('collection.html')

        # elif (Id.isalpha() and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        df = pd.read_csv("Student_Details/StudentDetails.csv")
        val = df.Id.values
        if Id in str(val):
            flash("Roll already exists", "danger")
            return render_template("index.html")

        else:
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # incrementing sample number
                    sampleNum = sampleNum + 1
                        # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage/ " + name + "." + Id + '.' + str(
                            sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                        # display the frame

                else:
                    cv2.imshow('frame', img)
                    # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                    # break if the sample number is morethan 100
                elif sampleNum > 100:
                        break
        cam.release()
        cv2.destroyAllWindows()
        # res = "Roll Number : " + Id + " Name : " + name
        row = [Id, name]
        with open('Student_Details/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        flash("Captured images successfully!!","success")
        return render_template("collection.html")

    return render_template("collection.html")
@app.route('/training')
def training():
    le = LabelEncoder()
    faces, Id = getImagesAndLabels("TrainingImage")
    Id = le.fit_transform(Id)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"Trained_Model\Trainner.yml")

    flash("Model Trained Successfully", "success")
    return render_template('index.html')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

@app.route('/prediction')
def prediction():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"Trained_Model\Trainner.yml")
    harcascadePath = r"Haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv(r"Student_Details\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    # print(df)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['First Name', 'Last Name']
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    det=0
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            # print(conf)
            if (conf < 50):
                det += 1
                tt = le.inverse_transform([Id])
                # print(type(tt))
                tt = tt[0]
                rno=str(tt)
                # print(rno)
                print(det)
                if det==40:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M')
                    in_time = int(timeStamp.split(':')[0]) * 60
                    # in_time=400
                    print(in_time)
                    sql = "select count(*) from attendance where rno='%s' and date1='%s'" % (rno, date)
                    z = pd.read_sql_query(sql, mydb)
                    count = z.values[0][0]
                    if 600 >= in_time:# Befor 10
                        print("hhhhhhhhhh")
                        if count == 0:
                            status='Early Coming'
                            sql = "insert into attendance(rno,in_time,in_status,date1) values(%s,%s,%s,%s)"
                            val = (rno, timeStamp, status, date)
                            cursor.execute(sql, val)
                            mydb.commit()
                        else:
                            sql = "select in_time from attendance where rno='%s' and date1='%s'" % (rno, date)
                            result = pd.read_sql_query(sql, mydb)
                            in_time1 = result.values[0][0]
                            in_min = int(in_time1.split(':')[0]) * 60
                            in_min = in_min + int(in_time1.split(':')[1])
                            timeStamp.split(':')
                            out_min = int(timeStamp.split(':')[0]) * 60
                            out_min = out_min + int(timeStamp.split(':')[1])
                            work_time = out_min - in_min
                            work_time1 = math.floor(work_time / 60)
                            work_time2 = work_time % 60
                            w_full = str(work_time1) + ':' + str(work_time2)
                            s = "Early Out"
                            sq = "update attendance set out_time='%s',out_status='%s',overall_time='%s' where rno='%s' and date1='%s'" % (
                            timeStamp, s, w_full, rno, date)
                            cursor.execute(sq)
                            mydb.commit()
                    elif 600 < in_time and 720 >= in_time: #after 10-12
                        print("ddddddddddd")
                        if count == 0:
                            status = 'Late Coming'
                            sql= "insert into attendance(rno,in_time,in_status,date1) values(%s,%s,%s,%s)"
                            val = (rno, timeStamp, status, date)
                            cursor.execute(sql, val)
                            mydb.commit()
                        else:
                            sql = "select in_time from attendance where rno='%s' and date1='%s'" % (rno, date)
                            result = pd.read_sql_query(sql, mydb)
                            in_time1 = result.values[0][0]
                            in_min = int(in_time1.split(':')[0]) * 60
                            in_min = in_min + int(in_time1.split(':')[1])
                            timeStamp.split(':')
                            out_min = int(timeStamp.split(':')[0]) * 60
                            out_min = out_min + int(timeStamp.split(':')[1])
                            work_time = out_min - in_min
                            work_time1 = math.floor(work_time / 60)
                            work_time2 = work_time % 60
                            w_full = str(work_time1) + ':' + str(work_time2)
                            s = "Early Out"
                            sq = "update attendance set out_time='%s',out_status='%s',overall_time='%s' where rno='%s' and date1='%s'" % (
                                timeStamp, s, w_full, rno, date)
                            cursor.execute(sq)
                            mydb.commit()

                    elif 720 <= in_time and 960 >= in_time:#12-4
                        print("lllllllllllllllll")
                        # sql="select count(*) from attendance where rno='%s' and date1='%s'"%(rno,date)
                        # x=pd.read_sql_query(sql,mydb)
                        # count=x.values[0][0]
                        if count==0:
                            print("kkkkkkkkkkkkkkk")
                            status = 'Afternoon Come'
                            sql = "insert into attendance(rno,in_time,in_status,date1) values(%s,%s,%s,%s)"
                            val = (rno, timeStamp, status, date)
                            cursor.execute(sql, val)
                            mydb.commit()
                        else:
                            sql = "select in_time from attendance where rno='%s' and date1='%s'" % (rno, date)
                            result = pd.read_sql_query(sql, mydb)
                            in_time1 = result.values[0][0]
                            in_min = int(in_time1.split(':')[0]) * 60
                            in_min = in_min + int(in_time1.split(':')[1])
                            timeStamp.split(':')
                            out_min = int(timeStamp.split(':')[0]) * 60
                            out_min = out_min + int(timeStamp.split(':')[1])
                            work_time = out_min - in_min
                            work_time1 = math.floor(work_time / 60)
                            work_time2 = work_time % 60
                            w_full = str(work_time1) + ':' + str(work_time2)
                            print("uuuuuuuuuuuuuuuuuu")
                            s="Early Out"
                            sq = "update attendance set out_time='%s',out_status='%s',overall_time='%s' where rno='%s' and date1='%s'" % (
                                timeStamp, s, w_full, rno, date)
                            cursor.execute(sq)
                            mydb.commit()
                    else:#after 4
                        sql = "select in_time from attendance where rno='%s' and date1='%s'" % (rno, date)
                        result = pd.read_sql_query(sql, mydb)
                        in_time1 = result.values[0][0]
                        in_min = int(in_time1.split(':')[0]) * 60
                        in_min = in_min + int(in_time1.split(':')[1])
                        timeStamp.split(':')
                        out_min = int(timeStamp.split(':')[0]) * 60
                        out_min = out_min + int(timeStamp.split(':')[1])
                        work_time = out_min - in_min
                        work_time1 = math.floor(work_time / 60)
                        work_time2 = work_time % 60
                        w_full = str(work_time1) + ':' + str(work_time2)
                        print("aaaaaaaaaaaaaaaa")
                        s = "Perfect Out"
                        sq = "update attendance set out_time='%s',out_status='%s',overall_time='%s' where rno='%s' and date1='%s'" % (
                            timeStamp, s, w_full, rno, date)
                        cursor.execute(sq)
                        mydb.commit()
                    det=0

            else:
                Id = 'Unknown'
                tt = str(Id)
                print(tt)
            if (conf > 55):
                noOfFile = len(os.listdir("ImageUnknown")) + 1
                cv2.imwrite(r"ImageUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            # print("tt:",str(tt))
            # print("x:", str(x))
            # print("y:", str(y))
            # print("font:", str(font))
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
    flash("Attendance taken","success")
    return render_template('index.html')

@app.route('/viewdata', methods=['POST','GET'])
def viewdata():
    if request.method=='POST':
        rno=request.form['rno']
    sql="select * from attendance where rno='%s' "%(rno)
    x=pd.read_sql_query(sql,mydb)
    x=x.drop(['id','rno'], axis=1)
    return render_template('single.html',row_val=x.values.tolist())

if __name__=='__main__':
    app.run(debug=True)
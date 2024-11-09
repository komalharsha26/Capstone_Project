from flask import Flask,render_template,request
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor
import catboost as cb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor




mydb = mysql.connector.connect(host='localhost',user='root',password='',port='3306',database='real_time')
cur = mydb.cursor()


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        psw = request.form['password']
        sql = "SELECT * FROM blood WHERE Email=%s and Password=%s"
        val = (email, psw)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loginhome.html', msg='login succesful')
        else:
            return render_template('login.html', msg='Invalid Credentias')

    return render_template('login.html')

@app.route('/loginhome')
def loginhome():
    return render_template('loginhome.html')

@app.route('/registration',methods=['GET','POST'])
def registration():

    if request.method == "POST":
        print('a')
        name = request.form['name']
        print(name)
        email = request.form['email']
        pws = request.form['psw']
        print(pws)
        cpws = request.form['cpsw']
        if pws == cpws:
            sql = "select * from blood"
            print('abcccccccccc')
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('registration.html', msg='a')
            else:
                sql = "INSERT INTO blood(name,email,password) values(%s,%s,%s)"
                values = (name, email, pws)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('registration.html', msg='success')
        else:
            return render_template('registration.html', msg='repeat')

    return render_template('registration.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        print(file)
        global df
        df = pd.read_csv(file)
        print(df)
        return render_template('upload.html', columns=df.columns.values, rows=df.values.tolist(),msg='success')
    return render_template('upload.html')

@app.route('/viewdata')
def viewdata():
    print(df.columns)
    df_sample = df.head(100)
    return render_template('viewdata.html', columns=df_sample.columns.values, rows=df_sample.values.tolist())


@app.route('/preprocessing',methods=['POST','GET'])
def preprocessing():
    global X, y, X_train, X_test, y_train, y_test
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 10
        print(size)
        df.drop(['Time', 'Clock'], axis=1, inplace=True)
        df.Pulse.fillna(value=df.Pulse.mode()[0], inplace=True)
        df.SpO2.fillna(value=df.SpO2.mode()[0], inplace=True)
        df.Perf.fillna(value=df.Perf.mode()[0], inplace=True)
        df.awRR.fillna(value=df.awRR.mode()[0], inplace=True)
        df['NBP (Sys)'].fillna(value=df['NBP (Sys)'].mode()[0], inplace=True)
        df['NBP (Dia)'].fillna(value=df['NBP (Dia)'].mode()[0], inplace=True)
        df['NBP (Mean)'].fillna(value=df['NBP (Mean)'].mode()[0], inplace=True)
        df.etSEV.fillna(value=df.etSEV.mode()[0], inplace=True)
        df.inSEV.fillna(value=df.inSEV.mode()[0], inplace=True)
        df['Pleth'].fillna(value=df['Pleth'].mode()[0], inplace=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(X_train)
        print(X_train.columns)
        return render_template('preprocessing.html', msg='Data Preprocessed and It Splits Succesfully')

    return render_template('preprocessing.html')



@app.route('/model',methods=['POST','GET'])
def model():
    if request.method=='POST':
        models = int(request.form['algo'])
        if models==1:
            print("==")
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = r2_score(y_pred, y_test)
            dts = mean_squared_error(y_test, y_pred, squared=False)
            msg = 'Accuracy  for Decision Tree is ' + str(acc)
            a = 'mean_squared_error  for Decision Tree is ' + str(dts)
        elif models== 2:
            print("======")
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = r2_score(y_pred, y_test)
            rfs = mean_squared_error(y_test, y_pred, squared=False)
            msg = 'Accuracy  for Random Forest is ' + str(acc)
            a = 'mean_squared_error  for Random Forest is '  + str(rfs)
            return render_template('model.html', msg=msg)
        return render_template('model.html')


if __name__=="__main__":
    app.run(debug=True)
from flask import Flask,render_template,request
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import numpy as np
import sklearn_relief as relief
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


mydb = mysql.connector.connect(host='localhost',user='root',password='',port='3306',database='cyber_threat')
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
        sql = "SELECT * FROM cyber WHERE Email=%s and Password=%s"
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
            sql = "select * from cyber"
            print('abcccccccccc')
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('registration.html', msg='email id already exist')
            else:
                sql = "INSERT INTO cyber(name,email,password) values(%s,%s,%s)"
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
        size = size / 100
        print(size)
        # df.RtpStateBitfield.fillna(value=df.RtpStateBitfield.mode()[0], inplace=True)
        #
        # df.DefaultBrowsersIdentifier.fillna(value=df.DefaultBrowsersIdentifier.mode()[0], inplace=True)
        #
        # df.AVProductStatesIdentifier.fillna(value=df.AVProductStatesIdentifier.mode()[0], inplace=True)
        #
        # df.AVProductsInstalled.fillna(value=df.AVProductsInstalled.mode()[0], inplace=True)
        #
        # df.CityIdentifier.fillna(value=df.CityIdentifier.mode()[0], inplace=True)
        #
        # df.OrganizationIdentifier.fillna(value=df.OrganizationIdentifier.mode()[0], inplace=True)
        #
        # df.GeoNameIdentifier.fillna(value=df.GeoNameIdentifier.mode()[0], inplace=True)
        #
        # df.IsProtected.fillna(value=df.IsProtected.mode()[0], inplace=True)
        #
        # df.SMode.fillna(value=df.SMode.mode()[0], inplace=True)
        #
        # df.Firewall.fillna(value=df.Firewall.mode()[0], inplace=True)
        #
        # df.UacLuaenable.fillna(value=df.UacLuaenable.mode()[0], inplace=True)
        #
        # df.Census_OEMNameIdentifier.fillna(value=df.Census_OEMNameIdentifier.mode()[0], inplace=True)
        #
        # df.Census_OEMModelIdentifier.fillna(value=df.Census_OEMModelIdentifier.mode()[0], inplace=True)
        #
        # df.Census_ProcessorCoreCount.fillna(value=df.Census_ProcessorCoreCount.mode()[0], inplace=True)
        #
        # df.Census_ProcessorManufacturerIdentifier.fillna(value=df.Census_ProcessorManufacturerIdentifier.mode()[0],
        #                                                  inplace=True)
        #
        # df.Census_PrimaryDiskTotalCapacity.fillna(value=df.Census_PrimaryDiskTotalCapacity.mode()[0], inplace=True)
        #
        # df.Census_SystemVolumeTotalCapacity.fillna(value=df.Census_SystemVolumeTotalCapacity.mode()[0], inplace=True)
        #
        # df.Census_InternalPrimaryDiagonalDisplaySizeInInches.fillna(
        #     value=df.Census_InternalPrimaryDiagonalDisplaySizeInInches.mode()[0], inplace=True)
        #
        # df.Census_InternalPrimaryDisplayResolutionHorizontal.fillna(
        #     value=df.Census_InternalPrimaryDisplayResolutionHorizontal.mode()[0], inplace=True)
        #
        # df.Census_InternalPrimaryDisplayResolutionVertical.fillna(
        #     value=df.Census_InternalPrimaryDisplayResolutionVertical.mode()[0], inplace=True)
        #
        # df.Wdft_IsGamer.fillna(value=df.Wdft_IsGamer.mode()[0], inplace=True)
        #
        # df.Wdft_RegionIdentifier.fillna(value=df.Wdft_RegionIdentifier.mode()[0], inplace=True)
        #
        # df.AVProductsEnabled.fillna(value=df.AVProductsEnabled.mode()[0], inplace=True)
        #
        # df.PuaMode.fillna(value=df.PuaMode.mode()[0], inplace=True)
        #
        # df.IeVerIdentifier.fillna(value=df.IeVerIdentifier.mode()[0], inplace=True)
        #
        # df.Census_ProcessorModelIdentifier.fillna(value=df.Census_ProcessorModelIdentifier.mode()[0], inplace=True)
        #
        # df.Census_ProcessorClass.fillna(value=df.Census_ProcessorClass.mode()[0], inplace=True)
        #
        # df.Census_OSInstallLanguageIdentifier.fillna(value=df.Census_OSInstallLanguageIdentifier.mode()[0],
        #                                              inplace=True)
        #
        # df.Census_IsFlightingInternal.fillna(value=df.Census_IsFlightingInternal.mode()[0], inplace=True)
        #
        # df.Census_IsFlightsDisabled.fillna(value=df.Census_IsFlightsDisabled.mode()[0], inplace=True)
        #
        # df.Census_ThresholdOptIn.fillna(value=df.Census_ThresholdOptIn.mode()[0], inplace=True)
        #
        # df.Census_FirmwareManufacturerIdentifier.fillna(value=df.Census_FirmwareManufacturerIdentifier.mode()[0],
        #                                                 inplace=True)
        #
        # df.Census_FirmwareVersionIdentifier.fillna(value=df.Census_FirmwareVersionIdentifier.mode()[0], inplace=True)
        #
        # df.Census_IsWIMBootEnabled.fillna(value=df.Census_IsWIMBootEnabled.mode()[0], inplace=True)
        #
        # df.Census_IsVirtualDevice.fillna(value=df.Census_IsVirtualDevice.mode()[0], inplace=True)
        #
        # df.Census_IsAlwaysOnAlwaysConnectedCapable.fillna(value=df.Census_IsAlwaysOnAlwaysConnectedCapable.mode()[0],
        #                                                   inplace=True)

        # df.drop(['OsVer'], axis=1, inplace=True)
        #
        # df.Census_TotalPhysicalRAM.fillna(value=df.Census_TotalPhysicalRAM.mode()[0], inplace=True)
        # df.drop(['OsPlatformSubRelease'], axis=1, inplace=True)
        #
        # df.drop(['PuaMode'], axis=1, inplace=True)
        #
        # df.drop(['Census_ProcessorClass'], axis=1, inplace=True)


        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)
        a = np.array(X_train)

        # b = np.array(y_train)
        # r = relief.Relief(n_features=25)

        # my_transformed = r.fit_transform(a, b)
        # col = ['IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode',
        #        'DefaultBrowsersIdentifier', 'AVProductStatesIdentifier',
        #        'AVProductsInstalled', 'HasTpm', 'CountryIdentifier', 'CityIdentifier',
        #        'OrganizationIdentifier', 'GeoNameIdentifier',
        #        'LocaleEnglishNameIdentifier', 'OsBuild', 'OsSuite', 'IsProtected', 'SMode', 'Firewall',
        #        'Census_InternalPrimaryDiagonalDisplaySizeInInches', 'Census_InternalPrimaryDisplayResolutionHorizontal',
        #        'Census_InternalPrimaryDisplayResolutionVertical',
        #        'Census_OSBuildNumber', 'Census_OSBuildRevision',
        #        'Census_OSInstallLanguageIdentifier',
        #        'Wdft_IsGamer', 'Wdft_RegionIdentifier']

        # z = pd.DataFrame(my_transformed, columns=col)
        global x_trains, x_tests, y_trains, y_tests
        # x_trains, x_tests, y_trains, y_tests = train_test_split(z, b, test_size=0.30, random_state=72)
        print(X_train)
        print(X_train.columns)
        return render_template('preprocessing.html', msg='Data Preprocessed and It Splits Succesfully')

    return render_template('preprocessing.html')



@app.route('/model',methods=['POST','GET'])
def model():
    # filename1 = 'models/cb_pickle.sav'
    # filename2 = 'models/dt_pickle.sav'
    # filename3 = 'models/gb_pickle.sav'
    # filename4 = 'models/lr_pickle.sav'
    # filename5 = 'models/rf_pickle.sav'
    # filename6 = 'models/sv_pickle.sav'
    global X_train, X_test, y_train, y_test
    if request.method=='POST':
        models = int(request.form['algo'])
        if models==1:
            print("==")
            rf = RandomForestClassifier(min_samples_leaf=2, random_state=10, ccp_alpha=0.4)
            rf.fit(X_train[:100], y_train[:100])
            rfp = rf.predict(X_test[:100])
            rfa = accuracy_score(y_test[:100], rfp)
            msg = 'Accuracy  for Random Forest  is ' + str(rfa)
        elif models== 2:
            print("======")
            dt = DecisionTreeClassifier(min_samples_leaf=2, random_state=10, ccp_alpha=0.19)
            dt.fit(X_train[:100], y_train[:100])
            dtp = dt.predict(X_test[:100])
            dta = accuracy_score(y_test[:100], dtp)
            print(dta)
            msg = 'Accuracy  for DecisionTreeClassifier is ' + str(dta)
        elif models==3:
            print("===============")
            gb = GradientBoostingClassifier(min_samples_leaf=2, random_state=10, ccp_alpha=0.4)
            gb.fit(X_train[:100], y_train[:100])
            gbp = gb.predict(X_test[:100])
            gba = accuracy_score(y_test[:100], gbp)
            print(gba)
            msg = 'Accuracy  for GradientBoostingClassifier is ' + str(gba)
        elif models==4:
            print("===============")
            lr = LogisticRegression(random_state=15)
            lr.fit(X_train[:100], y_train[:100])
            lrp = lr.predict(X_test[:100])
            lra = accuracy_score(y_test[:100], lrp)
            print(lra)
            msg = 'Accuracy  for Logistic Regression is ' + str(lra)

        elif models==5:
            print("===============")
            cb = CatBoostClassifier(depth=5,random_state=15,min_data_in_leaf=50)
            cb.fit(X_train[:100], y_train[:100])
            cbp = cb.predict(X_test[:100])
            cba = accuracy_score(y_test[:100], cbp)
            print(cba)
            msg = 'Accuracy  for CatBoost Classifier is ' + str(cba)

        elif models ==6:
            print("===============")
            sv = SVC(random_state=10)
            sv.fit(X_train[:100], y_train[:100])
            svp = sv.predict(X_test[:100])
            sva = accuracy_score(y_test[:100], svp)
            print(sva)
            msg = 'Accuracy  for SVM is ' + str(sva)

        elif models ==7:
            print("===============")
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier(random_state=10)
            adb.fit(X_train[:100],y_train[:100])
            adp = adb.predict(X_test[:100])
            ada = accuracy_score(y_test[:100], adp)
            print(ada)
            msg = 'Accuracy  for AdaBoostClassifier is ' + str(ada)

        elif models ==8:
            print("===============")
            from sklearn.ensemble import ExtraTreesClassifier
            etc = ExtraTreesClassifier(random_state=10)
            etc.fit(X_train[:100], y_train[:100])
            etp = etc.predict(X_test[:100])
            eta = accuracy_score(y_test[:100], etp)
            print(eta)
            msg = 'Accuracy  for AdaBoostClassifier is ' + str(eta)


        return render_template('model.html',msg=msg)
    return render_template('model.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    print('111111')
    if  request.method == 'POST':
        print('2222')
        IsBeta = request.form['IsBeta']

        RtpStateBitfield = request.form['RtpStateBitfield']

        IsSxsPassiveMode =request.form['IsSxsPassiveMode']

        DefaultBrowsersIdentifier =request.form['DefaultBrowsersIdentifier']

        AVProductStatesIdentifier = request.form['AVProductStatesIdentifier']

        AVProductsInstalled = request.form['AVProductsInstalled']

        HasTpm = request.form['HasTpm']

        CountryIdentifier = request.form['CountryIdentifier']

        CityIdentifier = request.form['CityIdentifier']

        OrganizationIdentifier = request.form['OrganizationIdentifier']

        GeoNameIdentifier = request.form['GeoNameIdentifier']

        LocaleEnglishNameIdentifier = request.form['LocaleEnglishNameIdentifier']

        OsBuild = request.form['OsBuild']

        OsSuite = request.form['OsSuite']

        IsProtected = request.form['IsProtected']

        SMode = request.form['SMode']

        Firewall = request.form['Firewall']

        Census_InternalPrimaryDiagonalDisplaySizeInInches = request.form['Census_InternalPrimaryDiagonalDisplaySizeInInches']

        Census_InternalPrimaryDisplayResolutionHorizontal = request.form['Census_InternalPrimaryDisplayResolutionHorizontal']

        Census_InternalPrimaryDisplayResolutionVertical = request.form['Census_InternalPrimaryDisplayResolutionVertical']

        Census_OSBuildNumber = request.form['Census_OSBuildNumber']

        Census_OSBuildRevision = request.form['Census_OSBuildRevision']

        Census_OSInstallLanguageIdentifier = request.form['Census_OSInstallLanguageIdentifier']

        Wdft_IsGamer = request.form['Wdft_IsGamer']

        Wdft_RegionIdentifier = request.form['Wdft_RegionIdentifier']

        m = [IsBeta, RtpStateBitfield, IsSxsPassiveMode,DefaultBrowsersIdentifier, AVProductStatesIdentifier, AVProductsInstalled, HasTpm, CountryIdentifier, CityIdentifier,
               OrganizationIdentifier, GeoNameIdentifier, LocaleEnglishNameIdentifier, OsBuild, OsSuite, IsProtected, SMode, Firewall, Census_InternalPrimaryDiagonalDisplaySizeInInches, Census_InternalPrimaryDisplayResolutionHorizontal,
               Census_InternalPrimaryDisplayResolutionVertical,Census_OSBuildNumber, Census_OSBuildRevision,Census_OSInstallLanguageIdentifier,Wdft_IsGamer, Wdft_RegionIdentifier]
        model = CatBoostClassifier()
        model.fit(X_train[:100],y_train[:100])
        result = model.predict([m])
        print(result)
        if result == 0:
           msg = '<b>Cyber <span style = color:red;>ATTACKS</span></b>'
        elif result == 1:
            msg = '<b>Not Cyber<span style = color:red;> ATTACKS</span></b>'
        return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')

@app.route("/graph",methods=['GET','POST'])
def graph():
    return render_template('graph.html')


if __name__=="__main__":
    app.run(debug=True)
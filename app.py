import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('Insurance_Claim_model.pickle','rb'))

df_smoteen= pd.read_csv('Data/data_smoteen.csv')  

X = df_smoteen.iloc[:, :-1]
y = df_smoteen.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train.loc[:,:] = sc.fit_transform(X_train.loc[:,:])
X_test.loc[:,:] = sc.transform(X_test.loc[:,:])


@app.route('/')
def home():
    return render_template('h2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]

    i1 = int(int_features[0:1][0])
    i2 = int(int_features[1:2][0])
    i3 = int(int_features[2:3][0])
    i4 = float(int_features[3:4][0])
    i5 = float(int_features[4:5][0])
    i6 = int(int_features[5:6][0])

    final_features1 = [i1,i2,i3,i4,i5,i6]

    agency = int_features[6:7]
    i7, i8, i9 = 0,0,0
    if agency == 'CWT':
        i7 = 1
    elif agency == 'EPX':
        i8 = 1
    elif agency == 'OTH':
        i9 = 1
    final_features2 = [i7,i8,i9]

    i10, i11, i12, i13, i14, i15= 0,0,0,0,0,0
    ProductName = int_features[7:8]
    if ProductName == '2 way Comprehensive Plan':
        i10 = 1
    elif ProductName == 'Basic Plan':
        i11 = 1
    elif ProductName == 'Bronze Plan':
        i12 = 1
    elif ProductName == 'Cancellation Plan':
        i13 = 1
    elif ProductName == 'Other':
        i14 = 1
    elif ProductName == 'Rental Vehicle Excess Insurance':
        i15 = 1
    final_features3 = [i10, i11, i12, i13, i14, i15]


    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    Destination = int_features[8:]
    if Destination == 'CHINA':
        i16 = 1
    elif Destination == 'HONG KONG':
        i17 = 1
    elif Destination == 'INDIA':
        i18 = 1
    elif Destination == 'INDONESIA':
        i19 = 1
    elif Destination == 'JAPAN':
        i20 = 1
    elif Destination == 'KOREA, REPUBLIC OF':
        i21 = 1
    elif Destination == 'MALAYSIA':
        i22 = 1
    elif Destination == 'OTHER':
        i23 = 1
    elif Destination == 'PHILIPPINES':
        i24 = 1
    elif Destination == 'SINGAPORE':
        i25 = 1
    if Destination == 'TAIWAN, PROVINCE OF CHINA':
        i26 = 1
    elif Destination == 'THAILAND':
        i27 = 1
    elif Destination == 'UNITED KINGDOM':
        i28 = 1
    elif Destination == 'UNITED STATES':
        i29 = 1
    elif Destination == 'VIET NAM':
        i30 = 1
    final_features4 = [i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30]

    f = []

    for i in final_features1:
        f.append(i)

    for i in final_features2:
        f.append(i)

    for i in final_features3:
        f.append(i)

    for i in final_features4:
        f.append(i)

    # assign values to lists.  
    data = [{'i1':i1,'i2':i2,'i3':i3,'i4':i4,'i5':i5,'i6':i6,'i7':i7,'i8':i8,'i9':i9,'i10':i10, 'i11':i11, 'i12':i12, 'i13':i13, 'i14':i14, 'i15':i15, 'i16':i16, 'i17':i17, 'i18':i18, 'i19':i19, 'i20':i20, 'i21':i21, 'i22':i22, 'i23':i23, 'i24':i24, 'i25':i25, 'i26':i26, 'i27':i27, 'i28':i28, 'i29':i29, 'i30':i30}] 
  
    # Creates DataFrame.  
    data_new = pd.DataFrame(data)
    
    ee = df_smoteen.columns

    data_new.columns = list(ee[:-1])

    data_new.loc[:,:] = sc.transform(data_new.loc[:,:])

    prediction = model.predict(data_new)

    output = prediction[0]

    #e = [1,2,3]

    #output = pd.DataFrame(data=[f])
    #output = f

    if output == 0 :
        output = "No"
    else:
        output = "Yes"
    

    return render_template('h2.html', prediction_text='Insurance Claim Status : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
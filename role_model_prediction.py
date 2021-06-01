# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:51:18 2021

@author: hp
"""

import numpy as np
import pandas
import re
from bpemb import BPEmb
import heapq
from geopy.geocoders import Nominatim
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
import mysql.connector
from pydantic import BaseModel

mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ayushi23@",
        database="user_role_model"
        )

mycursor = mydb.cursor()
app = FastAPI()

class User(BaseModel):
    Name : str
    address : str
    gender : str
    interest : str
    specialization : str
    hobbies : str

@app.get('/')
def index():
    value = mycursor.execute("SELECT * FROM predicted_rolemodels WHERE id=(SELECT max(id) FROM predicted_rolemodels)")
    value = mycursor.fetchall()
    return value

@app.post('/user_id/{i}')
def read_index(i: int):
    print(i)
    mycursor.execute("SELECT count(*) from predicted_rolemodels")
    iid = mycursor.fetchall()
    if i>iid:
        return "No user found, please make new entry"
    value = mycursor.execute("SELECT * FROM predicted_rolemodels WHERE id = i")
    value = mycursor.fetchall()
    return value

@app.post('/model_prediction')
def model_predict(user : User):
    id,r1,r2,r3,r4,r5 = myfunction(user.Name, user.address, user.gender, user.interest, user.specialization, user.hobbies)
    pred = {'id' : id, 'Name' : user.Name, 'Role Model 1' : r1, 'Role Model 2' : r2, 'Role Model 3' : r3, 'Role Model 4' : r4, 'Role Model 5' : r5}
    #print(mycursor.last_insert_id())
    return pred

def get_location(address):
    geolocator = Nominatim(user_agent="Ayushi")
    location = geolocator.geocode(address)
    while not location:
        address = address.split(' ')[1:]
        address = ' '.join(address)
        location = geolocator.geocode(address)
    Latitude = str(location.latitude)
    Longitude = str(location.longitude)

    location = geolocator.reverse(Latitude+","+Longitude)
  
    address = location.raw['address']
  
    # traverse the data
    #city = address.get('city', '')
    #state = address.get('state', '')
    country = address.get('country', '')
    #code = address.get('country_code')
    #zipcode = address.get('postcode')
    #print(f"City : {city}, State : {state}, Country : {country},Country Code : {code}  and Zip Code : {zipcode}")
    return country

def LR_algo_training(result):
    ytrain = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0]])
    #xtrain = np.stack((result[0],result[4]),axis=0)
    a = [[0.45793697, 0.32313621, 0.45793697, 0.45793697, 0.37593555, 0.45353344,
        0.79198682, 0.50830603, 0.41930699, 0.50425845, 0.57561713, 0.34304392,
        0.67922485, 0.44910306, 0.81286687, 0.72930759, 0.83716691, 0.83716708,
        0.83716691, 0.44093269, 0.63363123, 0.66757751, 0.41669154],
       [0.5,        1.,         0.5,        0.5,        0.,         0.,
        0.5,        0.,         0.5,        0.,         0.,         0.5,
        1.,        0.,         0.5,        1.,         0.5,        0.,
        0.5,        0.5,        0.,         0.5,        0.        ],
       [0.5,        0.,         1.,         0.5,        0.,         0.,
        0.5,        1.,         1.,         1.,         0.5,        0.,
        0.5,        1.,         1.,        1.,         0.5,        0.5,
        0.5,        1.,         0.5,        0.,         1.        ]]
    b = [[0.31238842, 0.3395268,  0.27673793, 0.27468252, 0.37086046, 0.37086052,
          0.09869017, 0.12877981, 0.14404683, 0.1455489,  0.11199841, 0.50590217,
          0.69774961, 0.44531024, 0.72210479, 0.69180965, 0.48016196, 0.48016196,
          0.57379758, 0.37141392, 0.53090942, 0.71710718, 0.49450907],
         [0.5,        1.,         0.5,        0.5,        0.,         0.,
          0.5,        0.,         0.5,        0.,         0.,         0.5,
          1.,         0.,         0.5,        1.,         0.5,        0.,
          0.5,        0.5,        0.,         0.5,        0.        ],
         [0.5,        1.,         0.,         0.5,        1.,         1.,
          0.5,        0.,         0.,         0.,         0.5,        1.,
          0.5,        0.,         0.,         0.,         0.5,        0.5,
          0.5,        0.,         0.5,        1.,         0.        ]]
    xtrain = np.stack((a,b),axis=0)
    #xtest = np.stack((result[1],result[2],result[3]),axis=0)
    lin_reg_mod = LinearRegression()
    for i in range(0,len(xtrain)):
        lin_reg_mod.fit(np.transpose(xtrain[i]), np.transpose(ytrain[i]))
    return lin_reg_mod, ytrain

def LR_algo_prediction(model, result, ytrain):  
    #xtest = np.stack((result[1],result[2],result[3]),axis=0)
    xtest=result
    pred = model.predict(np.transpose(xtest))
    res = np.array(pred)
    return res

def word_embedding(user):
    bpemb_en = BPEmb(lang="en", vs=50000) 
    ids = bpemb_en.encode_ids(user)
    l=bpemb_en.vectors[ids]
    return l

def extracting_words(user):
    l=[]
    input=[]
    for i in range(0,len(user)):
        text=user[i]
        t=re.split(',',text)
        for k in range(0,len(t)):
            l.append(t[k])
    text=" "
    input.append(text.join(l))
    return input

def normalize(vec):
    vec = sklearn.preprocessing.normalize(vec,norm="l2")
    return vec

def operation(user_vector, role_vector):
    m=[]
    for j in range(0,len(role_vector)):
        res = np.dot(user_vector,np.transpose(role_vector[j]))
        # print(user_vector[i].shape, role_vector[j].shape, res.shape)
        res = res.max(axis = 1)
        m.append(np.mean(sorted(res, reverse = True)[:2]))
    m=np.array(m)
    return m

def get_place_gender(df_user, df_role):
    country = get_location(df_user[0])
    temp_gender=[]
    temp_location=[]
    gen = df_user[1]
    for j in range(0,len(df_role)):
        #c=get_location(df_role[j][0])
        temp_location.append(0.5) if df_role[j][2]=="Universal" else temp_location.append(1) if country == df_role[j][0] else temp_location.append(0)
    temp_location = np.array(temp_location)
    for j in range(0,len(df_role)):
        temp_gender.append(0.5) if df_role[j][3]=="Universal" else temp_gender.append(1) if gen == df_role[j][1] else temp_gender.append(0)
    temp_gender = np.array(temp_gender)
    return temp_location, temp_gender

def myfunction(Name,address,gender,area_of_interest,specialization,hobbies):
    #print("Enter following details for user")
    #Name = input("Enter name : ")
    #address = input("Enter address : ")
    #gender = input("Enter gender : ")
    #area_of_interest = input("Enter area of interest : ")
    #specialization = input("Enter specialization : ")
    #hobbies = input("Enter hobbies : ")
    
    user = [area_of_interest, specialization, hobbies]
    user = extracting_words(user)
    user_vector = word_embedding(user)
    #print(user_ids)
    dataf_role= pandas.read_csv("Role model sample.csv")                     # Using Role Model Sample data
    dataf= dataf_role[['Known For']]
    dataf=dataf.to_numpy()
    role=[]
    for i in range(0,len(dataf)):
        role.append(extracting_words(dataf[i]))
    role_vector=[]
    for i in range(0,len(role)):
        role_vector.append(word_embedding(role[i]))
    user_vector = normalize(user_vector)
    for i in range(0,len(role_vector)):
        role_vector[i]=normalize(role_vector[i])
    embedding_result = operation(user_vector, role_vector)
    #print(embedding_result)
    df_user=[address,gender]               
    df_user=np.array(df_user)

    df_role=dataf_role[['Location','Gender','Fame_L','Fame_G']]               # Using these three attributes for each user
    df_role=df_role.to_numpy()
    for i in range(0,len(df_role)):
        df_role[i][0]=get_location(df_role[i][0])
    place, gender = get_place_gender(df_user, df_role)
    #print(place)
    result = np.stack((embedding_result, place, gender), axis=0)
    model, ytrain = LR_algo_training(result)
    result = LR_algo_prediction(model, result, ytrain)
    sorted_index = heapq.nlargest(5, range(len(result)), key=result.__getitem__)
    #print("For {0} the best suited role models are {1}, {2}, {3}, {4} and {5}\n".format(name,dataf_role['Role Model'][sorted_index[0]],dataf_role['Role Model'][sorted_index[1]],dataf_role['Role Model'][sorted_index[2]],dataf_role['Role Model'][sorted_index[3]],dataf_role['Role Model'][sorted_index[4]]))
    role1 = dataf_role['Role Model'][sorted_index[0]]
    role2 = dataf_role['Role Model'][sorted_index[1]]
    role3 = dataf_role['Role Model'][sorted_index[2]]
    role4 = dataf_role['Role Model'][sorted_index[3]]
    role5 = dataf_role['Role Model'][sorted_index[4]]

    #mycursor.execute("CREATE DATABASE user_role_model")
    #mycursor.execute("SHOW DATABASES")
    #mycursor.execute("CREATE TABLE predicted_rolemodels (id INT AUTO_INCREMENT PRIMARY KEY,name VARCHAR(255), role_model1 VARCHAR(255), role_model2 VARCHAR(255), role_model3 VARCHAR(255), role_model4 VARCHAR(255), role_model5 VARCHAR(255))")
    #mycursor.execute("SHOW TABLES")
    #for x in mycursor:
    #    print(x)
    sql = "INSERT INTO predicted_rolemodels (name, role_model1, role_model2, role_model3, role_model4, role_model5) VALUES (%s, %s, %s, %s, %s, %s)"
    #val = ("Ayushi Gupta","Bill gates", "AR Rahman", "Benedict Cumberbatch", "J.K Rowling", "Amitabh Bachan")
    val = (Name, role1, role2, role3, role4, role5)
    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")
    #mycursor.execute("SELECT * FROM predicted_rolemodels")

    #myresult = mycursor.fetchall()
    #for x in myresult:
    #    print(x)
    mycursor.execute("SELECT count(*) from predicted_rolemodels")
    iid = mycursor.fetchall()
    return iid,role1,role2,role3,role4,role5

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np 
import joblib

app = Flask(__name__)

model1 = joblib.load('SVM_Model.pkl')
model2 = joblib.load('NB_Model.pkl')
#model3 = joblib.load('Spamdetector\\RandomForest_Model.pkl')
model4 = joblib.load('DecisionTree_Model.pkl')

@app.route('/',methods=['GET', 'POST'])
#Max Voting: The final prediction in this technique is made based on majority voting for classification problems.
def index():
  spam_score=0
  if request.method == 'POST':
    message = request.form.get('message')
    output1 = model1.predict([message])
    output2 = model2.predict([message])
    #output3 = model3.predict([message])
    output4 = model4.predict([message])
    if output1 == ['spam']:
      spam_score=spam_score+1
      svm='spam'
    else:
      svm='ham'
    if output2 == ['spam']:
      spam_score=spam_score+1
      nb='spam'
    else:
      nb='ham'
    '''if output3 == ['spam']:
      spam_score=spam_score+1
      randomforest='spam'
    else:
      randomforest='ham'''''
    if output4 == ['spam']:
      spam_score=spam_score+1
      decisiontree='spam'
    else:
      decisiontree='ham'
    if spam_score > 1:
      result = "This Message is a SPAM Message."
    else:
      result = "This Message is Not a SPAM Message." 
    return render_template('index.html', result=result,message=message,spam_score=spam_score,output1=svm,output2=nb,output4=decisiontree)      

  else:
    return render_template('index.html')  


if __name__ == '__main__':
    app.run(debug=True)
import torch
from torch import nn 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

device = "cpu" #if torch.cuda.is_available() else "cpu"


data=pd.read_csv("ddeep_learning/new_aus1.csv")
wanted_data=pd.read_csv("ddeep_learning/test1.csv")
#wanted_data=wanted_data.dropna()
data=data.dropna()
print(data.head())



x=data[["day","windspeed","humidity","pressure","maxtemp"]]
y=data["maxtemp2"]
wanted_data=wanted_data[["ws","h","maxtemptest","p","d"]]
wanted_data=torch.tensor(wanted_data.values)
wanted_data=wanted_data.type(torch.float)
wanted_data=wanted_data.to(device=device)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=68)
x_train=torch.tensor(x_train.values)
x_train=x_train.type(torch.float)
y_train=torch.tensor(y_train.values)
y_train=y_train.type(torch.float)
x_test=torch.tensor(x_test.values)
x_test=x_test.type(torch.float)
y_test=torch.tensor(y_test.values)
y_test=y_test.type(torch.float)
x_train=x_train.to(device=device)
y_train=y_train.to(device=device)
x_test=x_test.to(device=device)
y_test=y_test.to(device=device)


print(len(x))
print(len(x_train))

network=nn.Sequential(nn.Linear(5,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,100),nn.ReLU(),
nn.Linear(100,1))

model=network.to(device=device)


lossf=nn.MSELoss()
optimizer=torch.optim.Adam(network.parameters(),lr=0.0001)
epochs=1000





for epoch in range(epochs):
    model.train()
    running_loss=0
    y_preds=model(torch.tensor(x_train).float()).squeeze()#remove dimension
    loss=lossf(y_preds,y_train)#loss function
    optimizer.zero_grad()#adjusting gradient value
    loss.backward()#back propagation
    optimizer.step()#
    running_loss += loss.item()
    
    #TESTING PART IN EACH EPOCH. IT WILL NOT SHOW THE FINAL RESULT!!!
    model.eval()
    with torch.inference_mode():
        test_pred=model(x_test)
        test_loss=lossf(test_pred,y_test.type(torch.float))
          

    print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


###########SAVING MODEL IN ORDER TO USE FOR MAKING PREDICTIONS
#print(model.state_dict())## her bir layerdeki weightlerin son değerlerini bu sayede görmüş olabiliyoruz.

modelpath=Path("trained_model")
modelpath.mkdir(parents=True,exist_ok=True)

model_name="01lineer_regression_model.pth"
model_save_path=modelpath/model_name

torch.save(obj=model.state_dict(),f=model_save_path)

trained_model=model
trained_model.load_state_dict(torch.load(f=model_save_path))
trained_model.eval()
with torch.inference_mode():
    loaded_model_preds=trained_model(wanted_data)

print(loaded_model_preds)
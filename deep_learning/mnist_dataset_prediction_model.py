"""
The purpose of this code is teaching AI some objects such as clothes or shoes. To do that, I've used MNIST fashion dataset. In addition, for reducing errors
which can be caused by zero values, I have used leaky-RelU function.
"""
import torch
import torch.nn as nn
import torchvision#libraries to setting up mnist datasets
from torchvision import datasets#libraries to setting up mnist datasets
from torchvision.transforms import ToTensor#libraries to setting up mnist datasets
import matplotlib.pyplot as plt
from tqdm.auto import tqdm #library to show us processing in a bar
from torch.utils.data import DataLoader
import random


device="cpu"
##loading training data
train_data=datasets.FashionMNIST(
    root="datasets",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

#loading test data
test_data=datasets.FashionMNIST(
    root="datasets",
    train=False,
    download=True,
    transform=ToTensor()
)


##we can see the classes with that code
class_names = train_data.classes
print(class_names)

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, #  samples per batch
    shuffle=True # shuffle data 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

print(train_data)

image,label=train_data[13]
plt.imshow(image.squeeze())
plt.title("deneme123")
plt.show()####each image has 28x28 pixels

class cnn_model(nn.Module):
    def __init__(self,output_shape:int):
        super().__init__()
        #In this part we have created blocks and made convolution method
        #Then, we made classifying in fully connected layers in forward prop
        self.block=nn.Sequential(
        nn.Conv2d(1,10,kernel_size=3,padding=1,stride=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(10,10,kernel_size=3,padding=1,stride=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2,2),#Applying max pooling method, then applying convolution again
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=160, out_features=output_shape)
            
        )

    def forward(self,x:torch.Tensor):
        x=self.block(x)
        x=self.classifier(x)
        return x



##setting up loss and optimizer 
network=cnn_model(output_shape=10)
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=network.parameters(),lr=0.1)

#training and testing our model
epochs=3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    train_loss=0
    for batch,(X,y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        network.train()
        
        y_pred=network(X)

        loss=loss_function(y_pred,y)
        train_loss+=loss
        loss.backward()
        optimizer.step()

    train_loss/=len(train_dataloader)

    test_loss=0
    network.eval()
    with torch.inference_mode():
        for X,y in test_dataloader:
            test_pred=network(X)
            test_loss+=loss_function(test_pred,y)
            
        test_loss /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")    
print(len(train_data.classes))

####  VISUALIZING AND GETTING VALUES OF DIFFERENCES BETWEEN TRAINED MODEL RESULTS AND ORIGINAL RESUTLS

def visualize_and_predict(model:torch.nn.Module, data:list, device: torch.device=device):
    pred_probs=[]
    model.eval()
    with torch.inference_mode():
        for ornekler in data:
            ornekler=torch.unsqueeze(ornekler,dim=0).to(device)
            #we are making forward bass without back prop.
            pred_logit=model(ornekler)##logit means prediction probability
            pred_prob=torch.softmax(pred_logit.squeeze(),dim=0)##a softmax function calculates probability for each class from which the input belongs.
            pred_probs.append(pred_prob.cpu())##for more accurate results we use cpu.

    return torch.stack(pred_probs)##merging tensors

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)


pred_probs= visualize_and_predict(model=network, 
                             data=test_samples)

print(pred_probs[:2])


pred_classes = pred_probs.argmax(dim=1)##by argmax, we label most probable class as 1 and others as 0.
###for more information about argmax and softmax, check your discord notes!!!!
print(pred_classes)
print(test_labels)


plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);



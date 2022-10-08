import pandas as pd 
import os
import torch
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.utils.data_plot_utils import KannadaDataClass, plot_digits, train_transform, test_transform
from src.utils.training_utils import Net, train, evaluate
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import numpy as np
import matplotlib.pyplot as plt

# load in env variables
load_dotenv()
N_CLASSES = int(os.getenv('N_CLASSES'))
N_EPOCHS = int(os.getenv('N_EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
IMG_DIM = int(os.getenv('IMG_DIM'))
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

# set seed to obtain consistent results 
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.enabled = False
# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load in data to train/test with
dir_data = 'data/Kannada-MNIST'
dir_train = dir_data +'/train.csv'
dir_test = dir_data + '/test.csv'
alternate_test = dir_data + '/Dig-MNIST.csv'

# read in data
df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)
df_alternate_test = pd.read_csv(alternate_test)

# drop training label from training data
target = df_train['label']
df_train.drop('label', axis=1, inplace=True)

# drop id column as uselss
df_test.drop('id', axis=1, inplace=True)

# drop label from alternate test data
alternate_test_label = df_alternate_test['label']
df_alternate_test.drop('label', axis=1, inplace=True)

# random train and test split on the training data.
# alternate_test can be used as a holdout test to evaluate the model.
X_train, X_test, y_train, y_test = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.1)

#plot_digits(X_train,y_train)
# above line is commented out as it takes a while to plot.
# the image is stored under plots/all_digits.png from the root

# create torch datasets for all datasets for easier path to training.
train_dataset = KannadaDataClass(images=X_train, labels=y_train, transform=train_transform, classes=10)
test_dataset = KannadaDataClass(images=X_test, labels=y_test, transform=test_transform, classes=10)
final_result_test_dataset = KannadaDataClass(images=X_test, labels=None, transform=test_transform, classes=10)
alternate_test_dataset = KannadaDataClass(images=df_alternate_test, labels=alternate_test_label, transform=test_transform, classes=10)

# Defining the data generators for producing batches of data
# these dataLoaders are useful for training and evaluation due to how they
# loop through data in during trianing and evaluation
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
final_result_loader = DataLoader(dataset=final_result_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
alternate_test_loader = DataLoader(dataset=alternate_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load the model
model = Net()
model = model.to(device)

# use the Adam optimizer as it is quite an established 
# optimizer which mantains a different learning rate per parameter.
optimizer = Adam(model.parameters(), lr=0.001)

# lists to store results for plotting purposes
train_loss = []
test_loss = []
test_accuracy = []
alternate_test_loss = []
alternate_test_accuracy = []

# run training and evaluation
for epoch in range(N_EPOCHS):
    print('Epoch: '+ str(epoch))
    train(epoch,model,train_loader, device, optimizer, train_loss)
    evaluate(model,test_loader, alternate_test_loader, device, test_loss, test_accuracy, alternate_test_loss, alternate_test_accuracy)

# plot test curves for test and alternate test data
plt.plot(test_accuracy)
plt.plot(alternate_test_accuracy)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'alternate_test'], loc='upper left')
plt.savefig('plots/test_accuracy.png')
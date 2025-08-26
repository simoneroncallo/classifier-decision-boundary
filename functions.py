# This module trains the a shallow neural network with arbitrary
# number of hidden neurons. Using the test dataset, it returns an image
# comparing the decision boundary against the generating one.
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import utils, layers, models, optimizers

def shallow_network(numHiddenNeurons, func, dataset, numEpochs = 150,\
                      learningRate = 0.05, batchSize = 2000):
  """ Train the shallow network, with arbitrary number of neurons
  and given hyperparameters. """
  utils.set_random_seed(2025)

  print('\nTraining...')

  # Initialization
  model = models.Sequential()

  model.add(layers.InputLayer(shape=(2, 1)))
  model.add(layers.Flatten())
  if numHiddenNeurons > 0:
    model.add(layers.Dense(numHiddenNeurons, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.build()

  # Training
  Xtrain, Xval, Ytrain, Yval = dataset
  optimizer = optimizers.Adam(learning_rate = learningRate)
  model.compile(optimizer=optimizer, loss='binary_crossentropy',\
                metrics=['accuracy'])
  history = model.fit(Xtrain, Ytrain, batch_size = batchSize, verbose=0, \
                      epochs = numEpochs, validation_data=(Xval, Yval))

  return model, history

def boundary(X, model, func, numHiddenNeurons, numPoints = 800):
  """ Test the model and visualize the decision boundary. """
  def buffer_plot_and_get():
    """ Convert a figure to an image. """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

  print('Testing...')

  # Meshgrid
  x1 = np.linspace(-1, 1, numPoints)
  x2 = np.linspace(-1, 1, numPoints)
  Xtest1, Xtest2 = np.meshgrid(x1, x2) # Grid
  Xtest = np.column_stack((Xtest1.ravel(), Xtest2.ravel())) # Samples
  Ytest = np.where(func(Xtest[:,0])<=Xtest[:,1],1,0) # Labels

  # Test
  out = model.predict(Xtest)[:,0]
  Yinf = np.where(out>=0.5,1,0)
  Z = Yinf.reshape(Xtest1.shape)

  # Canvas
  fig, ax1 = plt.subplots(1,1, figsize=(5, 5), dpi = 600,\
                                 constrained_layout = True)
  size = 14
  plt.rcParams.update({'font.size': size})
  plt.rcParams["font.family"] = "serif"
  lines = ['-','--','-','-.']
  colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',\
            'tab:purple','tab:brown']
  lWidth = 1.5 # Linewidth

  # Generators
  ax1.fill_between(X[:,0], func(X[:,0]), y2=-1,\
                   color = 'lightskyblue', alpha = .8)
  ax1.fill_between(X[:,0], func(X[:,0]),\
                   y2=1, color = colors[1], alpha = .35)

  ax1.contour(Xtest1, Xtest2, Z, levels=[0.5], colors='black',\
                linestyles = lines[0], linewidths = lWidth) # Boundary

  ax1.set_xlim(-1,1), ax1.set_ylim(-1,1)
  ax1.set_xticks([]), ax1.set_yticks([])

  plt.text(-0.5, -0.5, 'Class 1', color='black')
  plt.text(-0.5, 0.5, 'Class 2', color='black')

  ax1.set_xlabel('Feature 1'), ax1.set_ylabel('Feature 2')
  plt.suptitle(f'{numHiddenNeurons} Hidden Neurons')
  plt.close()
  return buffer_plot_and_get()
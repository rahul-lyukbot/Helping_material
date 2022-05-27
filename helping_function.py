# This functiion contain all helping function i need in Deep Learning with tensorflow
import tensorflow as tf


# Function 1 - Confusion matrix
import itertools
from sklearn.metrics import confusion_matrix
def con_matrix(Y_true, Y-pred, figsize=(10, 8), classes, text_size = 10):
  """
   :params:
          Y_true - it's contain the orginal values of Y.
          Y_pred - it's contain the prediction made by our model.
          figsize - contain the figure size by default is (10, 8) but you can set it.
          classes - list of classes you have 
          text_size - it contain the size of text to use 
   :return:
          it's return  a confusion matrix between true values of Y and the prediction our model make
  """
  cm = confusion_matrix(Y_true, Y_pred)
    # convort our confusion_matrix into normalization form
  cm_norm = cm.astype("float")/ cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  # let's pretify it's
  fig, ax = plt.subplots(figsize=figsize)
  # Create matrix plot
  cax = ax.matshow(cm, cmap = plt.cm.Blues)

  # create classes as labels
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title = "Confusion matrix",
        xlabel = "Predicted_Model",
        ylabel = "True_label",
        xticks = np.arange(n_classes),
        yticks = np.arange(n_classes),
        xticklabels = labels,
        yticklabels = labels

      
  )  

  # Set the xlabels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # increase the size of labels
  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Setting the threshold for different colours
  threshold = (cm.max() + cm.min())/2.

  # Plot the text on each cell
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
    horizontalalignment="center",
    color="white" if cm[i,j] > threshold else "black",
     size=text_size)
    
    
# Function 2 - Plot Decision Boundaries
import matplotlib.pyplot as plt
def plot_decision_boundaries(model, x, y):
  """
  :params: 
         model - it contain the model in which we want to plot decision boundaries
         x: input
         y: input
   :return:
          it's return the decision boundaries
  """

  # make meshgrid function for different values of x
  x_min, x_max = x[:, 0] - 0.1, x[:, 1] + 0.1
  y_min, y_max = x[:, 0] - 0.1 , x[:, 1] + 0.1

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


  # Creating x values (which we use for prediction)
  x_in = np.c_[xx.ravel(), yy.ravel()]

  # Make prediction
  y_pred = model.predict(x_in)

  # Creating statement for multiclass
  if len(y_pred[0]) > 0:
    print("Dealing with Multiclass problem")
    y_pred = np.argmax(y_pred, axis = 1).reshape(xx.shape)
  else:
    print("Dealing with binaryclass problem")
    y_pred = np.round(y_pred).reshape(xx.shape) 

    # Ploting all things together for visualize
  plt.contourf(xx, yy, cmap=plt.cm.RdYlBu, alpha= 0.7) 
  plt.scatter(x[:, 0], x[:, 1], c="y", s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  
  
  # Function - 3 vew random omage
  def plot_random_image(model, image, true_labels, classes ):
  """
  :params:
         model - contain the model which we use to prediction
         image - contain the sample which is use find that our model predictio is true or not
         true_labels - true labels of our imae
         classes - Contain the list of names as classes
  
  """
  # Set up a random integer
  i =random.randint(0, len(image))


  # Create prediction and target
  target_image = image[i]
  # here we need a prediction probability

  probs_pred = model.predict(target_image.reshape(1, 28, 28))
  pred_labels = classes[probs_pred.argmax()]
  true_label = classes[true_labels[i]]

  # Plot the image
  plt.figure(figsize=(10,10)) 
  plt.imshow(target_image, cmap = plt.cm.binary)

  # compare our prediction with actual data with our model prediction
  if pred_labels == true_label:
    color = "green"
  else:
    color = "red"  

    # Add xlabel information(prediction/true_labels)
   
  plt.xlabel(" Pred :{} {:2.0f}% (True: {} ".format(pred_labels,
                                                       100*tf.reduce_max(probs_pred),
                                                       true_label),
                                                       color=color)
# function - 4  view random image using OS or directories
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

def random_image_view(target_dir, target_class):

  '''
  :params:
          target_dir - contain the target directories from which we want to view images
          target_class - contain the class which we need to target as to view image
   :return:
          return a random image from our targeted_dir or from our targeted_class
  '''
  folder_path = target_dir+target_class

  # getting random image from 
  random_image = random.sample(os.listdir(folder_path), 1)
  print(random_image)

  # read and plot our random image
  img = mpimg.imread(folder_path + "/" + random_image[0])
  print(img.shape)
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  return img


# function - 5 plot loss curves
def plot_loss_curve(history):
  '''
  :params:
         contain model history to find the loss curves
  :return:
          plot loss and accuracy curves of given model history
  '''
  epochs= range(len(history.history["loss"]))
  # loss
  loss = history.history["loss"]
  valid_loss = history.history["val_loss"]
  
  # Accuracy
  accuracy = history.history["accuracy"]
  valid_accuracy = history.history["val_accuracy"]

  # loss curve
  plt.figure(figsize=(8,5))
  plt.plot(epochs, loss, label="Training_loss")
  plt.plot(epochs, valid_loss, label="Testing_loss")
  plt.xlabel('epochs')
  plt.title("Loss_curves")
  plt.legend()

  # Accuracy curve
  plt.figure(figsize=(8,5))
  plt.plot(epochs, accuracy, label="Training_accuracy")
  plt.plot(epochs, valid_accuracy, label="Testing_accuracy")
  plt.xlabel('epochs')
  plt.title("Accuracy_curve")
  plt.legend();
  
  # Function 6 load and resize image according to our need
  def load_resize_image(filename, img_shape=224):
  '''
    :params:
            filename - it contain the filename which we want to resize
            image_shape - this was the targeted size of image
    :return:
            return the resized image
  '''
  # Read image from filename
  img = tf.io.read_file(filename)
  
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)

  # resize the decoded img
  img = tf.image.resize(img, size=[img_shape, img_shape])

  # Rescale our image/normalize our image
  img = img/255.

  # Return our imag as output
  return img


# Function 7 - pred and plot
def pred_and_plot(model, filename, class_name):
  '''
  :params:
         model - it contain the model number which we use for prediction and plotting
         filename - Containg the filename on which we make prediction
         class_name - contain the classes we have for our experiments
   :return:
          return the image with prediction
  '''
  # load and preprocess our image
  img = load_resize_image(filename)

  # make prediction on our custom data
  
  pred = model.predict(tf.expand_dims(img, axis=0))

  if len(pred[0])>1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:
    pred_class = class_name[int(tf.round(pred))]
  #plot the image  our title
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  
  
  # function - 8 compare history by plots
  def compare_hist(original_hist, new_hist, initial_epochs=5):

  """
    :params:
      original_hist = it contain the history of model before fine tuning
      new_hist = it contain the history of model after fine tuning
      intial_epochs = how many epochs the model is use to run
    :return:
            a compare history of before and after fine_tuning  
    """
    # Accuracy and loss before fine tuning
  acc = original_hist.history["accuracy"]
  loss = original_hist.history["loss"]

  val_acc = original_hist.history["val_accuracy"]
  val_loss = original_hist.history["val_loss"]

  # Accuracy and loss before fine tuning
  total_acc = acc + new_hist.history["accuracy"]
  total_loss = loss + new_hist.history["loss"]

  total_val_acc = val_acc + new_hist.history["val_accuracy"]
  total_val_loss = val_loss + new_hist.history["val_loss"]

  # plot the compared loss curves of our models
  plt.figure(figsize= (8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label="Training_accuracy")
  plt.plot(total_val_acc, label = "Val_accuracy")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label = 'Start fine tuning')
  plt.legend(loc = "lower right")
  plt.title("Training and Validation accuracy")
  # plot of losses 
  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label="Training_accurloss")
  plt.plot(total_val_loss, label = "Val_accurloss")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label = 'Start fine tuning')
  plt.legend(loc = "upper right")
  plt.title("Training and Validation loss")

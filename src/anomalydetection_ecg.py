import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(filepath):
    data = pd.read_csv(filepath, header=None)
    return data

def get_features_labels(data):

    rdata=data.values

    labels=rdata[:,-1]
    features=rdata[:,0:-1]
 
    return features, labels

def split_train_test_data(features, labels):

    train_data, test_data, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state = 21)
    return train_data, test_data, train_labels, test_labels 

def compute_global_min_max(train_data):
    # Calculate global min and max from the training data
    global_min_val = tf.reduce_min(train_data)
    global_max_val = tf.reduce_max(train_data)
    return global_min_val, global_max_val

# Normalize the data
def normalize_data(data, min_val, max_val):
    data = (data - min_val) / (max_val - min_val)
    data = tf.cast(data, tf.float32)
    return data

def get_normal_abnormal_data(data, labels):

    labels=labels.astype(bool)

    normal_data = data[labels==True]
    anomalous_data = data[labels==False]

    return normal_data, anomalous_data

# Plot ECG
def plot_ecg(data, title):
    if isinstance(data, tf.Tensor):
        data = data.numpy()  

    plt.plot(np.arange(data.shape[1]), data[0])  # Use data.shape[1] to dynamically get the length of the ECG
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Time Steps")
    plt.grid(True)
    plt.show()

def plot_multi_ecg(data, type='Normal'):

    if isinstance(data, tf.Tensor):
        data = data.numpy()

    # Plotting the first three normal ECG data points
    plt.plot(data[0], label=f'{type} ECG 1')
    plt.plot(data[1], label=f'{type} ECG 2')
    plt.plot(data[2], label=f'{type} ECG 3')

    plt.title(f'First Three {type} ECG Data Points')
    plt.ylabel('Amplitude')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

class AnomalyDetector(tf.keras.models.Model):
    def __init__(self):
        super(AnomalyDetector,self).__init__()
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.Dense(32,activation="relu"),
            tf.keras.layers.Dense(16,activation="relu"),
            tf.keras.layers.Dense(8,activation="relu")])
        
        self.decoder=tf.keras.Sequential([
          tf.keras.layers.Dense(16, activation="relu"),
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(140, activation="sigmoid")])
        
    def call(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(x)
        return decoded
    
# Plot training and validation loss
def plot_loss(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

# Helper function to plot an ECG, its reconstruction, and the reconstruction error
def plot_reconstruction(model, data, title="Reconstruction"):
    predictions = model.predict(data)
    plt.plot(data[0], 'b', label='Original ECG')
    plt.plot(predictions[0], 'r', label='Reconstructed ECG')
    plt.fill_between(np.arange(140), data[0], predictions[0], color='lightcoral', label='Error')
    plt.legend()
    plt.title(title)
    plt.show()

def compute_reconstructions(data):
    reconstructions=autoencoder.predict(data)
    loss=tf.keras.losses.mae(reconstructions,data).numpy()
    return loss

def determine_threshold(train_loss):
    return np.mean(train_loss)+np.std(train_loss)

def plot_histogram(loss, threshold):
    sns.histplot(loss, bins=50, alpha=0.8)
    plt.axvline(x=threshold, color='r', linewidth = 2, linestyle = 'dashed', label = '{:0.3f}'.format(threshold))
    plt.xlabel("Train Loss")
    plt.legend(loc = 'upper right')
    plt.show()

def predict(model,data,threshold):
    reconstructions=model(data)
    loss=tf.keras.losses.mae(reconstructions,data)
    return tf.math.less(loss,threshold), loss


def compute_eval_metrics(predictions,labels):

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    
    print(f"""
        Accuracy = {accuracy}
        Precision = {precision}
        Recall = {recall}
""")

def compute_conf_matrix(test_labels, predictions):
    # Calculate the confusion matrix
    confmat = confusion_matrix(test_labels, predictions)

    return confmat

def plot_conf_matrix(confmat):

    # Normalize the confusion matrix by the number of instances in each actual class
    confmat_percent = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis] * 100

    labels = ['Anomalous','Normal']

    sns.heatmap(confmat_percent, annot=True, cmap="Reds", fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Percentage)")
    plt.show()

def compute_roc(test_labels, pred_scores):

    test_labels_rev=(1-test_labels)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test_labels_rev, pred_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# Plot the ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



# Load the dataset
filepath='data/ecg.csv'
dataframe=load_data(filepath)
print(dataframe.head())

features, labels= get_features_labels(dataframe)

train_data,test_data,train_labels,test_labels=split_train_test_data(features, labels)

print(f'Train data shape: {train_data.shape}')
print(f'Test data shape: {test_data.shape}')

global_min_val, global_max_val= compute_global_min_max(train_data)

train_data = normalize_data(train_data, global_min_val, global_max_val)
test_data = normalize_data(test_data, global_min_val, global_max_val)

normal_train_data, anomalous_train_data = get_normal_abnormal_data(train_data, train_labels)
normal_test_data, anomalous_test_data = get_normal_abnormal_data(test_data, test_labels)

print(f'Normal train data shape: {normal_train_data.shape}')
print(f'Anomalous train data shape: {anomalous_train_data.shape}')
print(f'Normal test data shape: {normal_test_data.shape}')
print(f'Anomalous test data shape: {anomalous_test_data.shape}')

plot_ecg(normal_train_data, "Normal ECG")

plot_ecg(anomalous_train_data, "Abnormal ECG")

plot_multi_ecg(normal_train_data, 'Normal')

plot_multi_ecg(anomalous_train_data, 'Abnormal')

# Build the model
autoencoder = AnomalyDetector()

# creating an early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=4, mode='min')

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = autoencoder.fit(normal_train_data, normal_train_data,
                          epochs=200,
                          batch_size=512,
                          validation_data=(test_data, test_data),
                          shuffle=True, callbacks = [early_stopping])

plot_loss(history)

# Plot a normal and an anomalous ECG reconstruction
plot_reconstruction(autoencoder,  normal_test_data, title="Normal ECG Reconstruction")

plot_reconstruction(autoencoder, anomalous_test_data, title="Anomalous ECG Reconstruction")
    
train_loss=compute_reconstructions(normal_train_data)
print("Train Loss:", train_loss)

threshold = determine_threshold(train_loss)
print("Threshold",threshold)

plot_histogram(train_loss, threshold)

test_loss=compute_reconstructions(anomalous_test_data)
print("Test Loss:", test_loss)

plot_histogram(test_loss, threshold)

preds, pred_scores =predict(autoencoder, test_data, threshold)

compute_eval_metrics(preds, test_labels)


conf_matrix = compute_conf_matrix(test_labels, preds)

plot_conf_matrix(conf_matrix)

fpr, tpr, roc_auc=compute_roc(test_labels, pred_scores)

plot_roc_curve(fpr, tpr, roc_auc)
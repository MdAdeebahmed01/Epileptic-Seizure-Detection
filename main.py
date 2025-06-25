from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import model_from_json

labels = ["Epileptic Seizure","BrainLesion","Relaxed State (Eyes Closed)","Alert State (Eyes Open)","Non-Seizure Brain Activity"]

 
#fucntion to upload dataset
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    
def preprocessing():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y,sc
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    dataset=dataset.drop('Unnamed',axis=1)
    X= dataset.iloc[:,0:178]
    Y = dataset.iloc[:, -1]
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records : "+str(X_test.shape[0])+"\n")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.fit_transform(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
def eda():
    text.delete('1.0', END)
    global dataset
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    # Assuming your data is in a DataFrame named 'df'
    # If not, replace 'df' with the actual variable containing your data

    # Create subplots
    plt.figure(figsize=(8,8))

    # Subplot 1
    plt.subplot(5, 1, 1)  # 5 rows, 1 column, 1st subplot
    plt.plot(dataset['X1'])
    plt.title('X1')
    plt.ylabel('Value')

    # Subplot 2
    plt.subplot(5, 1, 2)  # 5 rows, 1 column, 2nd subplot
    plt.plot(dataset['X2'])
    plt.title('X2')
    plt.ylabel('Value')

    # Subplot 3
    plt.subplot(5, 1, 3)  # 5 rows, 1 column, 3rd subplot
    plt.plot(dataset['X3'])
    plt.title('X3')
    plt.ylabel('Value')

    # Subplot 4
    plt.subplot(5, 1, 4)  # 5 rows, 1 column, 4th subplot
    plt.plot(dataset['X4'])
    plt.title('X4')
    plt.ylabel('Value')

    # Subplot 5
    plt.subplot(5, 1, 5)  # 5 rows, 1 column, 5th subplot
    plt.plot(dataset['X5'])
    plt.title('X5')
    plt.xlabel('Samples')
    plt.ylabel('Value')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()



def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

def run_RFC():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    
    # Check if the pkl file exists
    if os.path.exists('model/RFC_weights.pkl'):
        # Load the model from the pkl file
        rf_classifier= joblib.load('model/RFC_weights.pkl')
        predict = rf_classifier.predict(X_test)
        calculateMetrics("Random_Forest_Classifier", predict, y_test)
    else:
        clf = RandomForestClassifier()
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
        # Make predictions on the test data
        predict=clf.predict(X_test)
        joblib.dump(clf, 'model/RFC_weights.pkl')
        print("Random_Forest_Classifier Model trained and model weights saved.")
        calculateMetrics("Existing RFC", predict, y_test)
def runCNN():
    global loaded_model
    if os.path.exists(model_architecture_path) and os.path.exists(model_weights_path):
        # Load the pre-trained model
        X_train1 = X_train.reshape((9460, 178, 1))
        X_test1 = X_test.reshape((2366, 178, 1))
        with open(model_architecture_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        #print(loaded_model_json.summary())
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_path)
        # Calculate training accuracy
        y_train_pred = np.argmax(loaded_model.predict(X_train1), axis=1)
        training_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
        calculateMetrics("Existing CNN",y_train, y_train_pred)
    else:
        # Initialize the model
        model = Sequential()
        # First Convolutional Layer
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(178, 1)))
        model.add(MaxPooling1D(pool_size=2))
        # Second Convolutional Layer
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        # Third Convolutional Layer
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        # Global Average Pooling Layer
        model.add(GlobalAveragePooling1D())

        # Fully Connected Layer
        model.add(Dense(128, activation='relu'))

        # Output Layer (for classification tasks)
        model.add(Dense(6, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()
        print(model.summary)
        # Assuming X_train and X_test are your input data
        # Shape of X_train: (9460, 178)
        # Check actual shape before reshaping
        print(f"Original shape of X_train: {X_train.shape}")
        print(f"Original shape of X_test: {X_test.shape}") 
        # Reshape the data to have the necessary dimensions
        X_train1 = X_train.reshape((9460, 178, 1))
        X_test1 = X_test.reshape((2366, 178, 1))

        model.fit(X_train1, y_train,epochs=250, batch_size=64,validation_data=(X_test1, y_test))
        # Assuming 'model' is your trained model

        # Save model architecture as JSON
        model_json = model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)

        # Save model weights as HDF5
        model.save_weights("model_weights.h5")

        # Optionally, you can save the entire model (architecture + weights)
        # This allows you to later load the model in one step
        model.save("full_model.h5")


def Detection():
    text.delete('1.0', END)
    
    global sc, classifier, loaded_model, labels
    
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)

    # Ensure scaler is already trained
    scaled_test1 = sc.transform(dataset)  # Use transform instead of fit_transform
    reshaped_test = scaled_test1.reshape((scaled_test1.shape[0], scaled_test1.shape[1], 1))  # Dynamic reshape

    # Make predictions
    predictions = loaded_model.predict(reshaped_test)

    test_temp = pd.read_csv(filename)  # Read data from uploaded file
    for index, row in test_temp.iterrows():
        if index >= len(predictions):
            break  # Prevent index out of range
        
        predicted_index = np.argmax(predictions[index])  # Get class index
        predicted_outcome = labels[predicted_index]  # Convert index to label
        
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n')

                     
import tkinter as tk

def show_admin_buttons():
    # Clear ADMIN-related buttons
    clear_buttons()
    # Add ADMIN-specific buttons
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=330, y=550)
    tk.Button(main, text="Eda", command=eda, font=font1).place(x=500, y=550)  
    tk.Button(main, text="Preprocess Dataset", command=preprocessing, font=font1).place(x=600, y=550)
    tk.Button(main, text="Existing RFC", command=run_RFC, font=font1).place(x=800, y=550)
    tk.Button(main, text="Proposed CNN", command=runCNN, font=font1).place(x=1050, y=550)

def show_user_buttons():
    # Clear USER-related buttons
    clear_buttons()
    # Add USER-specific buttons
    tk.Button(main, text="Prediction From Test Data", command=Detection, font=font1).place(x=330, y=650)

def clear_buttons():
    # Remove all buttons except ADMIN and USER
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

# Initialize the main tkinter window
main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# Configure title
font = ('times', 18, 'bold')
title = Label(main, text='Deep Learning Approaches for Epileptic Seizure Detection and Classification')
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

# ADMIN and USER Buttons (Always visible)
font1 = ('times', 12, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=show_admin_buttons, font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=50, y=550)

user_button = tk.Button(main, text="USER", command=show_user_buttons, font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=50, y=650)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=180)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)
main.config(bg='Cyan2')
main.mainloop()
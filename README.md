
# Epileptic Seizure Detection And Classification Using Deeplearning

Using CNN and RFC for classification ,whether an EEG signal is Epileptical or not.

# Epilepsy
Epilepsy may occur as a result of a genetic disorder or an acquired brain injury, such as a trauma or stroke. During a seizure, a person experiences abnormal behaviour, symptoms and sensations, sometimes including loss of consciousness. There are few symptoms between seizures. Epilepsy is usually treated by medication and in some cases by surgery, devices or dietary changes.
# Seizure
A seizure is a sudden surge of electrical activity in the brain. A seizure usually affects how a person appears or acts for a short time. Many different things can occur during a seizure. Whatever the brain and body can do normally can also occur during a seizure.
# Detection
Epilepsy is the second most common brain disorder after migraine. Automatic detection of epileptic seizures can considerably improve the patients’ quality of life. CurrentElectroencephalogram (EEG)-based seizure detection systems encounter many challenges in real-life situations. The EEGs are non-stationary signals and seizure patterns vary across patients and recording sessions. Moreover, EEG data are prone to numerous noise types that negatively affect the detection accuracy of epileptic seizures. To address these challenges, we introduce the use of a deep learning-based approach that automatically learns the discriminative EEG features of epileptic seizures.Specifically, to reveal the correlation between successive data samples, Convolutional Neural Network (CNN) and A Random Forest Classifier(RFC)  is used to learn the high-level representations of the normal and the seizure EEG patterns. 


## Installation

Install python 3.7.6
Install below packages

```bash
python -m pip install --upgrade pip
pip install tensorflow==1.14.0
pip install keras==2.3.1
pip install pandas==1.3.5
pip install scikit-learn==1.0.2
pip install imutils
pip install matplotlib==3.2.2
pip install seaborn==0.12.2
pip install opencv-python== 4.1.1.26
pip install h5py==2.10.0
pip install numpy==1.19.2
pip install imbalanced-learn==0.7.0
pip install jupyter 
pip install protobuf==3.20.*
pip install scikit-image==0.16.2
```
    


## Deployment

 1.To deploy this project run open the project in Command Prompt

```bash
  python3 main.py
```
2.Train the Models using dataset Epileptic Seizure Recognition.csv

3.Test the trained Models by using sample test dataset test.csv









#! /usr/bin/env python3
print("its started ")
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

global filename
global df, X_train, X_test, y_train, y_test
global ada_acc, rf_acc, knn_acc, gnb_acc, lr_acc, svm_acc, lda_acc, smote_enn_acc

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace(0, np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["Class/ASD Traits "], axis=1))
    y = np.array(df["Class/ASD Traits "])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

def adaboost():
    global ada_acc, ada_cm, ada_sensitivity, ada_specificity, ada_fscore, ada_auc
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    ada_cm = confusion_matrix(y_test, y_pred)
    ada_sensitivity = ada_cm[0, 0] / (ada_cm[0, 0] + ada_cm[0, 1])
    ada_specificity = ada_cm[1, 1] / (ada_cm[1, 0] + ada_cm[1, 1])
    ada_fscore = 2 * (ada_sensitivity * ada_specificity) / (ada_sensitivity + ada_specificity)
    y_pred_proba = ada.predict_proba(X_test)[:, 1]
    ada_fpr, ada_tpr, _ = roc_curve(y_test, y_pred_proba)
    ada_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{ada_cm}\n'
    result_text += f'Sensitivity: {ada_sensitivity}\n'
    result_text += f'Specificity: {ada_specificity}\n'
    result_text += f'F-score: {ada_fscore}\n'
    result_text += f'AUC: {ada_auc}\n\n'
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(ada_fpr, ada_tpr, label='AdaBoost (AUC = %0.2f)' % ada_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AdaBoost')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)
def random_forest():
    global rf_acc, rf_cm, rf_sensitivity, rf_specificity, rf_fscore, rf_auc, rf
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test) 
    rf_acc = accuracy_score(y_test, y_pred)
    rf_cm = confusion_matrix(y_test, y_pred)
    rf_sensitivity = rf_cm[0, 0] / (rf_cm[0, 0] + rf_cm[0, 1])
    rf_specificity = rf_cm[1, 1] / (rf_cm[1, 0] + rf_cm[1, 1])
    rf_fscore = 2 * (rf_sensitivity * rf_specificity) / (rf_sensitivity + rf_specificity)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_proba)
    rf_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{rf_cm}\n'
    result_text += f'Sensitivity: {rf_sensitivity}\n'
    result_text += f'Specificity: {rf_specificity}\n'
    result_text += f'F-score: {rf_fscore}\n'
    result_text += f'AUC: {rf_auc}\n\n'
        # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(rf_fpr, rf_tpr, label='RF (AUC = %0.2f)' % rf_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Radom_Forest')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)
def knn():
    global knn_acc, knn_cm, knn_sensitivity, knn_specificity, knn_fscore, knn_auc
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred)
    knn_cm = confusion_matrix(y_test, y_pred)
    knn_sensitivity = knn_cm[0, 0] / (knn_cm[0, 0] + knn_cm[0, 1])
    knn_specificity = knn_cm[1, 1] / (knn_cm[1, 0] + knn_cm[1, 1])
    knn_fscore = 2 * (knn_sensitivity * knn_specificity) / (knn_sensitivity + knn_specificity)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, y_pred_proba)
    knn_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for K-Nearest Neighbors is {knn_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{knn_cm}\n'
    result_text += f'Sensitivity: {knn_sensitivity}\n'
    result_text += f'Specificity: {knn_specificity}\n'
    result_text += f'F-score: {knn_fscore}\n'
    result_text += f'AUC: {knn_auc}\n\n'
    plt.figure(figsize=(6, 6))
    plt.plot(knn_fpr, knn_tpr, label='knn (AUC = %0.2f)' % knn_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - knn')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)
    

def gnb():
    global gnb_acc, gnb_cm, gnb_sensitivity, gnb_specificity, gnb_fscore, gnb_auc
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    gnb_acc = accuracy_score(y_test, y_pred)
    gnb_cm = confusion_matrix(y_test, y_pred)
    gnb_sensitivity = gnb_cm[0, 0] / (gnb_cm[0, 0] + gnb_cm[0, 1])
    gnb_specificity = gnb_cm[1, 1] / (gnb_cm[1, 0] + gnb_cm[1, 1])
    gnb_fscore = 2 * (gnb_sensitivity * gnb_specificity) / (gnb_sensitivity + gnb_specificity)
    y_pred_proba = gnb.predict_proba(X_test)[:, 1]
    gnb_fpr, gnb_tpr, _ = roc_curve(y_test, y_pred_proba)
    gnb_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for Gaussian Naïve Bayes is {gnb_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{gnb_cm}\n'
    result_text += f'Sensitivity: {gnb_sensitivity}\n'
    result_text += f'Specificity: {gnb_specificity}\n'
    result_text += f'F-score: {gnb_fscore}\n'
    result_text += f'AUC: {gnb_auc}\n\n'
    plt.figure(figsize=(6, 6))
    plt.plot(gnb_fpr, gnb_tpr, label='gnb (AUC = %0.2f)' % gnb_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - gnb')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)

def lr():
    global lr_acc, lr_cm, lr_sensitivity, lr_specificity, lr_fscore, lr_auc
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred)
    lr_cm = confusion_matrix(y_test, y_pred)
    lr_sensitivity = lr_cm[0, 0] / (lr_cm[0, 0] + lr_cm[0, 1])
    lr_specificity = lr_cm[1, 1] / (lr_cm[1, 0] + lr_cm[1, 1])
    lr_fscore = 2 * (lr_sensitivity * lr_specificity) / (lr_sensitivity + lr_specificity)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_proba)
    lr_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for Logistic Regression is {lr_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{lr_cm}\n'
    result_text += f'Sensitivity: {lr_sensitivity}\n'
    result_text += f'Specificity: {lr_specificity}\n'
    result_text += f'F-score: {lr_fscore}\n'
    result_text += f'AUC: {lr_auc}\n\n'
    plt.figure(figsize=(6, 6))
    plt.plot(lr_fpr, lr_tpr, label='lr (AUC = %0.2f)' % lr_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - lr')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)

def svm():
    global svm_acc, svm_cm, svm_sensitivity, svm_specificity, svm_fscore, svm_auc
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    svm_cm = confusion_matrix(y_test, y_pred)
    svm_sensitivity = svm_cm[0, 0] / (svm_cm[0, 0] + svm_cm[0, 1])
    svm_specificity = svm_cm[1, 1] / (svm_cm[1, 0] + svm_cm[1, 1])
    svm_fscore = 2 * (svm_sensitivity * svm_specificity) / (svm_sensitivity + svm_specificity)
    y_pred_proba = svm.predict_proba(X_test)[:, 1]
    svm_fpr, svm_tpr, _ = roc_curve(y_test, y_pred_proba)
    svm_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for Support Vector Machine is {svm_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{svm_cm}\n'
    result_text += f'Sensitivity: {svm_sensitivity}\n'
    result_text += f'Specificity: {svm_specificity}\n'
    result_text += f'F-score: {svm_fscore}\n'
    result_text += f'AUC: {svm_auc}\n\n'
    plt.figure(figsize=(6, 6))
    plt.plot(svm_fpr, svm_tpr, label='SVM (AUC = %0.2f)' % svm_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVM')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)

def lda():
    global lda_acc, lda_cm, lda_sensitivity, lda_specificity, lda_fscore, lda_auc
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    lda_acc = accuracy_score(y_test, y_pred)
    lda_cm = confusion_matrix(y_test, y_pred)
    lda_sensitivity = lda_cm[0, 0] / (lda_cm[0, 0] + lda_cm[0, 1])
    lda_specificity = lda_cm[1, 1] / (lda_cm[1, 0] + lda_cm[1, 1])
    lda_fscore = 2 * (lda_sensitivity * lda_specificity) / (lda_sensitivity + lda_specificity)
    y_pred_proba = lda.predict_proba(X_test)[:, 1]
    lda_fpr, lda_tpr, _ = roc_curve(y_test, y_pred_proba)
    lda_auc = roc_auc_score(y_test, y_pred_proba)
    result_text = f'Accuracy for Linear Discriminant Analysis is {lda_acc * 100}%\n'
    result_text += f'Confusion Matrix:\n{lda_cm}\n'
    result_text += f'Sensitivity: {lda_sensitivity}\n'
    result_text += f'Specificity: {lda_specificity}\n'
    result_text += f'F-score: {lda_fscore}\n'
    result_text += f'AUC: {lda_auc}\n\n'
    plt.figure(figsize=(6, 6))
    plt.plot(lda_fpr, lda_tpr, label='LDA (AUC = %0.2f)' % lda_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - LDA')
    plt.legend(loc="lower right")
    plt.show()
    text.insert(tk.END, result_text)
    text.insert(END, result_text)

def plot_results():
    global ada_acc, rf_acc, knn_acc, gnb_acc, lr_acc, svm_acc, lda_acc
    global ada_sensitivity, rf_sensitivity, knn_sensitivity, gnb_sensitivity, lr_sensitivity, svm_sensitivity, lda_sensitivity
    global ada_specificity, rf_specificity, knn_specificity, gnb_specificity, lr_specificity, svm_specificity, lda_specificity
    global ada_fscore, rf_fscore, knn_fscore, gnb_fscore, lr_fscore, svm_fscore, lda_fscore

    algorithms = ['Ada_B', 'RF', 'KNN', 'GNB', 'LR', 'SVM', 'LDA']
    accuracies = [ada_acc * 100, rf_acc * 100, knn_acc * 100, gnb_acc * 100, lr_acc * 100, svm_acc * 100, lda_acc * 100]
    sensitivities = [ada_sensitivity, rf_sensitivity, knn_sensitivity, gnb_sensitivity, lr_sensitivity, svm_sensitivity, lda_sensitivity]
    specificities = [ada_specificity, rf_specificity, knn_specificity, gnb_specificity, lr_specificity, svm_specificity, lda_specificity]
    fscores = [ada_fscore, rf_fscore, knn_fscore, gnb_fscore, lr_fscore, svm_fscore, lda_fscore]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    # Plot accuracy
    axes[0, 0].bar(algorithms, accuracies, color='skyblue')
    axes[0, 0].set_xlabel('Algorithms')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy of Machine Learning Algorithms')
    axes[0, 0].set_xticklabels(algorithms, rotation=45)
    axes[0, 0].set_ylim(0, 100)
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 1, f'{acc:.2f}%', ha='center')

    # Plot sensitivity
    axes[0, 1].bar(algorithms, sensitivities, color='lightgreen')
    axes[0, 1].set_xlabel('Algorithms')
    axes[0, 1].set_ylabel('Sensitivity')
    axes[0, 1].set_title('Sensitivity of Machine Learning Algorithms')
    axes[0, 1].set_xticklabels(algorithms, rotation=45)
    axes[0, 1].set_ylim(0, 1)
    for i, sens in enumerate(sensitivities):
        axes[0, 1].text(i, sens + 0.02, f'{sens:.2f}', ha='center')

    # Plot specificity
    axes[1, 0].bar(algorithms, specificities, color='lightcoral')
    axes[1, 0].set_xlabel('Algorithms')
    axes[1, 0].set_ylabel('Specificity')
    axes[1, 0].set_title('Specificity_values')
    axes[1, 0].set_xticklabels(algorithms, rotation=45)
    axes[1, 0].set_ylim(0, 1)
    for i, spec in enumerate(specificities):
        axes[1, 0].text(i, spec + 0.02, f'{spec:.2f}', ha='center')

    # Plot F-score
    axes[1, 1].bar(algorithms, fscores, color='lightsalmon')
    axes[1, 1].set_xlabel('Algorithms')
    axes[1, 1].set_ylabel('F-score')
    axes[1, 1].set_title('F-score_values')
    axes[1, 1].set_xticklabels(algorithms, rotation=45)
    axes[1, 1].set_ylim(0, 1)
    for i, fscore in enumerate(fscores):
        axes[1, 1].text(i, fscore + 0.02, f'{fscore:.2f}', ha='center')

    plt.tight_layout()
    plt.show()


def real_time_prediction():
    # Define the functionality for real-time prediction here
    pass

def predict():
    global text
    
    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column (assuming similar preprocessing as done in other functions)
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Perform prediction using the Random Forest model
        y_pred = rf.predict(input_data)

        # Display the prediction result in the text box
        result_text = "Prediction Results:\n"
        for i, prediction in enumerate(y_pred):
            if prediction == 1:
                result_text += f"Row {i + 1}: Autism Spectrum Disorder Detected\n"
            else:
                result_text += f"Row {i + 1}: Autism Spectrum Disorders Not Detected\n"

        # Clear the text box and display the prediction results
        text.delete('1.0', END)
        text.insert(END, result_text)

        # Show a message box indicating the prediction results
        messagebox.showinfo("Prediction Results", "Prediction completed and displayed in the text box.")


# main = tk.Tk()
# main.title("A Machine Learning Framework for Early-Stage  Detection of Autism Spectrum Disorders") 
# main.geometry("500x600")

# font = ('times', 16, 'bold')
# title = tk.Label(main, text='A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders',font=("times"))
# title.config(bg='Dark Blue', fg='white')
# title.config(font=font)           
# title.config(height=3, width=145)
# title.place(x=0, y=5)

# font1 = ('times', 12, 'bold')
# text = tk.Text(main, height=20, width=180)
# scroll = tk.Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=50, y=120)
# text.config(font=font1)

# font1 = ('times', 13, 'bold')
# button_bg_color = "yellow"
# button_fg_color = "black"
# button_hover_bg_color = "grey"
# button_hover_fg_color = "white"
# bg_color = "#32d1a7"  # Light blue-green background color

# # Define button configurations
# button_config = {
#     "bg": button_bg_color,
#     "fg": button_fg_color,
#     "activebackground": button_hover_bg_color,
#     "activeforeground": button_hover_fg_color,
#     "width": 15,
#     "font": font1
# }

# uploadButton = tk.Button(main, text="Upload Dataset", command=upload, **button_config)
# pathlabel = tk.Label(main)
# splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, **button_config)
# adaboostButton = tk.Button(main, text="AdaBoost", command=adaboost, **button_config)
# rfButton = tk.Button(main, text="Random Forest", command=random_forest, **button_config)
# knnButton = tk.Button(main, text="KNN", command=knn, **button_config)
# gnbButton = tk.Button(main, text="Naïve Bayes", command=gnb, **button_config)
# lrButton = tk.Button(main, text="Logistic Regression", command=lr, **button_config)
# svmButton = tk.Button(main, text="SVM", command=svm, **button_config)
# ldaButton = tk.Button(main, text="LDA", command=lda, **button_config)
# plotButton = tk.Button(main, text="Plot Results", command=plot_results, **button_config)
# predict_button = tk.Button(main, text="Prediction", command=predict, **button_config)

# uploadButton.place(x=50, y=600)
# pathlabel.config(bg='DarkOrange1', fg='white', font=font1)  
# pathlabel.place(x=250, y=600)
# splitButton.place(x=450, y=600)
# adaboostButton.place(x=50, y=650)
# rfButton.place(x=250, y=650)
# knnButton.place(x=450, y=650)
# gnbButton.place(x=650, y=650)
# lrButton.place(x=850, y=650)
# svmButton.place(x=1050, y=650)
# ldaButton.place(x=50, y=700)
# plotButton.place(x=450, y=700)
# predict_button.place(x=650, y=700)

# main.config(bg=bg_color)
# main.mainloop()
def predict1():
    global text

    # Create a new window for input form
    input_window = tk.Toplevel(main)
    input_window.title("Enter Feature Values")
    input_window.geometry("1600x900")

    # Define labels and entry fields for numerical features
    numerical_features = ["Case No:", "A1:", "A2:", "A3:", "A4:", "A5:", "A6:", "A7:", "A8:", "A9:", "A10:",
                          "Age Mons:", "Qchat-10-Score:"]

    # Dictionary to store dropdown menus for categorical features
    categorical_features = {
        "Sex": ["", "f", "m"],
        "Ethnicity": ["", "asian", "black", "Hispanic", "Latino", "middle eastern", "mixed", "Native Indian", "Others"],
        "Jaundice": ["", "yes", "no"],
        "Family_mem_with_ASD": ["", "yes", "no"],
        "Who completed the test": ["", "family member", "Health Care Professional", "Others", "Self"]
    }

    feature_entries = {}
    i = 0
    for label in numerical_features:
        tk.Label(input_window, text=label).grid(row=i, column=0)
        feature_entries[label] = tk.Entry(input_window)
        feature_entries[label].grid(row=i, column=1)
        i += 1

    for label, options in categorical_features.items():
        tk.Label(input_window, text=label).grid(row=i, column=0)
        feature_entries[label] = tk.StringVar(input_window)
        feature_entries[label].set("")  # Set default value to empty string
        tk.OptionMenu(input_window, feature_entries[label], *options).grid(row=i, column=1)
        i += 1

    # Function to perform prediction using entered feature values
    def perform_prediction():
        # Extract numerical feature values from entry fields
        numerical_feature_values = [float(feature_entries[label].get()) for label in numerical_features]

        # Extract categorical feature values from dropdown menus and encode them
        label_encoder = LabelEncoder()
        categorical_feature_values = [label_encoder.fit_transform([feature_entries[label].get()])[0] for label in categorical_features]

        # Combine numerical and categorical feature values
        feature_values = numerical_feature_values + categorical_feature_values

        # Perform prediction using the Random Forest model
        y_pred = rf.predict([feature_values])

        # Display the prediction result in the text box
        result_text = "Prediction Result:\n"
        if y_pred[0] == 1:
            result_text += "Autism Spectrum Disorder Detected\n"
        else:
            result_text += "Autism Spectrum Disorder Not Detected\n"

        # Clear the text box and display the prediction result
        text.delete('1.0', END)
        text.insert(END, result_text)

        # Close the input window
        input_window.destroy()

    # Button to trigger prediction using entered feature values
    predict_button = tk.Button(input_window, text="Real_time_Predict", command=perform_prediction)
    predict_button.grid(row=len(numerical_features) + len(categorical_features), columnspan=2)


main = tk.Tk()
main.title("A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders") 
main.geometry("500x600")

font = ('times', 16, 'bold')
title = tk.Label(main, text='A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.pack()

bg_color = "#32d1a7"  # Light blue-green background color

canvas = tk.Canvas(main, bg=bg_color, width=500, height=600, bd=0, highlightthickness=0)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

frame = tk.Frame(canvas, bg=bg_color)
canvas.create_window((0, 0), window=frame, anchor="nw")

font1 = ('times', 12, 'bold')
text = tk.Text(frame, height=20, width=180)
text.config(yscrollcommand=scrollbar.set)
text.pack(side=tk.TOP, padx=50, pady=20)
text.config(font=font1)

button_config = {
    "bg": "yellow",
    "fg": "black",
    "activebackground": "grey",
    "activeforeground": "white",
    "width": 15,
    "font": ('times', 13, 'bold')
}

uploadButton = tk.Button(frame, text="Upload Dataset", command=upload, **button_config)
uploadButton.pack(side=tk.TOP)

pathlabel = tk.Label(frame, bg='DarkOrange1', fg='white', font=('times', 13, 'bold'))
pathlabel.pack(side=tk.TOP)

splitButton = tk.Button(frame, text="Split Dataset", command=splitdataset, **button_config)
splitButton.pack(side=tk.TOP)

adaboostButton = tk.Button(frame, text="AdaBoost", command=adaboost, **button_config)
adaboostButton.pack(side=tk.TOP)

rfButton = tk.Button(frame, text="Random Forest", command=random_forest, **button_config)
rfButton.pack(side=tk.TOP)

knnButton = tk.Button(frame, text="KNN", command=knn, **button_config)
knnButton.pack(side=tk.TOP)

gnbButton = tk.Button(frame, text="Naïve Bayes", command=gnb, **button_config)
gnbButton.pack(side=tk.TOP)

lrButton = tk.Button(frame, text="Logistic Regression", command=lr, **button_config)
lrButton.pack(side=tk.TOP)

svmButton = tk.Button(frame, text="SVM", command=svm, **button_config)
svmButton.pack(side=tk.TOP)

ldaButton = tk.Button(frame, text="LDA", command=lda, **button_config)
ldaButton.pack(side=tk.TOP)

plotButton = tk.Button(frame, text="Plot Results", command=plot_results, **button_config)
plotButton.pack(side=tk.TOP)

predict_button = tk.Button(frame, text="Prediction", command=predict1, **button_config)
predict_button.pack(side=tk.TOP)


# Button to trigger manual feature input and prediction
real_time_predict_button = tk.Button(frame, text="Real-Time Prediction", command=predict1, **button_config)
real_time_predict_button.pack(side=tk.TOP)


main.mainloop()
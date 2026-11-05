#Import Libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#PART A:CNN IMAGE CLASSIFICATION
@st.cache_resource
def train_cnn_model():

    #Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Normalize images
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    #Reshape images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    #Create CNN model
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    #Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    #Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    #Evaluate model
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    #Generate predictions
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    #Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
    #Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    return (
        model,
        history.history,
        test_loss,
        test_accuracy,
        X_test,
        y_test,
        y_pred,
        report,
        cm
    )
    
#PART B: STREAMLIT APPLICATION
#PAGE CONFIGURATION
st.set_page_config(
    page_title="CNN and Student Marks Calculator",
    page_icon="🎓",
    layout="wide"
)

#Application Title
st.title("CNN Image Classification and Streamlit Application")
st.write(
    "This project combines handwritten digit recognition "
    "using CNN and a Student Marks Calculator."
)
st.divider()

#Create Tabs
tab1, tab2 = st.tabs([
    "CNN IMAGE CLASSIFICATION",
    "Student Marks Calculator"
])
#Tab1: CNN IMAGE CLASSIFICATION
with tab1:
    st.header("Handwritten Digit Recognition")
    st.write(
        "This CNN model recognizes handwritten digits "
        "from 0 to 9 using the MNIST dataset."
    )
    st.info(
        "Click the button below to train or load the CNN model. "
        "The trained model is cached to avoid retraining "
        "during normal Streamlit page reruns."
    )
    if st.button("Train / Load CNN Model"):
        with st.spinner("Training CNN model. Please wait..."):
            (
                model,
                history,
                test_loss,
                test_accuracy,
                X_test,
                y_test,
                y_pred,
                report,
                cm
            ) = train_cnn_model()
        st.success("CNN Model Trained Successfully!")

        #Display Dataset Information
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Images", "60,000")
        col2.metric("Testing Images", "10,000")
        col3.metric("Number of Classes", "10")

        st.write("Image Dimensions: 28 × 28 pixels")
        st.write("Image Type: Grayscale")

        #Display Model Performance
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        col1.metric(
            "Test Accuracy",
            f"{test_accuracy * 100:.2f}%"
        )
        col2.metric(
            "Test Loss",
            f"{test_loss:.4f}"
        )

        #Display Accuracy Graph
        st.subheader("Training and Validation Accuracy")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(
            history["accuracy"],
            label="Training Accuracy"
        )
        ax1.plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )
        ax1.set_title("CNN Model Accuracy")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)
        fig1.savefig("accuracy_graph.png")
        plt.close(fig1)
        
        #Display Loss Graph
        st.subheader("Training and Validation Loss")
        fig2, ax2 = plt.subplots(figsize=(8, 4))

        ax2.plot(
            history["loss"],
            label="Training Loss"
        )
        ax2.plot(
            history["val_loss"],
            label="Validation Loss"
        )
        ax2.set_title("CNN Model Loss")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)
        fig2.savefig("loss_graph.png")
        plt.close(fig2)

        #Display Confusion Matrix
        st.subheader("Confusion Matrix")
        fig3, ax3 = plt.subplots(figsize=(8, 6))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=range(10)
        )
        display.plot(
            cmap="Blues",
            ax=ax3,
            colorbar=False
        )
        ax3.set_title("CNN Confusion Matrix")
        st.pyplot(fig3)
        plt.close(fig3)

        #Display Classification Report
        st.subheader("Classification Report")
        st.write(
            "Precision, recall and F1-score for each digit."
        )
        st.dataframe(
            {
                "Digit": list(range(10)),
                "Precision": [
                    report[str(i)]["precision"]
                    for i in range(10)
                ],
                "Recall": [
                    report[str(i)]["recall"]
                    for i in range(10)
                ],
                "F1 Score": [
                    report[str(i)]["f1-score"]
                    for i in range(10)
                ]
            }
        )

        #Display Actual vs Predicted Images
        st.subheader("Actual vs Predicted Images")
        fig4, axes = plt.subplots(2, 5, figsize=(12, 5))

        for i, ax in enumerate(axes.flat):
            ax.imshow(
                X_test[i].reshape(28, 28),
                cmap="gray"
            )
            ax.set_title(
                f"Actual: {y_test[i]}\nPredicted: {y_pred[i]}"
            )
            ax.axis("off")
        fig4.tight_layout()
        st.pyplot(fig4)
        fig4.savefig("prediction_results.png")
        plt.close(fig4)
        st.success("CNN Evaluation Completed!")

#Tab2: STUDENT MARKS CALCULATOR
with tab2:
    st.header("Student Marks Calculator")
    st.write(
        "Enter your subject marks to calculate "
        "total marks, percentage and grade."
    )
    st.divider()

    #Student Information
    st.subheader("Student Information")
    name = st.text_input("Enter Student Name")

    #Subject Marks
    st.subheader("Enter Subject Marks")
    col1, col2 = st.columns(2)
    with col1:
        python_marks = st.number_input(
            "Python Marks",
            min_value=0,
            max_value=100,
            step=1
        )
        java_marks = st.number_input(
            "Java Marks",
            min_value=0,
            max_value=100,
            step=1
        )
        dbms_marks = st.number_input(
            "DBMS Marks",
            min_value=0,
            max_value=100,
            step=1
        )
    with col2:
        cns_marks = st.number_input(
            "CNS Marks",
            min_value=0,
            max_value=100,
            step=1
        )
        ca_marks = st.number_input(
            "Computer Algorithms Marks",
            min_value=0,
            max_value=100,
            step=1
        )

    #Calculate Result
    if st.button("Calculate Result"):
        if name.strip() == "":
            st.error("Please enter your name.")
        else:
            marks = [
                python_marks,
                java_marks,
                dbms_marks,
                cns_marks,
                ca_marks
            ]

            #Total Marks
            total = sum(marks)

            #Percentage
            percentage = (total / 500) * 100

            #Grade and Result
            if min(marks) < 35:
                grade = "F"
                status = "FAIL"
            elif percentage >= 90:
                grade = "A+"
                status = "PASS"
            elif percentage >= 80:
                grade = "A"
                status = "PASS"
            elif percentage >= 70:
                grade = "B"
                status = "PASS"
            elif percentage >= 60:
                grade = "C"
                status = "PASS"
            elif percentage >= 50:
                grade = "D"
                status = "PASS"
            else:
                grade = "E"
                status = "PASS"

            #Display Result
            st.divider()
            st.subheader("Student Result")
            st.write("Student Name:", name)
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Marks",
                f"{total}/500"
            )
            col2.metric(
                "Percentage",
                f"{percentage:.2f}%"
            )
            col3.metric(
                "Grade",
                grade
            )
            if status == "PASS":
                st.success("Result: PASS")
            else:
                st.error("Result: FAIL")
            st.balloons()
#End Of Project
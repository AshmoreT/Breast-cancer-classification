import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

def main():
	st.title("Breast Cancer Classification")
	st.sidebar.title("Breast Cancer Classification")
	st.markdown("Maligne or Benigne??")
	st.sidebar.markdown("Maligne or Benigne")

	@st.cache_data(persist=True)
	def load_data():
		data=pd.read_csv("data.csv")
		label=LabelEncoder()
		for col in data.columns:
			data[col]=label.fit_transform(data[col])
		return data

	@st.cache_data(persist=True)
	def split(df):
		y=df.diagnosis
		x=df.drop(columns=['diagnosis'])
		x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
		return x_train,x_test,y_train,y_test

	def plot_metrics(metrics_list, model, x_test, y_test, class_names):
		y_pred = model.predict(x_test)
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			cm = confusion_matrix(y_test, y_pred, labels=class_names)
			fig, ax = plt.subplots()
			sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
			plt.xlabel("Predicted")
			plt.ylabel("Actual")
			st.pyplot(fig)

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			if hasattr(model, "decision_function"):
				y_score = model.decision_function(x_test)
			else:
				y_score = model.predict_proba(x_test)[:,1]
			fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=class_names[1])
			fig, ax = plt.subplots()
			ax.plot(fpr, tpr, label="ROC Curve")
			ax.plot([0, 1], [0, 1], 'k--')
			plt.xlabel("False Positive Rate")
			plt.ylabel("True Positive Rate")
			plt.title("ROC Curve")
			st.pyplot(fig)

		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			if hasattr(model, "decision_function"):
				y_score = model.decision_function(x_test)
			else:
				y_score = model.predict_proba(x_test)[:,1]
			precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=class_names[1])
			fig, ax = plt.subplots()
			ax.plot(recall, precision, label="Precision-Recall Curve")
			plt.xlabel("Recall")
			plt.ylabel("Precision")
			plt.title("Precision-Recall Curve")
			st.pyplot(fig)



	df= load_data()
	x_train,x_test,y_train,y_test=split(df)
	class_names = df['diagnosis'].unique()
	st.sidebar.subheader("choose classifier")
	classifier=st.sidebar.selectbox("Classifier",("support vector machine(SVM)","logistic regression"))


	if classifier == "support vector machine(SVM)":
		st.sidebar.subheader("Model Hyperparameters")
		C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
		kernel=st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
		gamma =st.sidebar.radio("Gamma (kernel Coefficient)",("scale","auto"),key='gamma')


		metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

		if st.sidebar.button("Classify",key='classify'):
			st.subheader("support vector machine(SVM) results")
			model=SVC(C=C,kernel=kernel,gamma=gamma, probability=True)
			model.fit(x_train,y_train)
			accuracy=model.score(x_test,y_test)
			y_pred=model.predict(x_test)
			st.write("Accuracy:", round(accuracy, 2))
			st.write("Precision:", round(precision_score(y_test,y_pred,labels=class_names), 2))
			st.write("Recall:", round(recall_score(y_test,y_pred,labels=class_names), 2))
			plot_metrics(metrics, model, x_test, y_test, class_names)


	if classifier == "logistic regression":
		st.sidebar.subheader("Model Hyperparameters")
		C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
		max_iter=st.sidebar.slider("Maximum number of iterations",100,500,key='max_iter')
		

		metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

		if st.sidebar.button("Classify",key='classify'):
			st.subheader("logistic regression results")
			model=LogisticRegression(C=C,max_iter=max_iter)
			model.fit(x_train,y_train)
			accuracy=model.score(x_test,y_test)
			y_pred=model.predict(x_test)
			st.write("Accuracy:", round(accuracy, 2))
			st.write("Precision:", round(precision_score(y_test,y_pred,labels=class_names), 2))
			st.write("Recall:", round(recall_score(y_test,y_pred,labels=class_names), 2))
			plot_metrics(metrics, model, x_test, y_test, class_names)


	if st.sidebar.checkbox("Show Raw data",False):
		st.subheader("Breast Cancer Dataset (classification)")
		st.write(df)

	


if __name__ == '__main__':
	main()

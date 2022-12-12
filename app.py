import pandas as pd
import streamlit as st
#from sklearn.linear_model import LogisticRegression

import plotly.express as px

import matplotlib.pyplot as plt
#import bokeh
import seaborn as sns


#df = pd.read_csv("data.txt",';')
df = pd.read_csv("datapoint.txt",';')

st.dataframe(df)
st.write(df['type'].value_counts())

st.write(df.groupby('type').size())
st.write(df.groupby('type').count())
df_st1 = df[df['type']=='st1']





#st.title("Hello")
#fig = sns.countplot(x='type',data=df,palette='hls')
#st.pyplot(fig)


fig = px.scatter(df_st1, x='Longueur Active La (cm)', y="KERMA (cgy cm²)")
st.plotly_chart(fig)

fig = px.scatter(df, x='Longueur Active La (cm)', y="KERMA (cgy cm²)",color='type')
st.plotly_chart(fig)

fig = px.box(df,  x='Longueur Active La (cm)', y="KERMA (cgy cm²)",color='type')
fig.add_scatter(x=[4.5],y=[3000], mode="markers", marker=dict(size=20, color="crimson"), name="a")

fig = px.box(df,  x='Longueur Active La (cm)', y="KERMA (cgy cm²)",)
st.plotly_chart(fig)

fig = px.box(df,  x='Longueur Active La (cm)', y="KERMA (cgy cm²)",color='type')
st.plotly_chart(fig)

Inserta = st.text_input('kerma', )
Insertb = st.text_input('longeur active', )

fig = px.box(df,  x='Longueur Active La (cm)', y="KERMA (cgy cm²)",color='type')
#fig.add_scatter(x=[4.5],y=[3000], mode="markers", marker=dict(size=20, color="crimson"), name="a")
fig.add_scatter(x=[Insertb],y=[Inserta], mode="markers", marker=dict(size=20, color="crimson"), name="a")

st.plotly_chart(fig)

#V CTV HR;V CTV RI

fig = px.scatter(df,  x='V CTV HR', y="V CTV RI",color='KERMA (cgy cm²)')
st.plotly_chart(fig)

fig = px.violin(df,  x='V CTV HR', y="KERMA (cgy cm²)",color='KERMA (cgy cm²)')
st.plotly_chart(fig)

#with open("datapoint.txt", "a") as myfile:
#    myfile.write(Inserta,";",Insertb)

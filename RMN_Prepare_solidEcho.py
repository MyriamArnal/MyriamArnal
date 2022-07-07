import streamlit as st
import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure
import pandas as pd
import re
import xlsxwriter

from RMN.RMN_relax import Remove_badpoints , Load_NormFID, RelaxT2, Couplage_dipolaire

from lmfit import Model

def gaussian_cent(x, amp, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x)**2 / (2*wid**2))
         
@st.cache
def Load_Clean_data(uploaded_file,dw):
    data = Load_NormFID(uploaded_file, d_w)
    data_clean = Remove_badpoints(data)
    return data_clean
    
    
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df.drop(columns=['derv'])
    return df.to_csv().encode('utf-8')    

st.title('Solid Echo analysis')    
"""
* Step 1: Load and clean FID data
* Step 2: Model data  
"""

st.sidebar.subheader('1. Clean  Data')   
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     d_w = 0.1 /1000.
     data_clean = Load_Clean_data(uploaded_file,d_w)
     
     if st.checkbox('Show clean data'):
         p = figure( title='Data', x_axis_label='Time [s]', y_axis_label='Norm')
         p.line(data_clean['Time'], data_clean['Mag_norm'], legend_label='Data', line_width=2)
         st.bokeh_chart(p, use_container_width=True)
         st.write(data_clean)
         
     csv = convert_df(data_clean)
     
     file_out = re.sub('.txt', '',uploaded_file.name)+'_clean.csv'
     st.download_button(
          label="Download clean data",
          data=csv,
          file_name=file_out,
          mime='text/csv',
      )
else : 
    st.write("Load a file to start ...")
     
     
st.sidebar.subheader('2. Model Data')    
option = st.sidebar.selectbox(
     'Which model do you want to fit',
     ('Mou', 'Rigide+mou', 'Rigide'))

if option ==  'Mou' :   
    st.sidebar.write("Mou") 
    AmpM = st.sidebar.slider('Amplitude (Mou)', 0., 1., value = 1.0, step = 0.1)
    T2 = st.sidebar.slider('T2', 0.001, 10., value = 0.5, step = 0.1)
    gmodel = Model(RelaxT2, prefix='Mou_') 
    gmodel.set_param_hint('Mou_amp', value= AmpM,  min=0, max=1)
    gmodel.set_param_hint('Mou_T2', value= T2,  min=0)

if option == 'Rigide' :   
    st.sidebar.write("Rigide") 
    AmpR = st.sidebar.slider('Amplitude (Rig.)', 0., 1., value = 0.9, step = 0.1)
    Wid = st.sidebar.slider('wid', 0.0001, 0.1, value = 0.01, step = 0.001)
   # gmodel = Model(Couplage_dipolaire, prefix='Rig_') 
  #  gmodel.set_param_hint('Rig_amp', value= AmpR,  min=0 , max=1)
   # gmodel.set_param_hint('Rig_k', value= k,  min=0)
    gmodel = Model(gaussian_cent, prefix='Rig_') 
    gmodel.set_param_hint('Rig_amp', value= AmpR,  min=0 , max=1)
    gmodel.set_param_hint('Rig_wid', value= Wid,  min=0)
 
if option == 'Rigide+mou' :   
    st.sidebar.write("Rigide") 
    #AmpR = st.sidebar.slider('Amplitude (Rig.)', 0., 1., 0.1)
    AmpM = st.sidebar.slider('Amplitude (Mou)', 0., 1., value = 0.8, step = 0.1)
    T2 = st.sidebar.slider('T2', 0.001, 15., value = 0.5, step = 0.1)
    Wid = st.sidebar.slider('wid', 0.0001, 0.1, value = 0.01, step = 0.001)
    gmodel = Model(RelaxT2, prefix='Mou_') + Model(gaussian_cent, prefix='Rig_') 
    gmodel.set_param_hint('Mou_amp', value= AmpM,  min=0)
    gmodel.set_param_hint('Rig_amp', value= 1 - AmpM,  min=0)
    gmodel.set_param_hint('Mou_T2', value= T2,  min=0)
    gmodel.set_param_hint('Rig_wid', value= Wid,  min=0)    
    

try:
     x_array= data_clean['Time']
     y_array= data_clean['Mag_norm']
     x_resampled = np.arange(0.0, 0.15, 0.001)
     
     params = gmodel.make_params()    
     result = gmodel.fit(y_array, params, x=x_array)
     
     p = figure( title='Data', x_axis_label='t [s]', y_axis_label='Current') 
     p.line(x_array, y_array, legend_label='Data', line_width=2)
     p.line(x_array, result.best_fit, legend_label='best fit', line_color = 'indianred', line_dash='dashed', line_width=2)
     
     if result.redchi < 5 :
         Amp_res = []
         T2_res = []
         index = []
             
         if option in ['Mou', 'Rigide+mou'] : 
             RelaxT2 = RelaxT2(x_resampled, result.params["Mou_amp"].value, result.params["Mou_T2"].value)
             Amp_res.append(result.params["Mou_amp"].value)
             T2_res.append(result.params["Mou_T2"].value)
             index.append('Mou')
             p.line(x_resampled, RelaxT2, legend_label='Mou', line_color = 'black', line_dash='dashed', line_width=2)
     
         if option in ['Rigide', 'Rigide+mou'] : 
             gaussian_cent = gaussian_cent(x_resampled, result.params["Rig_amp"].value, result.params["Rig_wid"].value)
             Amp_res.append(result.params["Rig_amp"].value)
             T2_res.append(result.params["Rig_wid"].value)
             index.append('Rigide')
             cst = 0
             if option == 'Rigide+mou':
                 cst = RelaxT2[-1]
                 st.write(cst)
             p.line(x_resampled, gaussian_cent + cst, legend_label='Rigide', line_color = 'grey', line_dash='dashed', line_width=2)
     
     st.bokeh_chart(p, use_container_width=True)  
     
     zipped =  list(zip(Amp_res, T2_res))    
     Fraction_df = pd.DataFrame(zipped,  index =index, columns =['Amplitude','T2/k'])
     st.write("Reduced chi-squared: %0.3f " % (result.redchi) )
     st.write(Fraction_df)
     
     if st.checkbox('Show fit results'):
         st.write(result)
         
except NameError:
    st.write('No data')          
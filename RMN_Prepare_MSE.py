import streamlit as st
import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure
from bokeh.palettes import all_palettes

import pandas as pd
import re

from RMN.RMN_relax import Remove_badpoints , Load_NormFID

#import scipy as scipy
#from scipy import optimize
#from lmfit import Model

    

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
  
def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept
    
def constant(x, cst):
    """a line"""
    return 0.*x + cst
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df.drop(columns=['derv'])
    return df.to_csv().encode('utf-8')  
    
@st.cache
def Load_Clean_data(uploaded_file,dw):
    data = Load_NormFID(uploaded_file, d_w)
    data_clean = Remove_badpoints(data)
    return data_clean      

@st.cache
def save_raw_excel(df):
    writer = pd.ExcelWriter('MSE.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Data')
    (max_row, max_col) = df.shape
    
    # Get the xlsxwriter objects from the dataframe writer object.
    workbook  = writer.book
    worksheet = writer.sheets['Data']

    # Create a chart object.
   # chart = workbook.add_chart({'type': 'scatter'})
    
    # Configure the first series.
   # category = ('=Data!$%s$2:$%s$%i' % (col, col, max_row ))
   # chart.add_series({
   #     'name': '=Data!$C$1',
#        'categories': '=Data!$B$2:$B$1501',
#        'values': '=Data!$C$2:$C$1501',
 #   })
    
    # Get the dimensions of the dataframe.
    

    # Configure the series of the chart from the dataframe data.
    #chart.add_chart({'values': 'scatter', 'subtype': 'smooth'})
    # Insert the chart into the worksheet.
    #worksheet.insert_chart(1, max_col + 1, chart)
    
    writer.save()
    

st.title('MSE Analysis')    
"""
* Step 1: Load and clean FID data
* Step 2: Model data    
"""

st.sidebar.subheader('1. Clean  Data') 
uploaded_files = st.sidebar.file_uploader("Choose MSE files",  accept_multiple_files=True)

if uploaded_files is not None:
    d_w = 0.1 /1000.
    
    i = 0 
    name = []
    for files in uploaded_files : 
     # Can be used wherever a "file-like" object is accepted:
          if i ==0 :
              data_clean = Load_Clean_data(files,d_w)
              name.append(files.name)
              Time = np.arange(0.0, 0.15, 0.0001)
              data_all = pd.DataFrame(data = Time, columns=['Time'])
              data_all[files.name] = np.interp(Time, data_clean['Time'], data_clean['Mag_norm'])
              
          else :       
              name.append(files.name)
              data = Load_NormFID(files, d_w)
              data_clean = Remove_badpoints(data)
              data_all[files.name] = np.interp(Time, data_clean['Time'],data_clean['Mag_norm'])
          i=+1
    if st.checkbox('Show fit results'):      
        st.write(data_all)
        
    colors = all_palettes['Category20'][14]
    p = figure( title='Data', x_axis_label='Time [s]', y_axis_label='Norm')   
    for line, color in zip(name, colors):
       p.line(data_all['Time'], data_all[line], legend_label=line, line_width=2 , line_color = color)
    st.bokeh_chart(p, use_container_width=True)  
    save_raw_excel(data_all)
    
    with open("MSE.xlsx", "rb") as file:
        st.download_button('Download Data', data = file,  file_name='MSE.xlsx')  # Defaults to 'application/octet-stream'  
    
else : 
    st.write("Load a file to start ...")
     

st.sidebar.subheader('2. Model Data')    
option = st.sidebar.selectbox(
     'Which model do you want to fit',
     ('Mou', 'Rigide+Mou', 'Rigide'))
     

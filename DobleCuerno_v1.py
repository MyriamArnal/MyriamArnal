import streamlit as st
import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure
import pandas as pd
import scipy as scipy
from scipy import optimize
from lmfit import Model



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
    return df.to_csv().encode('utf-8')    

st.title('Doble Cuerno')    
"""
Ajustement de multi-gaussiennes pour donnée **blablabla**. Le programme accepte en entrée les fichiers natifs produit par **blabla** (3 colonnes: index, t, current). 
* Etape 1 : Charger le fichier dans 'Drag and Drop' de la barre lateral
* Etape 2 : Choisir le nombre de composantes à ajuster. Il est conseillé de commencer avec une composante est rajouter d'autre par la suite. 
* Etape 3 : Ajuster les parametres des gaussiennes jusqu'à obtenir un ajustement. 
"""

uploaded_file = st.sidebar.file_uploader("Choose a file")
number = st.sidebar.number_input('Nombre de composantes', min_value =1, max_value = 3)
#st.sidebar.write("Continuum") 
Cont = st.sidebar.slider('Continuum', 0., 10., 1.2)
Start_values =[Cont]
if number >= 1:   
    st.sidebar.write("Gaussian #1") 
    Amp = st.sidebar.slider('Amplitude', 10, 40, 5)
    Sig = st.sidebar.slider('Sigma', 20, 800, 10)
    Cent = st.sidebar.slider('Centre', 500, 1600, 10)
    gmodel = Model(gaussian, prefix='g1_') + Model(constant, prefix = 'bg_')
    gmodel.set_param_hint('bg_cst', value= Cont)
    gmodel.set_param_hint('g1_amp', value= Amp * Sig /0.3989,  min=0)
    gmodel.set_param_hint('g1_cen', value= Cent,  min=0)
    gmodel.set_param_hint('g1_wid', value= Sig,  min=0)
    
if number >= 2:    
    st.sidebar.write("Gaussian #2") 
    Amp2 = st.sidebar.slider('Amplitude 2', 10, 40, 5)
    Sig2 = st.sidebar.slider('Sigma 2', 10, 800, 10)
    Cent2 = st.sidebar.slider('Centre 2', 500, 1600, 10)
    gmodel = gmodel + Model(gaussian, prefix='g2_') 
    gmodel.set_param_hint('g2_amp', value= Amp2 * Sig2 /0.3989,  min=0)
    gmodel.set_param_hint('g2_cen', value= Cent2,  min=0)
    gmodel.set_param_hint('g2_wid', value= Sig2,  min=0)
    
if number >= 3: 
    st.sidebar.write("Gaussian #3")    
    Amp3 = st.sidebar.slider('Amplitude 3', 0, 40, 5)
    Sig3 = st.sidebar.slider('Sigma 3', 0, 800, 10)
    Cent3 = st.sidebar.slider('Centre 3', 0, 1600, 10)
    gmodel = gmodel + Model(gaussian, prefix='g3_')
    gmodel.set_param_hint('g3_amp', value= Amp3 * Sig3 /0.3989,  min=0)
    gmodel.set_param_hint('g3_cen', value= Cent3,  min=0)
    gmodel.set_param_hint('g3_wid', value= Sig3,  min=0)


if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     data = pd.read_csv(uploaded_file, sep="\t")
     
     x_array= data['t']
     y_array= data['Current']

     if st.checkbox('Show raw data'):
         st.subheader('Raw data')
         st.write(data)
    

     params = gmodel.make_params()    
     result = gmodel.fit(y_array, params, x=x_array)
     
     p = figure( title='Data', x_axis_label='t [s]', y_axis_label='Current')
     p.line(x_array, y_array, legend_label='Data', line_width=2)
     p.line(x_array, result.best_fit, legend_label='best fit', line_color = 'indianred', line_dash='dashed', line_width=2)
     if result.redchi < 5 :
         gauss1 = gaussian(x_array, result.params["g1_amp"].value, result.params["g1_cen"].value, result.params["g1_wid"].value)
         Area =  [np.trapz(gauss1)]
         Amp_res = [result.params["g1_amp"].value]
         Sig_res = [result.params["g1_wid"].value]
         Cent_res = [result.params["g1_cen"].value]
         index = ['Gauss #1']
         p.line(x_array, gauss1, legend_label='gauss1', line_color = 'black', line_dash='dashed', line_width=2)
         
         if number >= 2: 
             gauss2 = gaussian(x_array, result.params["g2_amp"].value, result.params["g2_cen"].value, result.params["g2_wid"].value)
             Area.append(np.trapz(gauss2))
             index.append('Gauss #2')
             Amp_res.append(result.params["g2_amp"].value)
             Sig_res.append(result.params["g2_wid"].value)
             Cent_res.append(result.params["g2_cen"].value)
             p.line(x_array, gauss2, legend_label='gauss2', line_color = 'black', line_dash='dashed', line_width=2)
         if number >= 3: 
             gauss3 = gaussian(x_array, result.params["g3_amp"].value, result.params["g3_cen"].value, result.params["g3_wid"].value)
             Area.append(np.trapz(gauss3))
             index.append('Gauss #3')
             Amp_res.append(result.params["g3_amp"].value)
             Sig_res.append(result.params["g3_wid"].value)
             Cent_res.append(result.params["g3_cen"].value)
             p.line(x_array, gauss3, legend_label='gauss3', line_color = 'black', line_dash='dashed', line_width=2)  
             
         #st.download_button("Download Results",result)
                        
         st.bokeh_chart(p, use_container_width=True)     
         zipped =  list(zip(Area/np.sum(Area), Cent_res, Amp_res, Sig_res ))    
         Fraction_df = pd.DataFrame(zipped,  index =index, columns =['Fraction','Centroid','Amplitude','Width'])
         st.write("Reduced chi-squared: %0.3f " % (result.redchi) )
         st.write(Fraction_df)
         
         csv = convert_df(Fraction_df)

         st.download_button(
              label="Download results as CSV",
              data=csv,
              file_name='Fraction_df.csv',
              mime='text/csv',
          )
          
         if st.checkbox('Show fit results'):
             st.subheader('Fit results')
             st.write(result)
     else:
         st.write("Fit failed !")
         st.bokeh_chart(p, use_container_width=True)
      
     #st.write(result)

     
     


import streamlit as st
import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure
import pandas as pd
import scipy as scipy
from scipy import optimize
from lmfit import Model, Parameter
from bokeh.palettes import Set2


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

def lognormal(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(np.log(x)-cen)**2 / (2*wid**2))

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept

def constant(x, cst):
    """a line"""
    return 0.*x + cst

def _skew_gauss_vec(x, amp, cen, wid, skew):
    """vectorised version of skewed Gaussian, called by skew_gauss"""
    # From https://perso.ens-rennes.fr/~mwerts/skewed-gaussian-fraser-suzuki.html
    ndeps = np.finfo(x.dtype.type).eps
    lim0 = 2.*np.sqrt(ndeps)
    # Through experimentation I found 2*sqrt(machine_epsilon) to be
    # a good safe threshold for switching to the b=0 limit
    # at lower thresholds, numerical rounding errors appear
    if (abs(skew) <= lim0):
        sg = amp * np.exp(-4*np.log(2)*(x-cen)**2/wid**2)
    else:
        lnterm = 1.0 + ((2*skew*(x-cen))/wid)
        sg = np.zeros_like(lnterm)
        sg[lnterm>0] =\
            amp * np.exp(-np.log(2)*(np.log(lnterm[lnterm>0])/skew)**2)
    return sg

def skew_gauss(x,  amp, cen, wid, skew):
    """Fraser-Suzuki skewed Gaussian.

    A: peak height, x0: peak position,
    w: width, b: skewness"""
    # From https://perso.ens-rennes.fr/~mwerts/skewed-gaussian-fraser-suzuki.html

    if type(x)==np.ndarray:
        sg = _skew_gauss_vec(x, amp, cen, wid, skew)
    else:
        x = float(x)
        ndeps = np.finfo(type(x)).eps
        lim0 = 2.*np.sqrt(ndeps)
        if (abs(skew) <= lim0):
            sg = amp * np.exp(-4*np.log(2)*(x-cen)**2/wid**2)
        else:
            lnterm = 1.0 + ((2*skew*(x-cen))/wid)
            if (lnterm>0):
                sg = amp * np.exp(-np.log(2)*(np.log(lnterm)/skew)**2)
            else:
                sg = 0
    return sg



@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def fit_model(x_array, y_array, gmodel, params):

    #params = gmodel.make_params()
    result = gmodel.fit(y_array, params, x=x_array, method='leastsq')

    p1 = figure( title='Data', x_axis_label='t [s]', y_axis_label='leastsq')
    p1.line(x_array, y_array, legend_label='Data', line_width=2)
    p1.line(x_array, result.best_fit, legend_label='best fit', line_color = 'indianred', line_dash='dashed', line_width=2)

    st.write("Reduced chi-squared: %0.3f " % (result.redchi) )

    comps = result.eval_components(x=x_array)
    names = comps.keys()
    res_dic ={}

    if result.redchi < 25 :
        for name in names:
            p1.line(x_array, comps[name], legend_label= name, line_color = 'blue', line_dash='dashed', line_width=2)
            res_dic[name] =  {'Area': np.trapz(comps[name]), 'Amp': result.params[f'{name}amp'].value , 'Sig': result.params[f'{name}wid'].value, 'Cent': result.params[f'{name}cen'].value }
        st.bokeh_chart(p1, use_container_width=True)

        return result, res_dic

    else :
        st.write('Cannot fit data')
        return result , None



def add_component(dist, index, dic):
    col1, col2, col3, col4 = st.columns(4)

    if dist == 'Gaussian':
        dist_label = 'Gauss'
        Amp_range = [0.,100.,5.]
        Cent_range = [0.001,2.,0.1]
        Sig_range = [0.001,1., 0.01]

    if dist == 'LogNormal':
        dist_label = 'LogN'
        Amp_range = [0.,100.,5.]
        Cent_range = [-4.,4.,0.1]
        Sig_range = [0.,1.,0.01]

    if dist == 'Fraser-Suzuki':
        dist_label = 'FS'
        Amp_range = [0.001,100.,5.]
        Cent_range = [0.001,1.5,0.01]
        Sig_range = [0.001,1.,0.1]
        Skew_range = [0.00,3.,0.1]

    name_comp = f'{dist_label}_{index + 1}'
    dic[name_comp] = {}

    with col1:
        dic[name_comp]['Amp']  = st.slider(f'{dist_label}_{i+1}_Amp', Amp_range[0], Amp_range[1], Amp_range[2])
    with col2:
        dic[name_comp]['Sig'] = st.slider(f'{dist_label}_{i+1}_Sig', Sig_range[0], Sig_range[1], Sig_range[2])
    with col3:
        dic[name_comp]['Cent'] = st.slider(f'{dist_label}_{i+1}_Cent', Cent_range[0], Cent_range[1], Cent_range[2])
    if dist == 'Fraser-Suzuki':
        with col4:
            dic[name_comp]['Skew'] = st.slider(f'{dist_label}_{i+1}_Skew', Skew_range[0], Skew_range[1], Skew_range[2])

    if dist == 'Gaussian':
        gmodel = Model(gaussian, prefix=f'g{index}_')
    if dist == 'LogNormal':
        gmodel = Model(lognormal, prefix=f'g{index}_')
    if dist == 'Fraser-Suzuki':
        gmodel = Model(skew_gauss, prefix=f'g{index}_')

    params = gmodel.make_params()
    if dist == 'Fraser-Suzuki':
        params.add(f'g{index}_amp', value= dic[name_comp]['Amp'],  min=Amp_range[0], max=Amp_range[1])
    else :
        params.add(f'g{index}_amp', value= dic[name_comp]['Amp'] * dic[name_comp]['Sig'] /0.3989,  min=Amp_range[0], max=Amp_range[1])
    params.add(f'g{index}_cen', value= dic[name_comp]['Cent'], min=Cent_range[0], max=Cent_range[1])
    params.add(f'g{index}_wid', value= dic[name_comp]['Sig'], min=Sig_range[0], max=Sig_range[1],  vary=True)

    if dist == 'Fraser-Suzuki':
        params.add(f'g{index}_skew', value= dic[name_comp]['Skew'], min=Skew_range[0], max=Skew_range[1],  vary=True)

    print(params)
    return gmodel, dic, params

######### Main ############

st.sidebar.title('Decomposition Mass')

uploaded_file = st.sidebar.file_uploader("Choose CPS files",  accept_multiple_files=False)

option_col = st.sidebar.selectbox('Data column:',('Weigth', 'Number'))

number = st.sidebar.number_input('N# components', min_value =1, value=2, max_value = 3)

option_dist = st.sidebar.selectbox('Function type:',('Gaussian', 'LogNormal', 'Fraser-Suzuki'))

dic = {}

for i in range(0,number,1):
    if i == 0:
        gmodel, dic , params  = add_component(option_dist, i, dic)
    else :
        g_tmp, dic , par_tmp = add_component(option_dist, i, dic)
        gmodel += g_tmp
        params += par_tmp

if uploaded_file is not None:
     colnames = ['Diameter', 'Time','Empty1' , 'Empty2', 'Weight_Height', 'Weight_LogW', 'Weight_CumWt','Empty3', 'Surface_Height', 'Surface_LogSur', 'Surface_CumSurf', 'Empty4', 'Number_Height', 'Number_LogNum', 'Number_CumNum', 'Empty5', 'Absorbance_Height', 'Absorbance_LogAbs', 'Absorbance_CumAbs']
     data = pd.read_csv(uploaded_file, sep=",", skiprows=11, names=colnames, header=None)
     data.insert(1, 'Number_LogNum_norm', 100 * data['Number_LogNum'] / data['Number_LogNum'].max() )
     data.insert(1, 'Weight_LogW_norm', 100 * data['Weight_LogW']/ data['Weight_LogW'].max() )

     x_array= data['Diameter']
     if option_col == 'Number' :
          y_array= data['Number_LogNum_norm']
          label =  'Number'
     else :
          y_array= data['Weight_LogW_norm']
          label =  'LogW'

     #if option_dist == 'LogNormal' :
     #      x_array = np.log(x_array)

     p = figure( title='Data', x_axis_label='Diameter', y_axis_label=label)
     p.line(x_array, y_array, legend_label='Data', line_width=2)

    # total = np.empty([0,])
     col_scale = ['#3288bd', '#99d594', '#fc8d59']
     for (component,color) in zip(dic,col_scale) :
        if option_dist == 'Fraser-Suzuki' :
            comp_ = skew_gauss(x_array.to_numpy(), dic[component]['Amp'],  dic[component]['Cent'], dic[component]['Sig'] , dic[component]['Skew'] )
        elif option_dist == 'LogNormal' :
            comp_ = gaussian(np.log(x_array), dic[component]['Amp']* dic[component]['Sig'] /0.3989, dic[component]['Cent'], dic[component]['Sig'] )
        else:
            comp_ = gaussian(x_array, dic[component]['Amp']* dic[component]['Sig'] /0.3989, dic[component]['Cent'], dic[component]['Sig'] )
        p.line(x_array, comp_, legend_label=component, line_color =color, line_dash='dashed', line_width=2)
      #  total  = np.append(total, comp_.values.tolist(),axis = 0)
    # p.line(x_array, total, legend_label=component, line_color = 'red', line_dash='solid', line_width=3)
     st.bokeh_chart(p, use_container_width=True)
     if st.button('Fit'):

         result, dic_result = fit_model(x_array.to_numpy(), y_array.to_numpy(), gmodel, params)

         if dic_result == None:
             print('No fit')
         else :
             df = pd.DataFrame.from_dict(dic_result).T
             df.insert(1,'Fraction', df['Area']/df['Area'].sum())
             st.dataframe(df.style.format("{:.2f}"))

             csv = convert_df(df)
             st.download_button(
                     label="Download results as CSV",
                     data=csv,
                     file_name='Results.csv',
                     mime='text/csv',
                 )

             st.write(result)

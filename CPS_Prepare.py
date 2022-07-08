import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import xlsxwriter
from io import BytesIO


from operator import itemgetter
from itertools import groupby
import linecache

@st.cache
def save_raw_excel(result_5, result_079, data):
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    bold = workbook.add_format({'bold': 1})
    
    # Worksheet result 
    worksheet_1 = workbook.add_worksheet('Results')
    row = 1
    col = 0
    row_i =3
    col_i =3
    
    parameters = result_5.head()
    worksheet_1.write(row_i-1, col_i, "Result 5 microns",bold)
    worksheet_1.write_row(row_i, col_i, parameters)
    for index in result_5.index :
        worksheet_1.write(row_i + row ,col_i -1, index)
        worksheet_1.write_row(row_i + row, col_i , result_5.loc[index])
        row += 1
        
    worksheet_1.write(row_i + row, col_i, "Result 0.79 microns", bold)
    row += 1
    for index in result_079.index :
        worksheet_1.write(row_i + row ,col_i -1, index)
        worksheet_1.write_row(row_i + row, col_i , result_079.loc[index])
        row += 1
 
    # Worksheet result 
    mydict = {}
    for sample in data :
        name = sample['name']
        mydict[f"worksheet_{name}"] =  workbook.add_worksheet(name)  
        col =0
        data_sample = sample['data']
        for df_col in list(data_sample) : 
            mydict[f"worksheet_{name}"].write(row_i-1,col_i + col,df_col, bold) 
            mydict[f"worksheet_{name}"].write_column(row_i,col_i + col,data_sample[df_col])
            col+=1
    workbook.close()

    return output
    


def group_index(data) :
    ranges =[]
    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))  
    return ranges

def CSP_parameters(data):
    # Sur distribution cumulée : Weight_CumWt_norm
    distr_values = [10.,16.,25.,50.,75.,84.,90.]
    Values = np.interp(distr_values,  data['Weight_CumWt_norm'], data['Diameter'])
    #zipped =  list(zip(Parameter,Values))    
    #result = pd.DataFrame(zipped,  columns =['Parameter','Values'])
    Ld = (Values[1]-Values[5])/Values[3]
    Values = np.append(Values,Ld) 
    
    # Sur distribution cumulée : Weight_Height_norm
    mode_index = data['Weight_Height_norm'].idxmax()
    mode = data['Diameter'][mode_index] 
    Values = np.append(Values,mode) 
    
    #Find where Weight_Height_norm = 50
    norm_50 = np.where(np.abs(data['Weight_Height_norm'] - 50. ) < 2. )
    index_group_50 = group_index(norm_50[:][0])
    index_50 =[]
    for group in index_group_50:
        index_50.append(np.interp([50.], data['Weight_Height_norm'][group[0]:group[1]], data['Diameter'][group[0]:group[1]]))
    if len(index_50) == 2 :
        fwhm = abs(index_50[1]-index_50[0])[0]
    elif len(index_50) > 2 :
        t = np.squeeze(np.abs(index_50 - mode) )
        t_sort = np.argsort(t,axis=0)
        tmp = index_50[t_sort[:][1]] - index_50[t_sort[:][0]]
        fwhm = tmp 
    else:
        fwhm = np.nan 
    Values = np.append(Values,fwhm)    
    #result = result.append({'Parameter':'FWHM', 'Values':fwhm}, ignore_index=True)
    #FWHM/mode
    #result = result.append({'Parameter':'FWHM/Mode', 'Values':fwhm/mode}, ignore_index=True)
    Values = np.append(Values,fwhm/mode) 
    return Values

# main
st.set_page_config(layout="wide")
 
st.sidebar.title('Prepare CPS')   
st.sidebar.write("Preparation des **données CPS** et calcul des parametres pour la methode 5microns et 0.79 microns . Le programme accepte en entrée les fichiers natifs produit par **blabla** .")


uploaded_files = st.sidebar.file_uploader("Choose CPS files",  accept_multiple_files=True)
colnames = ['Diameter', 'Time','Empty1' , 'Empty2', 'Weight_Height', 'Weight_LogW', 'Weight_CumWt','Empty3', 'Surface_Height', 'Surface_LogSur', 'Surface_CumSurf', 'Empty4', 'Number_Height', 'Number_LogSur', 'Number_CumSurf', 'Empty5', 'Absorbance_Height', 'Absorbance_LogNum', 'Absorbance_CumNum']

Data_all = list()

Parameter = ['D90','D84','D75','D50','D25','D16','D10','Ld','Mode','FWHM','FWHM/Mode']

index =[]
for file in uploaded_files : 
    sample_name = file.name.replace('.csv','')
    index.append(sample_name)

result_5microns = pd.DataFrame(columns = Parameter ,  index = index)
result_079microns = pd.DataFrame(columns = Parameter ,  index = index)

for file in uploaded_files : 
    
    sample_name = file.name.replace('.csv','')
    
    data = pd.read_csv(file, sep=",", skiprows=11, names=colnames, header=None)
    data.drop(['Empty1' , 'Empty2', 'Empty3' , 'Empty4', 'Empty5'], axis = 1)
    
   # tmp = list(zip(data_raw['Diameter']*1000, data_raw['Weight_Height'], data_raw['Weight_LogW'],data_raw['Weight_CumWt']))
     # data = pd.DataFrame(tmp, columns =['Diameter', 'Weight_Height_norm', 'Weight_LogW_norm','Weight_CumWt_norm'])
    data['Diameter'] = data['Diameter'] * 1000
    #tmp = list(zip(data_raw['Diameter']*1000, data_raw['Weight_Height'], data_raw['Weight_LogW'],data_raw['Weight_CumWt']))
   # data = pd.DataFrame(tmp, columns =['Diameter', 'Weight_Height_norm', 'Weight_LogW_norm','Weight_CumWt_norm'])
    
    print(data)
    data_079 = data.loc[data['Diameter'] < 790].copy(deep=False).reset_index()
    
    data.insert(1, 'Absorbance_CumNum_norm', 100 * data['Absorbance_CumNum'] / data['Absorbance_CumNum'].max() ) 
    data.insert(1, 'Absorbance_LogNum_norm', 100 * data['Absorbance_LogNum'] / data['Absorbance_LogNum'].max() ) 
    data.insert(1, 'Absorbance_Height_norm', 100 * data['Absorbance_Height'] / data['Absorbance_Height'].max() ) 
    data.insert(1, 'Number_CumSurf_norm', 100 * data['Number_CumSurf']/ data['Number_CumSurf'].max() ) 
    data.insert(1, 'Number_LogSur_norm', 100 * data['Number_LogSur'] / data['Number_LogSur'].max() ) 
    data.insert(1, 'Number_Height_norm', 100 * data['Number_Height'] / data['Number_Height'].max() ) 
    data.insert(1, 'Surface_CumSurf_norm', 100 * data['Surface_CumSurf'] / data['Surface_CumSurf'].max() ) 
    data.insert(1, 'Surface_LogSur_norm', 100 * data['Surface_LogSur']/ data['Surface_LogSur'].max() ) 
    data.insert(1, 'Surface_Height_norm', 100 * data['Surface_Height']/ data['Surface_Height'].max() ) 
    data.insert(1, 'Weight_CumWt_norm', 100 * data['Weight_CumWt']/ data['Weight_CumWt'].max() ) 
    data.insert(1, 'Weight_LogW_norm', 100 * data['Weight_LogW']/ data['Weight_LogW'].max() ) 
    data.insert(1, 'Weight_Height_norm', 100 * data['Weight_Height'] / data['Weight_Height'].max() ) 
 
    data_079.insert(1, 'Weight_CumWt_norm', data_079['Weight_CumWt'] - data_079['Weight_CumWt'][0] ) 
    data_079['Weight_CumWt_norm'] *= 100 / data_079['Weight_CumWt_norm'].max() 
    data_079.insert(1, 'Weight_LogW_norm', 100 * data_079['Weight_LogW']/ data_079['Weight_LogW'].max() ) 
    data_079.insert(1, 'Weight_Height_norm', 100 * data_079['Weight_Height']/ data_079['Weight_Height'].max() )
 
    result_5microns.loc[sample_name] = CSP_parameters(data)
    result_079microns.loc[sample_name] = CSP_parameters(data_079)
  
    Data_all.append({'name':sample_name, 'data': data})

#Print results

label = st.text_area("Description echantillon", height=4 )

with st.container() :
    st.write('Methode 5 microns')
    st.dataframe(result_5microns.style.format("{:.2f}"))

    st.write('Methode 0.79 microns')
    st.dataframe(result_079microns.style.format("{:.2f}"))

    
#Plot figures    
fig = make_subplots(rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
col_scale = px.colors.qualitative.Plotly
i =0                     
for sample in Data_all :
    fig.add_trace(go.Scatter( x=sample['data']["Diameter"], y=sample['data']["Weight_Height_norm"], name = sample['name'], legendgroup='group1', line=dict(color=col_scale[i], width=2)),row=3, col=1)
    fig.add_trace(go.Scatter( x=sample['data']["Diameter"], y=sample['data']["Weight_LogW_norm"], name = sample['name'], legendgroup='group2',showlegend=False, line=dict(color=col_scale[i],width=2)),row=2, col=1)
    fig.add_trace(go.Scatter( x=sample['data']["Diameter"], y=sample['data']["Weight_CumWt_norm"], name = sample['name'], legendgroup='group3', showlegend=False, line=dict(color=col_scale[i], width=2)),row=1, col=1)
    i+=1
    
fig.update_yaxes(title_text="Height", row=3, col=1)
fig.update_yaxes(title_text="LogW", row=2, col=1)
fig.update_yaxes(title_text="CumWt", row=1, col=1)
fig.update_xaxes(type="log", row=1, col=1)
fig.update_xaxes(type="log", row=2, col=1)
fig.update_xaxes(title_text="Diameter [microns]", type="log", row=3, col=1)


fig.update_layout(
    autosize=False,
    width=800,
    height=800,)
                           
st.plotly_chart(fig)

st.download_button( label="Download Excel workbook", data=save_raw_excel(result_5microns, result_079microns, Data_all).getvalue(), file_name="workbook.xlsx", mime="application/vnd.ms-excel")


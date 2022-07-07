import pandas as pd
import numpy as np

def RelaxT2(x, amp,T2):
    return (amp * np.exp(-(x)/T2))
    
def Couplage_dipolaire(x, amp, k):
    return (amp * np.exp(-((x)**(2.))/k)) 
    
def Load_NormFID(File, d_w):
    data = pd.read_csv(File, sep=",")
    data['Real'] = True
    #Separate Real and Imaginary 
    odd = data['T'] % 2 == 0
    data['Real'][odd] = False
    even = data['T'] % 2 != 0
     
    Time = data['T'][even] * d_w 
    Imag =  data['B'][even]
    Real = data['B'][odd]
    zipped =  list(zip(Time , Imag , Real)) 
    df = pd.DataFrame(zipped, columns =['Time','Imag','Real'])
    df['Mag'] =  np.sqrt(df['Imag']**2 + df['Real']**2)
    df['Mag_norm'] =  df['Mag'] / np.max(df['Mag'])
    
    return df
    
def Remove_badpoints(df):
    #Identify points to erase
    # 1) remove echo
    Data = df.copy()    
    Data['derv'] = np.gradient(Data['Mag_norm'])
    selec_bp = Data['derv'] > 0  & (Data['Time'] < 0.030)
    Data.drop(Data[selec_bp].index, inplace = True)
    
    # 2)Find max and remove points before the max
    Data['Mag_norm'] =  Data['Mag_norm'] / np.max(Data['Mag_norm'])        
    t_norm = Data.loc[Data['Mag_norm'] == np.max(Data['Mag_norm'])]['Time'].values  
    selec_tnorm =  Data['Time'] < np.squeeze(t_norm)[()]    
    Data.drop(Data[selec_tnorm].index , inplace = True)
    selec_tnorm =  Data['Time'] > 0.15    
    Data.drop(Data[selec_tnorm].index , inplace = True)    
    Data['Time'] = Data['Time'] - np.squeeze(t_norm)     
    return Data

'''
Created on Jul 7, 2020

@author: eljurros
'''
import os
import pandas as pd
import numpy as np
df_all = []

root_dir = '/media/eljurros/Transcend/CoordConv/ACDC'

def Get_Mean_Scores(df):
    try:
        assert (df[' dice'].size == df[' haussdorf'].size == df['connecterror '].size == df['file'].size )
        dice= np.mean(df[' dice'])
        hauss = np.mean(df[' haussdorf'])
        ce = np.mean(df['connecterror '])
        
        return [dice, hauss, ce]
    except:
        pass

dice = pd.DataFrame()
hauss = pd.DataFrame()
cerror = pd.DataFrame()
for _,dirs,_ in os.walk(root_dir):
    for dir in dirs:
        if 'Fold' in str(dir):
            print(dirs)
            df_clean = []
            df_all = []
            df_name = []
                    
            for loss in ['CoordConvUNet', 'UNet']:

                if os.path.exists(os.path.join(root_dir,dir,'results_CoordConvProject',loss,'CSV_RESULTS','{}_all.csv'.format(loss))) is True:
                    df_name.append(loss)
                    print(loss)
                    df_all.append(Get_Mean_Scores(pd.read_csv (os.path.join(root_dir,dir,'results_CoordConvProject',loss,'CSV_RESULTS','{}_all.csv'.format(loss)))))
                    df_clean.append(Get_Mean_Scores(pd.read_csv (os.path.join(root_dir,dir,'results_CoordConvProject',loss,'CSV_RESULTS','{}_all.csv'.format(loss) ))))
                else:
                    print('doesnt:',os.path.join(root_dir,dir,'results_CoordConvProject',loss,'CSV_RESULTS','{}_all.csv'.format(loss)) )
                #break
                #assert (len(df_name) == len(df_all) == len(df_clean))
            #dice = dice.append(pd.DataFrame(data = [dir], columns = ['Name']))  
            #hauss = hauss.append(pd.DataFrame(data = [dir], columns= ['Name']))
            #cerror = cerror.append(pd.DataFrame(data = [dir], columns= ['Name']))
    
       
            np.transpose(np.array(df_all))
            dice = dice.append(pd.DataFrame(data = [np.transpose(np.array(df_all))[0]], columns = np.array(df_name)))
            hauss = hauss.append(pd.DataFrame(data = [np.transpose(np.array(df_all))[1]], columns = np.array(df_name)))
            cerror = cerror.append(pd.DataFrame(data = [np.transpose(np.array(df_all))[2]], columns = np.array(df_name)))
    
    dice = dice.append(pd.DataFrame(data = [dice.mean(axis = 0), dice.std(axis=0)], columns = np.array(df_name)))
    hauss = hauss.append(pd.DataFrame(data = [hauss.mean(axis = 0), hauss.std(axis=0)], columns = np.array(df_name)))
    cerror = cerror.append(pd.DataFrame(data = [cerror.mean(axis = 0), cerror.std(axis=0)], columns = np.array(df_name)))
    dice.to_csv(os.path.join(root_dir, 'dice.csv'))
    hauss.to_csv(os.path.join(root_dir, 'hauss.csv'))
    cerror.to_csv(os.path.join(root_dir, 'cerror.csv'))
    break

       
                

import pandas as pd

#-- Tensor Flow (NN and Deep Learning)
from tensorflow.keras import Sequential,models, layers, losses, optimizers, activations, metrics, regularizers, callbacks, utils, initializers
import tensorflow as tf

from utils import vec_2_esc, cohort_type


def escalar_narmax_build(df,look_back=12):
  df_temp = df[['date','value_norm']].copy()
  df_temp.columns = ['date','lag_12']
  for i in range(1,look_back + 1):
    df_temp['lag_'+str(12-i)] = df_temp['lag_12'].shift(-i)


  df_temp.rename({'lag_0':'pred_1'},inplace=True,axis=1)
  df_temp.dropna(how='any',inplace=True,axis=0)
  df_temp.reset_index(drop=True,inplace=True)

  df_temp['date'] = df['date'].tail(df_temp.shape[0]).to_list()
  df_temp['date'] = pd.to_datetime(df_temp['date'])
  df_temp['cohort'] = df_temp['date'].apply(lambda x: cohort_type(x, 'escalar'))
                
  return df_temp 

def vetorial_narmax_build(df,look_back=12, vec_num = 12):
    df_temp = df[['date','value_norm']].copy()
    df_temp.columns = ['date','lag_12']

    for i in range(1,12):
        df_temp['lag_'+str(12-i)] = df_temp['lag_12'].shift(-i)

    for j in range(1,13):
        df_temp['pred_'+str(j)] = df_temp['lag_12'].shift(-(i+j))

    df_temp['date'] = pd.to_datetime(df_temp['date'])

    df_temp['start_date'] = df_temp['date'].shift(-(vec_num))
    df_temp['end_date'] = df_temp['date'].shift(-(vec_num+look_back-1))
    # df_temp['start_date'] = df_temp['date'].shift(-i-1)
    # df_temp['end_date'] = df_temp['date'].shift(-i-j)

    df_temp.dropna(how='any', axis=0,inplace=True)
    df_temp.reset_index(drop=True,inplace=True)
    df_temp['cohort'] = df_temp['end_date'].apply(lambda x: cohort_type(x, 'vetorial'))
    return df_temp 


# Create NN model
def NarmaxModel(look_back,vec_num=1):

    Relu = activations.relu
    Linear = activations.linear
    LecunNormal = initializers.LecunNormal()


    model = Sequential([
        layers.Input(shape=(look_back,)),
        layers.Dense(24, activation=Relu,kernel_initializer=LecunNormal),
        layers.Dense(12, activation=Relu,kernel_initializer=LecunNormal),
        layers.Dense(12, activation=Relu,kernel_initializer=LecunNormal),
        layers.Dense(6, activation=Relu,kernel_initializer=LecunNormal),
        layers.Dense(vec_num, activation=Linear,kernel_initializer=LecunNormal)
    ])
    
    model = models.Model(inputs=model.inputs, outputs=model.outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='mse',
        metrics=['mae','mse','mape'])
    return model


# def vec_bootstrap(main_df,scaler,X,y,lag_list,pred_list,split_esc,sample=5):
#   metrics_list = []

#   for i in range(sample):

#     model_temp = MLP(look_back=12,vec_num=12)

#     model_temp.fit(X_train, y_train,
#                 epochs=1000,
#                 batch_size=batch_size,
#                 verbose=0,
#                 validation_data = (X_train, y_train),
#                 shuffle=True)
    
#     df_vec = main_df[['date']].copy()
#     df_vec['real'] = scaler.inverse_transform(vec_2_esc(y))
#     df_vec['pred'] = scaler.inverse_transform(vec_2_esc(pd.DataFrame(model_temp.predict(X,verbose=0))))
#     df_vec['tipo'] = ['Treino' if i < split_esc else 'Teste' for i in df_vec.index]
    
#     test = df_vec[df_vec.tipo == 'Teste']
#     train = df_vec[df_vec.tipo == 'Treino']

#     metric_df_final = pd.DataFrame(
#       {
#       'sample':[i,i],
#       'tipo':['Treino','Teste'],
#       'MSE': [MSE(train['real'],train['pred']), MSE(test['real'],test['pred'])],
#       'MAE': [MAE(train['real'],train['pred']), MAE(test['real'],test['pred'])],
#       'MAPE': [MAPE(train['real'],train['pred']), MAPE(test['real'],test['pred'])],
#       'R2':  [R2(train['real'],train['pred']), R2(test['real'],test['pred'])],
#       }
#     )

#     metrics_list.append(metric_df_final.copy())
#     teste = pd.concat(metrics_list)
#     teste.to_csv('teste.csv',index=False)
#     print(i)

#   return metrics_list
    
# def esc_bootstrap(main_df,scaler,X,y,lag_list,pred_list,split_esc,sample=5):
  
#   metrics_list = []

#   for i in range(sample):

#     model_temp = MLP(look_back=12,vec_num=1)

#     model_temp.fit(X_train, y_train,
#                 epochs=250,
#                 batch_size=batch_size,
#                 verbose=0,
#                 validation_data = (X_train, y_train),
#                 shuffle=True)
    
#     df_escalar = main_df[['date']].copy()
#     df_escalar['real'] = scaler.inverse_transform(y)
#     df_escalar['pred'] = scaler.inverse_transform(model_temp.predict(X, verbose=0))
#     df_escalar['tipo'] = ['Treino' if i < split_esc else 'Teste' for i in df_escalar.index]
#     df_escalar.to_csv('camargos_pred_results.csv',index=False)

#     test = df_escalar[df_escalar.tipo == 'Teste']
#     train = df_escalar[df_escalar.tipo == 'Treino']


#     metric_df_final = pd.DataFrame(
#         {
#         'sample': [i,i],
#         'tipo':['Treino','Teste'],
#         'MSE': [MSE(train['real'],train['pred']), MSE(test['real'],test['pred'])],
#         'MAE': [MAE(train['real'],train['pred']), MAE(test['real'],test['pred'])],
#         'MAPE': [MAPE(train['real'],train['pred']), MAPE(test['real'],test['pred'])],
#         'R2':  [R2(train['real'],train['pred']), R2(test['real'],test['pred'])],
#         }
#     )

#     metrics_list.append(metric_df_final.copy())
#     teste = pd.concat(metrics_list)
#     teste.to_csv('teste.csv',index=False)
#     print(i)

#   return metrics_list
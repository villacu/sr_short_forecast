'''
Script with the transformer-based forecasting model

Based on:
maxcohen positional encoding:  https://github.com/maxjcohen/transformer/blob/master/tst/utils.py

{missing}

check these links:
checar https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/16.attention-is-all-you-need.ipynb
https://arxiv.org/pdf/2001.08317.pdf

'''

import torch
from torch import nn
import numpy as np
import time 


#positional encoding based on maxjcohen implementation https://github.com/maxjcohen/transformer/blob/master/tst/utils.py

def generate_regular_PE(length: int, d_model: int, period=24):
    PE = torch.zeros((length, d_model))
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))
    return PE


class forecast_transformer(nn.Module):
    def __init__(self,d_input,d_model,heads,num_lys,window,decoder_window):
        super(forecast_transformer,self).__init__()
        
        self.DW = decoder_window
        self.window_size = window
        self.d_model = d_model
        self.embed1 = nn.Linear(d_input,self.d_model)
        self.embed2 = nn.Linear(d_input,self.d_model)
        
        #positional embedding
        self.encode = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=heads, 
                                                 dim_feedforward=1024, dropout=0.1,
                                                 activation='gelu')
        self.stackedEncoder = torch.nn.TransformerEncoder(encoder_layer=self.encode, 
                                                          num_layers=num_lys, norm=None)
        self.decode = torch.nn.TransformerDecoderLayer(d_model=self.d_model, nhead=heads, dim_feedforward=1024, 
                                                       dropout=0.1, activation='gelu')
        self.stackedDecoder = torch.nn.TransformerDecoder(decoder_layer = self.decode,
                                                          num_layers=num_lys, norm=None)
                                                       
        self.output1 = nn.Linear(self.d_model*self.DW,512) #
        self.output2 = nn.Linear(512,1)
        self.gen_PE = generate_regular_PE

    def forward(self,x):
        '''
        input [b_sz,tiempos(48),features(17)]
        K -> ventana de tiempo (48)
        

        returns [b_sz,1,1]
    
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        K = x.shape[1]
        
        x_enc = x[:,:(self.window_size-self.DW),:]
        x_dec = x[:,(self.window_size-self.DW):,:]
        
        x_enc = self.embed1(x_enc) 
        K_enc = x_enc.shape[1]
        PE_gen1 = self.gen_PE(length = K_enc, d_model=self.d_model, period = K)
        PE_gen1 = PE_gen1.to(device)
        x_enc = x_enc.add_(PE_gen1) 
        x_enc = self.stackedEncoder(x_enc)
        
        #decoder
        
        K_dec = x_dec.shape[1]
        x_dec = self.embed1(x_dec)
        PE_gen2 = self.gen_PE(length = K_dec, d_model=self.d_model, period = K)
        PE_gen2 = PE_gen2.to(device)    
        x_dec =  x_dec.add_(PE_gen2) 
        x = self.stackedDecoder(x_dec,x_enc)
        x = torch.flatten(x,start_dim=1)

        #salida del decoder
        x = self.output1(x)
        x = self.output2(x)
        x = x.view(-1,1,1)
        return x

def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fcn,
    num_epochs=50,
    savename = 'checkpoint'):
    tic = time.time()
    count = 0
    loss_list = []
    iteration_list = []
    #accuracy_list = []
    prev = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        for _,(x,y) in enumerate (train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad() #borra el gradiente
            outputs = model(x) #propagacion
            loss = loss_fcn(outputs,y) #calcula el error
            loss.backward()    #retropropaga error
            optimizer.step()   #actualiza los parámetros
            count +=1
            
            #evalúa la prediccion
            
            if count%50 ==0:
                #correct = 0
                #total = 0
                loss_list2 = []
                #Predict test datase
                with torch.no_grad():
                    for x,y in val_loader:
                        #predice en lote
                        #inferencia
                        tmp_losslist = []
                        x =x.to(device)
                        y = y.to(device)
                        out = model(x)
                        loss2 = loss_fcn(out,y)
                        loss_list2.append(loss2)
                        tmp_losslist.append(loss2.data)
                print('val loss: ',loss2)
                avg_loss = torch.mean(torch.Tensor(tmp_losslist))
                if(avg_loss<prev):
                    torch.save(model,'saves/'+savename+'.pt')
                    print('saved model')
                    prev = avg_loss.data
                #almacena evaluacion de desempeño
                iteration_list.append(count)
                loss_list.append(loss.data)
                #accuracy_list.append(accuracy)
            
            
            
            #despliega evaluacion
            #print(loss)
            if count%500==0:
                #loss_avg = np.mean(loss_list[(len(loss_list)-10):])
                
                print('epoch: ',epoch)
                print('iter:{} loss:{}'.format(count,loss.data))
    print('elapsed = ',time.time()-tic)
    return loss_list
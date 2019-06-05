#author:lianjie
#created:2019.4.29
#description:the implementation of the convlstm

import torch
import torch.nn as nn
from torch.autograd import Variable

class convlstm_cell(nn.Module):
    def __init__(self,input_size,input_channels,output_channels,kernel_size):
        '''
        Parameters:
        input_size:
            the size of single image of sequence.(h,w)
        input_channels:
            the number of channels of inputs.
        output_channels:
            the number of channels of hidden state's ouputs.
        kernerl_size:
            the size of convolutional kernel.
        '''
        super(convlstm_cell,self).__init__()
        
        self.height, self.width = input_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size-1)//2
        self.conv = nn.Conv2d(self.input_channels+self.output_channels,
            4*self.output_channels,kernel_size=self.kernel_size,padding=self.padding)

    def forward(self,inputs,hidden_state): #inputs [b,c,h,w]
        h_cur,c_cur = hidden_state 
        combined = torch.cat((h_cur,inputs),1)
        combined_conv = self.conv(combined)

        f_in,i_in,c_in,o_in = torch.split(combined_conv,self.output_channels,dim=1)
        f = torch.sigmoid(f_in)
        i = torch.sigmoid(i_in)
        c = torch.tanh(c_in)
        o = torch.sigmoid(o_in)

        c_next = f*c_cur + i*c
        h_next = o*torch.tanh(c_next)

        return h_next,c_next
    
    def hidden_init(self,batch_size):
        return (Variable(torch.zeros(batch_size,self.output_channels,self.height,self.width)),
            Variable(torch.zeros(batch_size,self.output_channels,self.height,self.height)))

class convlstm(nn.Module):
    def __init__(self,input_size,input_channels,output_channels,kernel_size,num_layers):
        '''
        Parameters:
        input_size,output_channels,kernel_size:
            the form is a list or tuple.
        '''
        super(convlstm,self).__init__()
    
        self.input_size = input_size #(h,w)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            input_channels_cur = self.input_channels if i == 0 else self.output_channels[i-1]
            cell_list.append(convlstm_cell(self.input_size,input_channels_cur,
                self.output_channels[i],self.kernel_size[i]))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self,inputs,hidden_states): #inputs [t,b,c,h,w]
        seq_len = inputs.size(0)
        layer_in_cur = inputs
        layer_out_list = []
        last_state_list = []
        
        for layer_idx in range(self.num_layers):
            h,c = hidden_states[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h,c = self.cell_list[layer_idx](layer_in_cur[t,:,:,:,:],(h,c))
                output_inner.append(h) #output_inner [b,c,h,w]

            layer_out = torch.stack(output_inner,0) #[t,b,c,h,w]
            layer_in_cur = layer_out
            layer_out_list.append(layer_out)
            last_state_list.append((h,c))

        return layer_out_list,last_state_list

    def hidden_init(self,batch_size):
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].hidden_init(batch_size))

        return hidden_states

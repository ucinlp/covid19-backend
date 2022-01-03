import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 fc_layers,
                 output_dim, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
                                
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)        
        self.translation = nn.Linear(embedding_dim, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, 
                            hidden_dim, 
                            num_layers = lstm_layers, 
                            bidirectional = True, 
                            dropout=dropout if lstm_layers > 1 else 0)
        
        fc_dim = hidden_dim * 2        
        fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(fc_layers)]
        
        self.fcs = nn.ModuleList(fcs)        
        self.fc_out = nn.Linear(fc_dim * 2, output_dim)        
        self.dropout = nn.Dropout(dropout)
        self.field = []
                
    def forward(self, prem, hypo):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        prem_seq_len, batch_size = prem.shape
        hypo_seq_len, _ = hypo.shape
        
       
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)
        
        translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))

        outputs_prem, (hidden_prem, cell_prem) = self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(translated_hypo)
        

        dif = torch.abs (hidden_prem[-1] - hidden_hypo[-1])
        mult = hidden_prem[-1] * hidden_hypo[-1]

        hidden = torch.cat((hidden_prem[-1], hidden_hypo[-1], dif, mult), dim=1)

            
        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.tanh(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden).to(device = device)
               
        return prediction

    def _encode (self, sentences):

      ## Tokenize
      tokens = []
      for s in sentences:
          toks = self.field.tokenize(s)
          tokens.append ( self.field.preprocess(toks) )

      ## Pad
      lens = [len(x) for x in tokens]
      max_length = max(lens)

      for t in tokens:
        if len(t) < max_length:
          pads = ['<pad>'] * (max_length-len(t))
          t.extend(pads)
      
      ## Encode
      enc = []
      for t in tokens:        
        num = self.field.numericalize([t])
        enc.append (num.flatten())

      enc_sentences = torch.stack(enc, dim=1)
      return (enc_sentences.cuda())

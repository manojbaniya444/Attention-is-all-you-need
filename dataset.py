import torch
from torch import nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        
        super().__init__()  
             
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(["EOS"])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(["PAD"])], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: any):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only [SOS] in the decoder side.
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add SOS, EOS  and Padding Tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # Add SOS and Padding to the input of the decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # Adding SOS and Padding to the label tokens
        label = torch.cat(
            [
                torch.tensor(
                    dec_input_tokens, dtype=torch.int64
               ),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len, "Encoder input size not equals the sequence length"
        assert decoder_input.size(0) == self.seq_len, "Decoder input size not equals the sequence length"
        assert label.size(0) == self.seq_len, "Label size not equals the sequence length"
        
        # returning the dataset
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            # encoder mask to mask the pad token
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, sequence_length)
            # cross attention mask
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, sequence_length)
            "label": label, # (seq_length)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
# causal masking
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
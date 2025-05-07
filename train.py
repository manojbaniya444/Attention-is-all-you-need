import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_weights_file_path, get_config

import warnings

from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from tqdm import tqdm

# Greedy decoding
def greedy_decoding(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # start of sentence and end of sentence token id here
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the initial input for inference with the SOS Token (Auto Regressive Type)
    decoder_input = torch.empty(1, 1).fill(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # building mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output from the projection layer
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

        

# validation
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, global_state, num_examples):
    # put the model in eval mode
    model.eval()
    
    count = 0
    source_texts = []
    expected = []
    predicted_output = []
    
    # size of the control window (just use a default value)
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_output = greedy_decoding(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted_output.append(model_out_text)
            
            # Printing the console
            
            
            if count == num_examples:
                break
            

# get all sentences
def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, dataset, lang):
    
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset("opus_books", f"{config["lang_src"]}-{config["lang_tgt"]}", split="train")
    
    # Building Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # Train and Val split dataset
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    # return the dataloader
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

# training model
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exits_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        
    # loss function training example
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    # training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch}")

        for batch in batch_iterator:
            
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            # final linear layer
            projection_output = model.project(decoder_output)
            label = batch["label"].to(device)

            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({f"Loss": f"{loss.item()}"})
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            
            # increase the global step by 1
            global_step = global_step + 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, global_step, 10)
            
    # saving the model
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step
    }, model_filename)
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config=config)
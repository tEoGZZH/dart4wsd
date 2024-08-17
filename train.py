import torch
import torch.nn as nn
from tqdm import tqdm
from util import log_cosh_loss


def train_shoot(model, dataloader, optimizer, scheduler, num_epochs, mode, device, data):
    last_loss = 0
    if mode == "norm":
        loss_fn = log_cosh_loss
    elif mode == "direction":
        loss_fn = nn.CosineEmbeddingLoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True, position=0)
        for batch in progress_bar:
            # Unpack batch and send to device
            batch_input_ids, batch_attention_masks, batch_word_indices, \
            batch_sense_indices, _, _ = [b.to(device) for b in batch]

            optimizer.zero_grad()
            # Forward pass
            output = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, \
                            word_index=batch_word_indices).squeeze()
            # print(output)
            if mode == "norm":
                batch_norms = data[batch_sense_indices]
                loss = loss_fn(output, batch_norms * 1e-5)
            elif mode == "direction":
                labels = torch.ones(output.size(0), device=device)
                batch_senses = data[batch_sense_indices]
                loss = loss_fn(output, batch_senses, labels)
                
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Logging average metrics per epoch
        avg_loss = total_loss / len(dataloader)
        improvement = (last_loss - total_loss) / len(dataloader) if last_loss != 0 else 0
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss}, Improvement: {improvement}')
        last_loss = total_loss

        scheduler.step()
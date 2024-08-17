import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from util import log_cosh_loss

def predict_with_tree(tree, output_emb, candidate_indices, candidate_norms, candidate_directions, candidate_radii, nball_norms, nball_embeddings, nball_radius, index_to_label, label_to_index):
    current_nodes = [index_to_label[idx] for idx in candidate_indices]
    visited_nodes = set()
    
    while current_nodes:
        distances = [F.pairwise_distance(output_emb.unsqueeze(0), (cand_dir * cand_norm.unsqueeze(-1)).unsqueeze(0)).item()
                     for cand_dir, cand_norm in zip(candidate_directions, candidate_norms)]
        within_radius_indices = [idx for idx, (distance, radius) in enumerate(zip(distances, candidate_radii)) if distance <= radius.item()]
        
        if within_radius_indices:
            return label_to_index[current_nodes[within_radius_indices[0]]]  # Return the first one within the radius
        
        visited_nodes.update(current_nodes)  # Add current nodes to visited set
        
        # Move to the parent nodes, avoiding revisiting nodes
        parent_nodes = set()
        for node in current_nodes:
            for parent, children in tree.items():
                if node in children and parent not in visited_nodes:
                    parent_nodes.add(parent)
        
        current_nodes = list(parent_nodes)
        candidate_norms = [nball_norms[label_to_index[node]] * 1e-5 for node in current_nodes]
        candidate_directions = [nball_embeddings[label_to_index[node]] for node in current_nodes]
        candidate_radii = [nball_radius[label_to_index[node]] for node in current_nodes]
    
    return None  # Return None if no suitable candidate is found


def evaluate_shoot(model_n, model_d, dataloader, nball_norms, nball_embeddings, nball_radius, \
                   device, lemma_pair, tree, index_to_label, label_to_index):
    model_n.eval()
    model_d.eval()

    loss_fn_n = log_cosh_loss
    loss_fn_d = nn.CosineEmbeddingLoss()

    loss_n = 0.0
    loss_d = 0.0
    accuracy = 0.0
    correct_predictions = 0
    pred_indices = {}
    size = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", leave=True, position=0):
            batch_input_ids, batch_attention_masks, batch_word_indices, \
            batch_sense_indices, batch_lemma_indices, batch_idices = [b.to(device) for b in batch]

            # Forward pass
            norms = model_n(input_ids=batch_input_ids, attention_mask=batch_attention_masks, \
                            word_index=batch_word_indices).squeeze()
            directions = model_d(input_ids=batch_input_ids, attention_mask=batch_attention_masks, \
                                 word_index=batch_word_indices).squeeze()

            target_norms = nball_norms[batch_sense_indices]
            target_directions = nball_embeddings[batch_sense_indices]

            labels = torch.ones(directions.size(0), device=device)
            loss_n_batch = loss_fn_n(norms, target_norms * 1e-5)
            loss_d_batch = loss_fn_d(directions, target_directions, labels)
            loss_n += loss_n_batch.item()
            loss_d += loss_d_batch.item()

            # Predict
            output_embeddings = directions * norms.unsqueeze(-1)
            for i in range(len(output_embeddings)):
                size += 1
                output_emb = output_embeddings[i]
                candidate_indices = lemma_pair[batch_lemma_indices[i].item()]
                candidate_norms = [nball_norms[idx] * 1e-5 for idx in candidate_indices]
                candidate_directions = [nball_embeddings[idx] for idx in candidate_indices]
                candidate_radii = [nball_radius[idx] for idx in candidate_indices]
                predicted_index = predict_with_tree(tree, output_emb, candidate_indices, candidate_norms, candidate_directions, candidate_radii, nball_norms, nball_embeddings, nball_radius, index_to_label, label_to_index)
                pred_indices[batch_idices[i].item()] = predicted_index
                if predicted_index == batch_sense_indices[i].item():
                    correct_predictions += 1
                
        total_batches = len(dataloader)
        loss_n /= total_batches
        loss_d /= total_batches
        accuracy = correct_predictions / size

    return loss_n, loss_d, accuracy, pred_indices


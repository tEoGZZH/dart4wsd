import os
import json
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from util import load_balls, preprocess_data, load_tree
from dataset import nballDataset
from model import nballBertNorm, nballBertDirection
from train import train_shoot
from evaluation import evaluate_shoot

def train_model(train_params, mode, device, output_dim, dataloader, target_data, output_model):
    """
    Train a model based on provided parameters and save the trained model.

    Args:
        train_params (dict): Dictionary containing training parameters (lr, step_size, gamma, epoch, model_url).
        mode (str): The mode of the model to be trained ("norm" or "direction").
        device (torch.device): The device to train the model on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): The DataLoader for the training data.
        target_data (Any): The target data for training.
        output_model (str): The filename to save the trained model.
        output_dim (int): The output dimension for the model.

    Returns:
        nn.Module: The trained model.
    """
    try:
        # Model selection
        model_norm_url = train_params["model_url"]
        if mode == "norm":
            print("Generating norm")
            model = nballBertNorm(model_norm_url, output_dim).to(device)
        elif mode == "direction":
            model = nballBertDirection(model_norm_url, output_dim).to(device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Training parameters
        lr = train_params["lr"]
        step_size = train_params["step_size"]
        gamma = train_params["gamma"]
        epochs = train_params["epoch"]

        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Train the model
        print("Starting training...")
        train_shoot(model=model, dataloader=dataloader, optimizer=optimizer, scheduler=scheduler, 
                    num_epochs=epochs, mode=mode, device=device, data=target_data)
        print("Training completed.")

        # Save the model
        checkpoint_dir = '../data/output/checkpoint/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, output_model)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_url': model.model_url,
            'output_dim': model.projection.out_features
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        return model

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def evaluate_model(sense_labels, wsChildrenFile, output_model_norm, output_model_d, \
                   nball_norms, nball_embeddings, nball_radius, dataloader, device, lemma_pair):
    # Load the model
    checkpoint_dir = '../data/output/checkpoint/'
    checkpoint_path_n = os.path.join(checkpoint_dir, output_model_norm)
    checkpoint_path_d = os.path.join(checkpoint_dir, output_model_d)
    checkpoint_n = torch.load(checkpoint_path_n)
    checkpoint_d = torch.load(checkpoint_path_d)
    model_norm = nballBertNorm(model_url=checkpoint_n['model_url'], output_dim=checkpoint_n['output_dim'])
    model_norm.load_state_dict(checkpoint_n['model_state_dict'])
    model_norm.to(device)
    model_d = nballBertDirection(model_url=checkpoint_d['model_url'], output_dim=checkpoint_d['output_dim'])
    model_d.load_state_dict(checkpoint_d['model_state_dict'])
    model_d.to(device)
    
    
    label_to_index = {label: idx for idx, label in enumerate(sense_labels)}
    index_to_label = {idx: label for idx, label in enumerate(sense_labels)}
    tree = load_tree(wsChildrenFile)
    loss_n, loss_d, accurancy, pred_indices = evaluate_shoot(model_n=model_norm, model_d=model_d,dataloader=dataloader,\
                                                             nball_norms=nball_norms, nball_embeddings=nball_embeddings, \
                                                             nball_radius=nball_radius, device=device, lemma_pair=lemma_pair,\
                                                             label_to_index=label_to_index, index_to_label=index_to_label,\
                                                             tree=tree)
    
    print(f"Average loss of norm:{loss_n}, average loss of dorection:{loss_d}")
    print(f"Accurancy:{accurancy*100}%")
    # train_semcor['pred_sense_idx'] = train_semcor.index.map(pred_indices)
    # display(train_semcor.head())

    
def main():
    # Read parameters
    with open('model_params.json') as parms_file:
        params = json.load(parms_file)
    nball_path = params["nball_path"]
    train_path = params["train_path"]
    eval_path = params["eval_path"]
    max_length = params["max_length"]
    train_params_norm = params["train_params_norm"]
    train_params_d = params["train_params_d"]
    output_model_norm = params["output_model_norm"]
    output_model_d = params["output_model_d"]
    tree_path = params["tree_path"]
    
    
    # Load nball and preprocess training data
    nball = load_balls(nball_path)
    train_data, lemma_pair = preprocess_data(train_path, nball)

    # Load nball embeddings, make it to cuda if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sense_labels = list(nball.keys())
    nball_embeddings = [nball[label].center for label in sense_labels]
    nball_embeddings = torch.tensor(np.array(nball_embeddings), dtype=torch.float32).to(device)
    nball_norms = [nball[label].distance for label in sense_labels]
    nball_norms = torch.tensor(np.array(nball_norms), dtype=torch.float32).to(device)
    nball_radius = [nball[label].radius for label in sense_labels]
    nball_radius = torch.tensor(np.array(nball_radius), dtype=torch.float32).to(device)

    # Construct dataset
    model_norm_url = train_params_norm["model_url"]
    norm_train_batch_size = train_params_norm["batch_size"]
    dataset_train = nballDataset(train_data, nball, model_norm_url, max_length)
    dataloader_train = DataLoader(dataset_train, norm_train_batch_size, shuffle=True)

    # Set model parameters
    output_dim = len(nball_embeddings[0])
    # # Generate, train and save the model for norm
    # model_norm = train_model(train_params=train_params_norm, mode="norm", device=device, output_dim=output_dim,\
    #                          dataloader=dataloader_train, target_data=nball_norms, output_model=output_model_norm)
    # # Generate, train and save the model for direction
    # model_d = train_model(train_params=train_params_d, mode="direction", device=device, output_dim=output_dim,\
    #                       dataloader=dataloader_train, target_data=nball_embeddings, output_model=output_model_d)

    # Load evaluation data
    eval_data, _ = preprocess_data(eval_path, nball)
    model_norm_url = train_params_norm["model_url"]
    eval_batch_size = train_params_norm["batch_size"]
    dataset_eval = nballDataset(eval_data, nball, model_norm_url, max_length)
    dataloader_eval = DataLoader(dataset_eval, eval_batch_size, shuffle=True)
    evaluate_model(sense_labels=sense_labels, wsChildrenFile=tree_path, output_model_norm=output_model_norm, \
                   output_model_d=output_model_d, nball_norms=nball_norms, nball_embeddings=nball_embeddings,\
                   nball_radius = nball_radius, dataloader=dataloader_eval, device=device, lemma_pair=lemma_pair)
    
    

if __name__ == "__main__":
    main()
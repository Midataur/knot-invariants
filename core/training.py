from utilities import try_loading_model, save_model_and_config, SimCLRLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel
from constants_and_types import ConfigObject
from torch_datasets import get_dataset_and_loader
from accelerate import Accelerator
from tqdm.auto import tqdm
import plotly.express as px
import torch.optim as optim
import math
import torch
import random
import wandb

def train(config: ConfigObject):
    """
        The main model training function.
    """

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Logging in...")
        wandb.login()

        # load the data
        print("Loading data...")

    # set seed for deterministic behaviour
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)

    # try loading model and config
    model, config = try_loading_model(config)

    dataset, dataloader = get_dataset_and_loader(config, verbose=accelerator.is_local_main_process)

    # Define the loss function
    loss_function = SimCLRLoss(
        temperature=config.simclr_temp
    )

    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    lr_factor = config.lr_factor
    lr_patience = config.lr_patience
    threshold = config.threshold

    # Define the optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
        threshold=threshold
    )

    # set up accelerator
    model, optimizer, dataloader, scheduler, loss_function = accelerator.prepare(
        model, optimizer, dataloader, scheduler, loss_function
    )

    if accelerator.is_local_main_process:
        print("Training...")

        # train the model
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.wandb_project,

            # track run hyperparameters and metadata
            config=config.as_wandb_legal_dict(),
            settings=wandb.Settings(),
            resume="allow",
            id=config.model_name
        )

    epoch = 0

    last_train_loss = None
    last_val_loss = None

    # training loop
    while True:
        epoch += 1
        model.train()  # Set the model to training mode

        total_loss = 0.0
        num_batches = 0

        accelerator.print("Training...")
        
        for originals, transformed in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            # get rid of the weird third dimension that gets addded for some reason
            num_rows, _, __   = originals.shape
            
            original_codes    = originals.reshape((num_rows, -1))
            transformed_codes = transformed.reshape((num_rows, -1))

            # zero the gradients
            optimizer.zero_grad()  

            # calculate the embeddings
            originals_embedded   = model(original_codes)
            transformed_embedded = model(transformed_codes)

            # get the loss
            loss = loss_function(originals_embedded, transformed_embedded)

            # do backprop
            accelerator.backward(loss)
            optimizer.step()

            # track stats
            total_loss += loss.item()
            num_batches += 1

        # find what the loss would be if we had perfect orthogonality
        orthogonal_loss = -math.log(
            math.exp(1/config.simclr_temp)/(
                math.exp(1/config.simclr_temp) + 2*(num_rows-1)
            )
        )

        # find what the loss would be if everything was the same vector
        all_aligned_loss = -math.log(
            math.exp(1/config.simclr_temp)/(
                math.exp(1/config.simclr_temp)*(2*num_rows-1)
            )
        )

        train_loss = total_loss / num_batches

        # log the similarity matrix
        similarity_matrix = loss_function.calculate_similarities(
            originals_embedded, transformed_embedded
        ).tolist()

        metrics = {
            "loss": train_loss,
            "tensor_shape": originals_embedded.shape,
            "orthogonal_loss": orthogonal_loss,
            "all_aligned_loss": all_aligned_loss,
            "similarity_matrix": px.imshow(similarity_matrix, zmin=0, zmax=1)
        }

        # to show how fast we're plateauing
        if epoch > 1:
            metrics["delta_train_loss"] = train_loss - last_train_loss
        
        last_train_loss = train_loss

        if accelerator.is_local_main_process:
            accelerator.print(
                f"Epoch {epoch + 1}, Loss {train_loss}"
            )

            # log metrics to wandb
            wandb.log(metrics)
            
        # always save the model
        accelerator.wait_for_everyone()
        save_model_and_config(model, config, accelerator)
        
        # save embedding pictures so we can make gifs later
        # this is broken since we added accelerate
        # TODO: FIX this later
        # UPDATE: three projects later, this code is still here and broken
        # maybe one day :')

        # if accelerator.is_local_main_process:
        #     save_embedding_pictures(model)

        # learning rate scheduling
        scheduler.step(train_loss)
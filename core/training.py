from utilities import try_loading_state, save_state_and_config, try_loading_config, SimCLRLoss, sort_tensor_by_indices
from torch_datasets import get_dataset_and_loader
from constants_and_types import ConfigObject, TrainingState
from accelerate import Accelerator
from tqdm.auto import tqdm
import plotly.express as px
import model_types
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
    config = try_loading_config(config)

    ModelType = model_types.MODELS[config.model_type]
    model = ModelType(config)

    dataset, dataloader = get_dataset_and_loader(config, verbose=accelerator.is_local_main_process)

    # Define the loss function
    loss_function = SimCLRLoss(
        temperature=config.simclr_temp
    )

    # Define the optimizer and scheduler
    optimizer = config.optimizer(
        model.parameters()
    )

    scheduler = config.scheduler(
        optimizer,
    )

    # set up accelerator
    model, optimizer, dataloader, scheduler, loss_function = accelerator.prepare(
        model, optimizer, dataloader, scheduler, loss_function
    )

    # try loading the state
    try_loading_state(config, accelerator)

    if accelerator.is_local_main_process:
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

    # log model gradients
    wandb.watch(
        model, 
        loss_function, 
        log_freq=config.logging_frequency, 
        log="all"
    )

    last_train_loss = dataloader.dataset.training_state.current_loss

    # training loop
    while True:
        epoch = dataloader.dataset.training_state.epoch

        model.train()  # Set the model to training mode

        total_loss = 0.0
        num_batches = 0

        accelerator.print("Training...")

        # keep track of these in a seperate tensor for non-full batch training
        saved_embeddings = torch.empty((0, config.proj_dim), device=accelerator.device)
        
        # keep track of the true sample order so we can sort them again later.
        # (they're shuffled by the dataloader).
        saved_indices = torch.empty((0, 1), device=accelerator.device)
        
        for first, second, real_indices in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            # get rid of the weird third dimension that gets addded for some reason
            num_rows, _, __   = first.shape
            
            first_codes  = first.reshape((num_rows, -1))
            second_codes = second.reshape((num_rows, -1))

            # zero the gradients
            optimizer.zero_grad()  

            # calculate the embeddings
            first_embedded = model(first_codes)
            second_embedded = model(second_codes)

            # get the loss
            loss = loss_function(first_embedded, second_embedded)

            # do backprop
            accelerator.backward(loss)
            optimizer.step()

            # track stats
            total_loss += loss.item()
            num_batches += 1

            # save embeddings for later
            saved_embeddings = torch.cat((saved_embeddings, first_embedded))
            saved_indices = torch.cat((saved_indices, real_indices))

        train_loss = total_loss / num_batches

        # put the saved embeddings back in the correct order
        saved_embeddings = sort_tensor_by_indices(saved_embeddings, saved_indices)

        # calculate number of false negatives
        similarity_matrix = loss_function.calculate_similarities(
            saved_embeddings, saved_embeddings
        )

        num_knots = len(dataset)
        similarity_matrix_top_quad = similarity_matrix[:num_knots, :num_knots]
        top_quad_rounded = similarity_matrix_top_quad.round()

        # we subtract of identities and divide by two.
        # this gives the number of non-zero entries above the diagonal.
        collision_count = (top_quad_rounded.sum() - num_knots)/2

        metrics = {
            "loss": train_loss,
            "current_lr": scheduler.get_last_lr()[0],
            "tensor_shape": first_embedded.shape,
            "orthogonal_loss": config.orthogonal_loss(num_knots),
            "constant_fn_loss": config.constant_fn_loss(num_knots),
            "collision_count": collision_count
        }

        if epoch % config.logging_frequency == 0:
            # log the top quadrant
            metrics["similarity_matrix_top_quadrant"] = px.imshow(
                similarity_matrix_top_quad.tolist(), zmin=0, zmax=1
            )

            # log the rounded matrix
            # rounding tells us "are these more same or more different?"
            metrics["similarity_matrix_top_quadrant_rounded"] = px.imshow(
                top_quad_rounded.tolist(), zmin=0, zmax=1
            )

            # save embedding pictures so we can make gifs later

            # token embeddings
            tok_emb = model.token_embedding_table.weight.cpu().detach()
            tok_emb_similarity = loss_function.calculate_similarities(tok_emb, tok_emb)
            metrics["tok_emb_similarity"] = px.imshow(
                tok_emb_similarity[:tok_emb.shape[0], :tok_emb.shape[0]].tolist(), zmin=-1, zmax=1
            )

            # position embeddings
            pos_emb = model.position_embedding.weight.cpu().detach()
            pos_emb_similarity = loss_function.calculate_similarities(pos_emb, pos_emb)
            metrics["pos_emb_similarity"] = px.imshow(
                pos_emb_similarity[:pos_emb.shape[0], :pos_emb.shape[0]].tolist(), zmin=-1, zmax=1
            )

        # to show how fast we're plateauing
        if epoch > 1:
            metrics["delta_train_loss"] = train_loss - last_train_loss

        # update the current training state
        dataloader.dataset.set_training_state(TrainingState(
            epoch=epoch+1,
            current_loss=train_loss
        ))
        
        last_train_loss = train_loss

        if accelerator.is_local_main_process:
            accelerator.print(
                f"Epoch {epoch}, Loss {train_loss}"
            )

            # log metrics to wandb
            wandb.log(metrics)
            
        # always save the model
        accelerator.wait_for_everyone()
        save_state_and_config(config, accelerator)

        # learning rate scheduling
        scheduler.step(epoch=epoch)

        # potentially quit if we need to
        if config.max_epochs is not None and epoch > config.max_epochs:
            break
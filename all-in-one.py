# DDP Implementation modified to train across all datasets

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from loguru import logger
from rich.logging import RichHandler
from dataset import NewPersona, MovieLines, SherlockLiterature, MovieConversations
from model import BigramLangModel

def setup(rank, world_size):
    # Setup for distributed processes
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # Cleanup distributed processes
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    rank = args.local_rank
    torch.cuda.set_device(rank)
    world_size = torch.cuda.device_count()

    setup(rank, world_size)

    logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": 'INFO'}])

    # List of dataset instances
    datasets = [NewPersona(), MovieLines(), SherlockLiterature(), MovieConversations()]

    model = BigramLangModel(
        vocab_size=max(ds.vocab_size() for ds in datasets),  # Assuming all datasets have the same vocab_size
        block_size=args.block_size,
        embedding_size=args.embed_size,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(100001):
        opt.zero_grad()
        model.train()
        
        # Iterate over each dataset within a step
        for ds in datasets:
            inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size)
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))
            loss.backward()
        
        opt.step()

        # Logging and validation loss calculation for rank 0
        if step % 100 == 0 and rank == 0:
            model.eval()
            total_vloss = 0
            for ds in datasets:
                inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size, split="val")
                inputs, targets = inputs.to(rank), targets.to(rank)
                outputs = model(inputs)
                vloss = nn.functional.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))
                total_vloss += vloss.item()
            logger.info(f"Step: {step}, avg val loss: {total_vloss / len(datasets):.4f}")

        # Model saving for rank 0
        if step % 2000 == 0 and rank == 0:
            torch.save(model.state_dict(), "model.pt")

    cleanup()

if __name__ == "__main__":
    main()

# Note: This script iterates over all specified datasets within the training loop, allowing training across all data sources.
# Assumes all datasets have the same vocab_size; adjust if necessary.
# Model saving and logging are handled by the process with rank 0 to avoid redundancy.

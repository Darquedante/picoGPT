# DDP Implementation of the training script

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
    # setup for distributed processes
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # cleanup distributed processes
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

    ds = SherlockLiterature()
    model = BigramLangModel(
        vocab_size=ds.vocab_size(),
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

        inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size)
        outputs = model(inputs)  # shape: (batch_size, block_size, vocab_size)

        loss = nn.functional.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))
        loss.backward()
        opt.step()

        if step % 100 == 0 and rank == 0:  # Logging only by rank 0
            model.eval()
            inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size, split="val")
            outputs = model(inputs)
            vloss = nn.functional.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))
            logger.info(f"Step: {step}, train loss: {loss.item():.4f}, val loss: {vloss.item():.4f}")

        if step % 2000 == 0 and rank == 0:  # Model saving only by rank 0
            torch.save(model.state_dict(), "model.pt")

    cleanup()

if __name__ == "__main__":
    main()

# Note: The `setup` and `cleanup` functions are used to initialize and finalize the torch distributed environment.
# The `main` function now includes a `--local_rank` argument which is used by torch.distributed to specify the rank of the process.
# The `torch.cuda.set_device` call ensures that each process uses a specific GPU.
# The model is wrapped with `DistributedDataParallel`, enabling it to run in a distributed manner.
# Logging and model saving are performed only by the process with rank 0 to avoid redundancy.

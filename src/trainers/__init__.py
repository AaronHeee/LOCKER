from .bert import BERTAllTrainer

TRAINERS = {
    BERTAllTrainer.code(): BERTAllTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq)
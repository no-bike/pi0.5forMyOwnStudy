from openpi.training.data_loader import create_data_loader

def read_one_batch(train_config):
    loader = create_data_loader(train_config, num_batches=1, skip_norm_stats=True)
    batch = next(iter(loader))
    return batch

def show_action_shape(batch):
    observation, actions = batch
    print("动作形状：", actions.shape)

if __name__ == "__main__":
    from train_config import build_train_config

    config = build_train_config()
    batch = read_one_batch(config)
    show_action_shape(batch)
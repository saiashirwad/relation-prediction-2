from train import * 

kg_train, kg_test, kg_val = load_fb15k237()

dataloader = DataLoader(kg_train, batch_size=10, shuffle=False, pin_memory=True)
ent_embed, rel_embed = get_init_embed()

data = [d for d in dataloader]
triplets = ent_emed, rel_embed = get_init_embed()

# data [dfor d in dataloadera]- e
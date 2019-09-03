import torch
root = "./cifar100_resnet56_5.pth"
checkpoint_params = torch.load(root)
new_checkpoint = {}
new_checkpoint["fc_opt"] = checkpoint_params["fc_opt"]
new_checkpoint["model"] = checkpoint_params["model"]
new_checkpoint["seg_opt"] = checkpoint_params["seg_opt"]
aux_fc_list = []
for single_fc in checkpoint_params["aux_fc"]:
    aux_fc_list.append(single_fc.state_dict())
    print type(single_fc)
new_checkpoint["aux_fc"] = aux_fc_list
torch.save(new_checkpoint, "./cifar100_resnet56_5_1.pth")
print checkpoint_params.keys()

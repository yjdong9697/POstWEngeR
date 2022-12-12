import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
	
def check_device():
	print("### Device Check list ###")
	print("GPU available?:", torch.cuda.is_available())
	device_number = torch.cuda.current_device()
	print("Device number:", device_number)
	print("Is device?:", torch.cuda.device(device_number))
	print("Device count?:", torch.cuda.device_count())
	print("Device name?:", torch.cuda.get_device_name(device_number))
	print("### ### ### ### ### ###\n\n")

def save_model(model, step, dir):
	fname = "{:06d}_model.pt"
	torch.save(model.state_dict(), dir+fname.format(step))
	print("Model saved.")
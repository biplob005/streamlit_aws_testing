



labels = ['green', 'red', 'yellow']
model2 = torch.load('helment_no_helmet98.7.pth', map_location=device) 
model2.eval()

transform = transforms.Compose([
			transforms.Resize(92),  # resize img ... square image
			transforms.CenterCrop(90), # then center crop it
			transforms.ToTensor(), # convert into tensor format
			transforms.Normalize([0.5], [0.5]) # normalize tensors.
		  ]) 


def img_classify(frame):
	frame = transform(Image.fromarray(frame))
	frame = frame.unsqueeze(0)
	prediction = model2(frame)
	result_idx = torch.argmax(prediction)
	# prediction_conf = sorted(prediction[0]) 
	return True if result_idx == 0 else False
from skimage.transform import resize
import torch
import numpy as np

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

def whatIs(array, msg=""):
    #print(f'{msg} type {type(array)}, {array.dtype}, {array.shape}')
    return

class U2netModel:
    # Constructor loads the model: u2net or u2netp
    def __init__(self, model_name='u2net'):
        if(model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            self.net = U2NET(3,1)
        elif(model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            self.net = U2NETP(3,1)
        else:
            raise ValueError('Invalid model: ' + model_name)

        model_path = 'model/' + model_name + '.pth'
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.net.eval()

        self.output_size = 320

    
    # Normalize image and make a tensor out of it
    # image: ndarray rgb image
    # Code extracted from data_loader.py
    def image2Tensor(self, image):
        # resize always casts to float64, astype to float32
        image = resize(image,(self.output_size,self.output_size),mode='constant').astype(np.float32)

        # elements range 0.0..1.0
        image = image/np.max(image)

        # center and scale
        tmpImg = np.zeros((image.shape[0],image.shape[1],3), dtype=np.float32)
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
        
        # channels go first
        tmpImg = tmpImg.transpose((2, 0, 1))

        tensor = torch.from_numpy(tmpImg[np.newaxis, :])
        tensor = tensor.type(torch.FloatTensor)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        whatIs(tensor, 'image2Tensor inputs_test')
        return tensor


    def predict(self, tensor):
        whatIs(tensor, 'predict tensor')
        d1,d2,d3,d4,d5,d6,d7= self.net(tensor)
        output_map = d1[:,0,:,:]
        whatIs(output_map, 'predict pred')
        output_map = self.normPRED(output_map)
        whatIs(output_map, 'predict after norm pred')
        return output_map

    # normalize the predicted SOD probability map
    # elements will range from 0.0 to 1.0
    def normPRED(self, tensor):
        element_max = torch.max(tensor)
        element_min = torch.min(tensor)

        normalized_tensor = (tensor-element_min)/(element_max-element_min)
        return normalized_tensor


    # map image gets its dims from im
    # map image in grayscale
    def mapTensor2Image(self, map_tensor, input_image):
        whatIs(map_tensor, 'mapTensor2Image map_tensor')
        map_np = map_tensor.squeeze().cpu().data.numpy()*255
        whatIs(map_np, 'mapTensor2Image map_np')

        map_image = resize(map_np, input_image.shape[0:2]).astype(np.uint8)
        whatIs(map_image, 'mapTensor2Image map_image')

        return map_image

    def __call__(self, input_image):
        whatIs(input_image, 'input_image')
        image_tensor = self.image2Tensor(input_image)
        whatIs(image_tensor, 'image_tensor')
        map_tensor = self.predict(image_tensor)
        whatIs(map_tensor, 'map_tensor')
        map_image = self.mapTensor2Image(map_tensor, input_image)
        whatIs(map_image, 'map_image')
        return map_image
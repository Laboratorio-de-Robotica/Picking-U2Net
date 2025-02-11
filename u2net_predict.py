from skimage.transform import resize
import torch
import numpy as np

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# Debug info, deactivate it by commenting out print statement
def whatIs(array, msg=""):
    #print(f'{msg} type {type(array)}, {array.dtype}, {array.shape}')
    return

class U2netModel:
    """U2netModel para segmentación de objetos salientes en imágenes.

    Attributes:
        net (torch.nn.Module): Modelo U2Net para segmentar la imagen.
        output_size (int): Tamaño de salida de la imagen segmentada.
    """

    def __init__(self, model_name:str='u2net'):
        """Inicializa el objeto U2netModel con el modelo especificado.
        
        Args:
            model_name (str, optional): Nombre del modelo a usar, u2net o u2netp. Por defecto "u2net".            
        """
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
    def image2Tensor(self, image:np.ndarray):
        """Convierte una imagen en un tensor normalizado para ser procesado por U2Net.
        
        Args:
            image (np.ndarray): Imagen de entrada.

        Returns:
            torch.Tensor: Tensor normalizado de la imagen.
        """
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


    def predict(self, tensor:torch.Tensor):
        """Predice la segmentación de la imagen a partir del tensor de entrada.
        
        Args:
            tensor (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida con la segmentación de la imagen.
        """
        whatIs(tensor, 'predict tensor')
        d1,d2,d3,d4,d5,d6,d7= self.net(tensor)
        output_map = d1[:,0,:,:]
        whatIs(output_map, 'predict pred')
        output_map = self.normPRED(output_map)
        whatIs(output_map, 'predict after norm pred')
        return output_map

    def normPRED(self, tensor:torch.Tensor):
        """Normaliza el tensor de salida de la segmentación.
        
        Args:
            tensor (torch.Tensor): Tensor de salida.

        Returns:
            torch.Tensor: Tensor normalizado con valores entre 0.0 y 1.0 .
        """
        element_max = torch.max(tensor)
        element_min = torch.min(tensor)

        normalized_tensor = (tensor-element_min)/(element_max-element_min)
        return normalized_tensor


    def mapTensor2Image(self, map_tensor:torch.Tensor, input_image:np.ndarray):
        """Convierte el tensor de segmentación en una imagen en escala de grises.
        
        Args:
            map_tensor (torch.Tensor): Tensor de segmentación.
            input_image (np.ndarray): Imagen de entrada, sólo para tomar sus medidas.

        Returns:
            np.ndarray: Imagen de la segmentación en escala de grises, del mismo tamaño que input_image.
        """

        whatIs(map_tensor, 'mapTensor2Image map_tensor')
        map_np = map_tensor.squeeze().cpu().data.numpy()*255
        whatIs(map_np, 'mapTensor2Image map_np')

        map_image = resize(map_np, input_image.shape[0:2]).astype(np.uint8)
        whatIs(map_image, 'mapTensor2Image map_image')

        return map_image

    def __call__(self, input_image):
        """Procesa la imagen de entrada y devuelve la segmentación de la imagen.

        Este método ejecuta en secuencia otros métodos de la clase para procesar la imagen de entrada y devolver la segmentación.

        Args:
            input_image (np.ndarray): Imagen de entrada.

        Returns:
            np.ndarray: Imagen de la segmentación en escala de grises, del mismo tamaño que input_image
        """
        whatIs(input_image, 'input_image')
        image_tensor = self.image2Tensor(input_image)
        whatIs(image_tensor, 'image_tensor')
        map_tensor = self.predict(image_tensor)
        whatIs(map_tensor, 'map_tensor')
        map_image = self.mapTensor2Image(map_tensor, input_image)
        whatIs(map_image, 'map_image')
        return map_image
    
if __name__ == '__main__':
    import argparse
    #import cv2 as cv
    from skimage import io
    import matplotlib.pyplot as plt

    print('u2net/u2netp prediction')
    print('Processes the provided input image, shows the prediction SOD heatmap, and optionally saves it')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image file path", default="images/imagen_13r.jpeg")
    parser.add_argument("-o", "--output", help="output image file path", default="")
    parser.add_argument("-m", "--model", help="model, either u2net (default) or u2netp", default="u2net")
    args = parser.parse_args()

    model_name = args.model
    input_image_path = args.input
    output_image_path = args.output

    # Input image ndarray
    #input_image = cv.imread(input_image_path)
    input_image = io.imread(input_image_path)

    model = U2netModel(model_name)
    output_image = model(input_image)

    if(output_image_path):
        io.imsave(output_image_path, output_image)

    #cv.imshow('map', output_image)
    #cv.waitKey(0)
    plt.imshow(output_image)
    plt.show()
from params_proto import Proto, ParamsProto
import cv2

class FileArgs(ParamsProto):
    source_path: str = Proto(help='Path to the audio file.')


def get_similarity(audio, img):
    """
    Returns 2D tensor with same shape as image measuring similarity of each 
    pixel of the image against the audio on a scale of 0-1 
    audio: (512) 1D tensor embedding
    img: (N,M,512) 3D tensor embedding, dim0=img.height, dim1=img.width,dim2=tensor embedding
    """
    return img @ audio


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (21, 21), 0)

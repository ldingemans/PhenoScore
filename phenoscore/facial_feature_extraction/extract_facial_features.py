from deepface import DeepFace
import os
from deepface.DeepFace import build_model
from deepface.commons import functions
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from QMagFace.preprocessing.insightface.src.mtcnn_detector import MtcnnDetector
from QMagFace.preprocessing.align import preprocess
from QMagFace.preprocessing.magface.network_inf import builder_inf
import numpy as np
import cv2
import torch
import torchvision.transforms
from collections import namedtuple
import io
import tensorflow as tf
import urllib.request

class FacialFeatureExtractor:
    def __init__(self):
        """
        Constructor
        """
        pass

    def process_file(self, path_to_img):
        """
        Extract facial features for an image

        Parameters
        ----------
        path_to_img: str
            Path to image to process
        """
        pass

    def get_norm_image(self, img_path):
        pass

    def predict_aligned_img(self, X_face):
        pass

class VGGFaceExtractor(FacialFeatureExtractor):
    def __init__(self):
        """
        Constructor
        """
        self.face_vector_size = 2622
        self.input_image_size = (224, 224)
        self.build_model = build_model("VGG-Face")
        pass

    def process_file(self, path_to_img):
        """
        Extract facial features for an image

        Parameters
        ----------
        path_to_img: str
            Path to image to process
        """
        try:
            result = DeepFace.represent(path_to_img, model_name='VGG-Face', detector_backend='mtcnn')
        except ValueError as e:
            if 'Face could not be detected.' in str(e):
                result = None
            else:
                raise e
        return result

    @staticmethod
    def get_norm_image(img_path):
        """
        Preprocess the image for VGG-Face, using MTCNN (detect face, alignment, etc)

        Parameters
        ----------
        img_path: str
            Path to the image to process

        Returns
        -------
        img_tensor: numpy array
            The preprocessed image in array form
        """
        input_shape_x, input_shape_y = 224, 224

        img = functions.preprocess_face(img=img_path
                                        , target_size=(input_shape_y, input_shape_x)
                                        , enforce_detection=False
                                        , detector_backend='mtcnn'
                                        , align=True,
                                        )

        img_tensor = functions.normalize_input(img=img, normalization='base')
        return img_tensor[0]

    def predict_aligned_img(self, X_face):
        return self.build_model.predict(X_face, verbose=False)


class QMagFaceExtractor(FacialFeatureExtractor):
    def __init__(self, path_to_dir, use_cpu='auto'):
        """
        Constructor
        """
        if use_cpu == 'auto':
            devices = tf.config.list_physical_devices('GPU')
            if len(devices) == 0:
                self._use_cpu = True
            else:
                self._use_cpu = False
        else:
            self._use_cpu = use_cpu
        self.face_vector_size = 512
        self.input_image_size = (112, 112)
        self._path_to_dir = path_to_dir
        self._det = MtcnnDetector(
            model_folder=os.path.join(self._path_to_dir, 'QMagFace', '_models', 'mtcnn-model'),
            accurate_landmark=True,
            minsize=50,
            threshold=[0.6, 0.7, 0.8]
        )
        Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
        path_pretrained_weights = os.path.join(self._path_to_dir, 'QMagFace', '_models', 'magface_models', 'magface_epoch_00025.pth')
        args = Args('iresnet100', path_pretrained_weights, 512, self._use_cpu)

        # Check if the file already exists locally
        if os.path.isfile(path_pretrained_weights):
            print("Pretrained weights already exist, skipping download.")
        else:
            # Send a GET request to download the file
            print("Pretrained weights not yet available, downloading them now.")
            file_url = 'https://www.dropbox.com/s/jube2kt201muqki/magface_epoch_00025.pth?dl=1'
            dirname = os.path.dirname(path_pretrained_weights)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            def show_progress(block_num, block_size, total_size):
                print(str(round(block_num * block_size / total_size * 100, 2)) + '%', end="\r")

            urllib.request.urlretrieve(file_url, path_pretrained_weights, show_progress)
            print("Downloaded pretrained weights.")

        model = builder_inf(args)
        self.build_model = torch.nn.DataParallel(model)

    def process_file(self, path_to_img):
        """
        Extract facial features for an image

        Parameters
        ----------
        path_to_img: str
            Path to image to process
        """
        p_img = self.get_norm_image(path_to_img)
        if p_img is not None:
            embedding = self.predict_aligned_img(p_img)
            if embedding is not None:
                embedding = list(embedding[0])
        else:
            embedding = None
        return embedding

    def get_norm_image(self, path_to_img):
        if type(path_to_img) == str:
            p_img = preprocess(self._det, cv2.imread(path_to_img))
        else:
            p_img = preprocess(self._det, path_to_img)
        #  normally aligned images get saved to disc, and imwrite changes the array a bit
        #  so for consistent performance we need to do this in memory
        try:
            is_success, buffer = cv2.imencode(".jpg", p_img)
            io_buf = io.BytesIO(buffer)
            decode_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
            decode_img = decode_img / 255
            decode_img = decode_img.astype(np.float32)
        except:
            decode_img = None
        return decode_img

    def predict_aligned_img(self, X_face):
        self.build_model.eval()
        trans = torchvision.transforms.ToTensor()
        if X_face.shape == (112, 112, 3):
            with torch.no_grad():
                try:
                    input_ = trans(X_face).unsqueeze(0)
                    if self._use_cpu:
                        embedding = self.build_model(input_).to('cpu')
                        embedding = embedding.numpy()
                    else:
                        embedding = self.build_model(input_).to('cuda')
                        embedding = embedding.cpu().numpy()
                except:
                    embedding = None
        else:
            all_inputs = torch.tensor([], dtype=torch.float32)
            with torch.no_grad():
                for img in X_face:
                        input_ = trans(img).unsqueeze(0)
                        all_inputs = torch.cat([all_inputs, input_])
                if self._use_cpu:
                    embedding = self.build_model(all_inputs).to('cpu').numpy()
                else:
                    embedding = self.build_model(all_inputs).to('cuda').cpu().numpy()
        return embedding

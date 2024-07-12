import os
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
import urllib.request
import pandas as pd
import json


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


class QMagFaceExtractor(FacialFeatureExtractor):
    def __init__(self, path_to_dir, use_cpu='auto'):
        """
        Constructor
        """
        if use_cpu == 'auto':
            devices = torch.cuda.device_count()
            if devices == 0:
                self._use_cpu = True
            else:
                self._use_cpu = False
        else:
            self._use_cpu = use_cpu
        self.face_vector_size = 512
        self.input_image_size = (112, 112)
        self._path_to_dir = path_to_dir
        self._det = MtcnnDetector(
            model_folder=os.path.join(self._path_to_dir, 'QMagFace', 'models_qmag', 'mtcnn-model'),
            accurate_landmark=True,
            minsize=50,
            threshold=[0.6, 0.7, 0.8]
        )
        Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
        path_pretrained_weights = os.path.join(self._path_to_dir, 'QMagFace', 'models_qmag', 'magface_models', 'magface_epoch_00025.pth')
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


class GestaltMatcherFaceExtractor(FacialFeatureExtractor):
    def __init__(self, path_to_dir, use_cpu='auto'):
        if use_cpu == 'auto':
            devices = torch.cuda.device_count()
            if devices == 0:
                self._use_cpu = True
            else:
                self._use_cpu = False
        else:
            self._use_cpu = use_cpu
        sys.path.append(os.path.join(path_to_dir))
        sys.path.append(os.path.join(path_to_dir, 'GestaltEngine-FaceCropper-retinaface'))
        from detect_pipe import init_model
        from predict_ensemble import load_models

        model_detect, device_detect = init_model(self._use_cpu)
        models, device = load_models(self._use_cpu)

        self._models, self._device, self._model_detect, \
            self._device_detect = models, device, model_detect, device_detect
        self.face_vector_size = 1536
        self.input_image_size = (112, 112)
        return

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
        else:
            embedding = None
        return embedding

    def get_norm_image(self, path_to_img):
        from detect_pipe import detect_pipe
        from align_pipe import align_pipe
        coords, img_rot = detect_pipe(path_to_img, self._model_detect, self._device_detect, self._use_cpu)
        aligned_img = align_pipe(coords, img_rot)
        return aligned_img

    def predict_aligned_img(self, aligned_img):
        from predict_ensemble import predict_memory
        representation = np.array(predict_memory(self._models, self._device, aligned_img))
        return np.concatenate([np.mean(representation[:4, :], axis=0), np.mean(representation[4:8, :], axis=0),
                               np.mean(representation[8:, :], axis=0)])
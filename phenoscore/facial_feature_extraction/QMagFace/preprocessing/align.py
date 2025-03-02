import os
import click
import tqdm
import cv2
from QMagFace.utils.files import list_all_files
from QMagFace.preprocessing.insightface.src.mtcnn_detector import MtcnnDetector
from QMagFace.preprocessing.insightface.src import face_preprocess


def preprocess(det, img):
    detected = det.detect_face(img, det_type=0)
    if detected is None:
        return None

    bbox, points = detected
    if bbox.shape[0] == 0:
        return None

    points = points[0, :].reshape((2, 5)).T
    image = face_preprocess.preprocess(img, bbox, points, image_size="112,112")
    return image


@click.command()
@click.option('--result_dir', '-r', type=click.Path())
@click.option('--source_dir', '-s', type=click.Path())
@click.option('--model_path', '-m', type=click.Path(), default='models_qmag/mtcnn-model/')
def main(result_dir, source_dir, model_path):
    os.makedirs(result_dir, exist_ok=True)
    det = MtcnnDetector(
        model_folder=model_path,
        accurate_landmark=True,
        minsize=50,
        threshold=[0.6, 0.7, 0.8]
    )

    filenames = list_all_files(source_dir)
    for filename in tqdm.tqdm(filenames):
        img = cv2.imread(filename)
        p_img = preprocess(det, img)
        if p_img is None:
            continue
        results_filename = filename.replace(source_dir, result_dir)
        os.makedirs(results_filename[:results_filename.rindex('/')], exist_ok=True)
        cv2.imwrite(results_filename, p_img)
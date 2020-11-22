import numpy as np
import json


def multivariate_gaussian(pos, mu, sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(sigma)
    Sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def get_head_pose_and_gaze(data_dict):
    """
    Retrieves head_pose and gaze_direction from .json file
    """

    head_pose = data_dict['head_pose']
    if head_pose is None:
        exit(-1)
    head_pose = [float(val) for val in str(head_pose)[1:-1].replace(',', '').split(' ')][:-1]
    head_pose[0] = (head_pose[0] + 180) % 360 - 180
    head_pose[1] = -(head_pose[1] - 180)

    look_vec = data_dict["eye_details"]
    if look_vec is None:
        exit(-1)
    look_vec = look_vec["look_vec"]
    if look_vec is None:
        exit(-1)
    ax, ay, az, _ = [float(val) for val in str(look_vec)[1:-1].replace(',', '').split(' ')]
    horizontal = np.rad2deg(np.arctan(ax / -az))
    vertical = np.rad2deg(np.arctan(ay / -az))

    return {
        "head_pose": head_pose,
        "gaze": [vertical, horizontal]
    }


def load_landmarks(json_path):
    def str_to_float(row):

        return list(map(float, row[1:-1].split(", ")))

    # load json
    with open(json_path, encoding="utf8") as f:
        data_dict = json.load(f)

    for key in data_dict.keys():
        if key in ["interior_margin_2d", "caruncle_2d", "iris_2d"]:
            data_dict[key] = [str_to_float(row) for row in data_dict[key]]
        elif key in ["eye_details"]:
            data_dict[key] = {
                "look_vec": str_to_float(data_dict[key]["look_vec"]),
                "pupil_size": float(data_dict[key]["pupil_size"]),
                "iris_size": float(data_dict[key]["iris_size"]),
                "iris_texture": data_dict[key]["iris_texture"]
            }
        elif key in ["lighting_details"]:
            data_dict[key] = {
                "skybox_texture": data_dict[key]["skybox_texture"],
                "skybox_exposure": float(data_dict[key]["skybox_exposure"]),
                "skybox_rotation": int(data_dict[key]["skybox_rotation"]),
                "ambient_intensity": float(data_dict[key]["ambient_intensity"]),
                "light_rotation": str_to_float(data_dict[key]["light_rotation"]),
                "light_intensity": float(data_dict[key]["light_intensity"]),
            }
        elif key in ["eye_region_details"]:
            data_dict[key] = {
                "pca_shape_coeffs": list(map(float, data_dict[key]["pca_shape_coeffs"])),
                "primary_skin_texture": data_dict[key]["primary_skin_texture"]
            }
        elif key in ["head_pose"]:
            data_dict[key] = str_to_float(data_dict[key])
        else:
            raise Exception("Key not found")
    return data_dict

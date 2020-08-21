import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import tensorflow as tf
from os import makedirs
from os.path import exists
import keras
from properties import IMG_SIZE, MODEL

load_model = keras.models.load_model
mnist = keras.datasets.mnist

dataset = "mnist"
target = "test"
save_path = "tmp"
batch_size = 128
upper_bound = 2000
n_bucket = 1000
num_classes = 10

if not exists("tmp"):
    makedirs("tmp")



def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred
    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.
    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )


def get_ats(
    model,
    dataset,
    name,
    layer_names,
    save_path=None,
    batch_size=128,
    is_classification=True,
    num_classes=10,
    num_proc=10,
):
    """Extract activation traces of dataset from model.
    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.
    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = "[" + name + "] "
    if is_classification:
        #p = Pool(num_proc)
        #print(prefix + "Model serving")
        pred = np.argmax(model.predict(dataset, batch_size=batch_size, verbose=1), axis=1)
        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

       # print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            #print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                p = Pool(num_proc)
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                )
                print(len(dataset))
                exit()
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None


    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


def _get_train_ats(model, x_train, layer_names):
    """Extract ats of train and target inputs. If there are saved files, then skip it.
    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        layer_names (list): List of selected layer names.
    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
    """

    saved_train_path = _get_saved_path(save_path, "mnist", "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        #print(("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=num_classes,
            is_classification=True,
            save_path=saved_train_path,
        )
        #print(("train ATs is saved at " + saved_train_path[0]))

    return train_ats, train_pred


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.
    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def get_at(
    model,
    test_input,
    name,
    layer_names,
    batch_size=128,
    is_classification=True,
    num_classes=10,
    num_proc=10,
):
    """Extract activation traces of dataset from model.
    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.
    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = "[" + name + "] "
    if is_classification:
        #TODO: issue pool
        #p = Pool(num_proc)
        with Pool(processes=10) as p:
            # print(prefix + "Model serving")
            pred = np.argmax(model.predict(test_input, batch_size=batch_size, verbose=0), axis=1)
            if len(layer_names) == 1:
                layer_outputs = [
                    temp_model.predict(test_input, batch_size=batch_size, verbose=0)
                ]
            else:
                layer_outputs = temp_model.predict(
                    test_input, batch_size=batch_size, verbose=0
                )

            #print(prefix + "Processing AT")
            at = None
            for layer_name, layer_output in zip(layer_names, layer_outputs):
                #print("Layer: " + layer_name)
                if layer_output[0].ndim == 3:
                    #TODO
                    #p = Pool(num_proc)
                    # For convolutional layers
                    layer_matrix = np.array(
                        p.map(_aggr_output, [layer_output[i] for i in range(len(test_input))])
                    )
                else:
                    layer_matrix = np.array(layer_output)

                if at is None:
                    at = layer_matrix
                else:
                    at = np.append(at, layer_matrix, axis=1)
                    layer_matrix = None


    return at, pred



def _get_target_at(model, test_input, layer_names):
    """Extract ats of train and target inputs. If there are saved files, then skip it.
    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.
    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """

    at, prediction = get_at(
        model,
        test_input,
        "test",
        layer_names,
        num_classes=num_classes,
        is_classification=True
    )
    return at, prediction


def calculate_dsa(model, x_target, train_ats, class_matrix, layer_names, all_idx):

    target_at, target_prediction = _get_target_at(
        model, x_target, layer_names
    )

    label = int(target_prediction)
    a_dist, a_dot = find_closest_at(target_at, train_ats[class_matrix[label]])
    b_dist, _ = find_closest_at(
        a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
    )
    dsa = a_dist / b_dist

    prediction = target_prediction


    return dsa, prediction


class DSA_calculator:
    # Load the pre-trained model.
    CLIP_MIN = -0.5
    CLIP_MAX = 0.5
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = load_model(MODEL)
    #model.summary()

    # You can select some layers you want to test.
    # layer_names = ["activation_1"]
    # layer_names = ["activation_2"]
    layer_names = ["activation_3"]  # TODO Check out which layer we should pick - I just took the same as Kim
    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    train_ats, train_pred = _get_train_ats(
        model, x_train, layer_names
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    @staticmethod
    def predict(img, label, seed):

        test_dsa, prediction = calculate_dsa(DSA_calculator.model, img,
                                             DSA_calculator.train_ats, DSA_calculator.class_matrix,
                                             DSA_calculator.layer_names, DSA_calculator.all_idx)

        correctly_classified = prediction == label
        confidence = test_dsa
        return correctly_classified, confidence, prediction

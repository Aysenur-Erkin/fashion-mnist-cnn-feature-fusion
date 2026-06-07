import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Veri setinin tamamını kullanmak uzun süreceğinden. Belirli sayıda örnek kullanacağım.
TRAIN_LIMIT = 4000
VAL_LIMIT = 800
TEST_LIMIT = 1200
PICTURE_SIZE = 96
BATCH_NO = 16
MLP_TURN = 8
FINETUNE_TURN = 2
OPEN_LAST_LAYER_COUNT = 25

DO_FINE_TUNING = True
RANDOM_NO = 42

USED_NETS = ["ResNet50", "MobileNetV2", "EfficientNetB0"]

LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_fashion_data():
    (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.fashion_mnist.load_data()

    random_maker = np.random.default_rng(RANDOM_NO)

    train_order = random_maker.permutation(len(x_train_all))
    test_order = random_maker.permutation(len(x_test_all))

    train_ids = train_order[:TRAIN_LIMIT]
    val_ids = train_order[TRAIN_LIMIT:TRAIN_LIMIT + VAL_LIMIT]
    test_ids = test_order[:TEST_LIMIT]

    train_x = x_train_all[train_ids]
    train_y = y_train_all[train_ids]

    val_x = x_train_all[val_ids]
    val_y = y_train_all[val_ids]

    test_x = x_test_all[test_ids]
    test_y = y_test_all[test_ids]

    print("Veri seti yuklendi")
    print("Train:", train_x.shape, "Val:", val_x.shape, "Test:", test_x.shape)
    print("Sinif sayisi:", len(LABEL_NAMES))

    return train_x, train_y, val_x, val_y, test_x, test_y


def choose_cnn(net_name):
    net_list = {
        "ResNet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
        "MobileNetV2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
        "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input)
    }

    if net_name in net_list:
        return net_list[net_name]

    print(net_name, "listede yok, bu model atlandi")
    return None, None


# FEATURE EXTRACTION
def make_cnn_input(images, labels, prep_func, only_image=True, shuffle=False):
    def fix_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, (PICTURE_SIZE, PICTURE_SIZE))
        image = prep_func(image)

        if only_image:
            return image
        return image, label

    data = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        data = data.shuffle(2000, seed=RANDOM_NO)

    data = data.map(fix_image)
    data = data.batch(BATCH_NO)
    return data


def create_feature_net(net_name):
    NetClass, _ = choose_cnn(net_name)

    if NetClass is None:
        return None

    net = NetClass(
        weights="imagenet",
        include_top=False,
        input_shape=(PICTURE_SIZE, PICTURE_SIZE, 3),
        pooling="avg"
    )
    net.trainable = False
    return net


def take_features(net_name, train_x, train_y, test_x, test_y):
    print("\n", net_name, "icin feature extraction basladi")

    _, prep_func = choose_cnn(net_name)
    if prep_func is None:
        return None, None

    net = create_feature_net(net_name)
    if net is None:
        return None, None

    train_ready = make_cnn_input(train_x, train_y, prep_func, only_image=True)
    test_ready = make_cnn_input(test_x, test_y, prep_func, only_image=True)

    start_time = time.time()
    train_feature = net.predict(train_ready, verbose=1)
    test_feature = net.predict(test_ready, verbose=1)
    spent_time = (time.time() - start_time) / 60

    print(net_name, "feature boyutu:", train_feature.shape[1])
    print("Gecen sure:", round(spent_time, 2), "dakika")

    return train_feature, test_feature


def normalize_feature(train_feature, test_feature):
    scaler = StandardScaler()
    train_new = scaler.fit_transform(train_feature)
    test_new = scaler.transform(test_feature)
    return train_new, test_new


# MLP
def make_mlp(in_size, label_count):
    mlp = models.Sequential()
    mlp.add(layers.Input(shape=(in_size,)))
    mlp.add(layers.Dense(256, activation="relu"))
    mlp.add(layers.Dropout(0.30))
    mlp.add(layers.Dense(128, activation="relu"))
    mlp.add(layers.Dropout(0.20))
    mlp.add(layers.Dense(label_count, activation="softmax"))

    mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return mlp


def run_mlp(test_name, train_feature, train_label, test_feature, test_label, label_count):
    print("\nDeney:", test_name)

    mlp = make_mlp(train_feature.shape[1], label_count)

    start_time = time.time()
    history = mlp.fit(
        train_feature,
        train_label,
        epochs=MLP_TURN,
        batch_size=BATCH_NO,
        validation_split=0.20,
        verbose=1
    )
    spent_time = (time.time() - start_time) / 60

    pred_prob = mlp.predict(test_feature, verbose=0)
    pred_label = np.argmax(pred_prob, axis=1)

    acc = accuracy_score(test_label, pred_label)
    pre = precision_score(test_label, pred_label, average="macro", zero_division=0)
    rec = recall_score(test_label, pred_label, average="macro", zero_division=0)
    f1 = f1_score(test_label, pred_label, average="macro", zero_division=0)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))

    row = {
        "experiment": test_name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "time_min": spent_time,
        "epoch": len(history.history["loss"]),
        "feature_dim": train_feature.shape[1]
    }
    return row


# FEATURE FUSION
def join_features(feature_box, selected, side):
    parts = []
    for net_name in selected:
        parts.append(feature_box[net_name][side])

    joined = np.concatenate(parts, axis=1)
    return joined


def weighted_join(feature_box, selected, side):
    smallest_size = min(feature_box[net_name][side].shape[1] for net_name in selected)

    if len(selected) == 2:
        rate_list = [0.5, 0.5]
    else:
        rate_list = [0.4, 0.3, 0.3]

    first_net = selected[0]
    total_feature = np.zeros((feature_box[first_net][side].shape[0], smallest_size))

    for rate, net_name in zip(rate_list, selected):
        one_feature = feature_box[net_name][side]
        one_feature = one_feature[:, :smallest_size]
        total_feature = total_feature + rate * one_feature

    return total_feature


# FINE-TUNING
def make_finetune_input(images, labels, prep_func, shuffle=False):
    def fix_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, (PICTURE_SIZE, PICTURE_SIZE))
        image = prep_func(image)
        return image, label

    data = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        data = data.shuffle(2000, seed=RANDOM_NO)

    data = data.map(fix_image)
    data = data.batch(BATCH_NO)
    return data


def create_finetune_net(net_name, label_count):
    NetClass, prep_func = choose_cnn(net_name)
    if NetClass is None:
        return None

    inputs = layers.Input(shape=(PICTURE_SIZE, PICTURE_SIZE, 3))
    base = NetClass(
        weights="imagenet",
        include_top=False,
        input_shape=(PICTURE_SIZE, PICTURE_SIZE, 3),
        pooling="avg"
    )

    for layer in base.layers:
        layer.trainable = False

    for layer in base.layers[-OPEN_LAST_LAYER_COUNT:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    x = base(inputs, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(label_count, activation="softmax")(x)

    final_model = models.Model(inputs, outputs)
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return final_model


def run_finetune(net_name, train_x, train_y, val_x, val_y, test_x, test_y, label_count):
    print("\nFine-tuning:", net_name)

    _, prep_func = choose_cnn(net_name)
    if prep_func is None:
        return None

    train_ready = make_finetune_input(train_x, train_y, prep_func, shuffle=True)
    val_ready = make_finetune_input(val_x, val_y, prep_func)
    test_ready = make_finetune_input(test_x, test_y, prep_func)

    model = create_finetune_net(net_name, label_count)
    if model is None:
        return None

    start_time = time.time()
    history = model.fit(
        train_ready,
        validation_data=val_ready,
        epochs=FINETUNE_TURN,
        verbose=1
    )
    spent_time = (time.time() - start_time) / 60

    pred_prob = model.predict(test_ready, verbose=0)
    pred_label = np.argmax(pred_prob, axis=1)

    acc = accuracy_score(test_y, pred_label)
    pre = precision_score(test_y, pred_label, average="macro", zero_division=0)
    rec = recall_score(test_y, pred_label, average="macro", zero_division=0)
    f1 = f1_score(test_y, pred_label, average="macro", zero_division=0)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))

    row = {
        "experiment": "FineTuning_" + net_name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "time_min": spent_time,
        "epoch": len(history.history["loss"]),
        "feature_dim": "fine_tuning"
    }
    return row


def print_results_table(all_rows):
    all_rows = sorted(all_rows, key=lambda x: x["f1"], reverse=True)

    print("\nSONUC TABLOSU")
    print("-" * 105)
    print(f"{'Deney':55s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Sure':>8s}")
    print("-" * 105)

    for row in all_rows:
        print(
            f"{row['experiment'][:55]:55s} "
            f"{row['accuracy']:8.4f} "
            f"{row['precision']:8.4f} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:8.4f} "
            f"{row['time_min']:8.2f}"
        )


def run_project():
    keras.utils.set_random_seed(RANDOM_NO)

    train_x, train_y, val_x, val_y, test_x, test_y = get_fashion_data()
    label_count = len(LABEL_NAMES)

    feature_store = {}
    table_rows = []

    for net_name in USED_NETS:
        train_feature, test_feature = take_features(net_name, train_x, train_y, test_x, test_y)

        if train_feature is None:
            continue

        train_feature, test_feature = normalize_feature(train_feature, test_feature)
        feature_store[net_name] = {
            "train": train_feature,
            "test": test_feature
        }

    for net_name in USED_NETS:
        if net_name not in feature_store:
            continue

        table_rows.append(run_mlp(
            "Single_" + net_name,
            feature_store[net_name]["train"],
            train_y,
            feature_store[net_name]["test"],
            test_y,
            label_count
        ))

    two_net_tests = [
        ("ResNet50", "MobileNetV2"),
        ("ResNet50", "EfficientNetB0"),
        ("MobileNetV2", "EfficientNetB0")
    ]

    #İki CNN birleşimi
    for pair in two_net_tests:
        if pair[0] not in feature_store or pair[1] not in feature_store:
            continue

        pair_name = pair[0] + "+" + pair[1]

        train_concat = join_features(feature_store, pair, "train")
        test_concat = join_features(feature_store, pair, "test")

        table_rows.append(run_mlp(
            "Concat_2CNN_" + pair_name,
            train_concat,
            train_y,
            test_concat,
            test_y,
            label_count
        ))

        train_weighted = weighted_join(feature_store, pair, "train")
        test_weighted = weighted_join(feature_store, pair, "test")

        table_rows.append(run_mlp(
            "Weighted_2CNN_" + pair_name,
            train_weighted,
            train_y,
            test_weighted,
            test_y,
            label_count
        ))

    #Üç CNN birleşimi
    train_concat = join_features(feature_store, USED_NETS, "train")
    test_concat = join_features(feature_store, USED_NETS, "test")

    table_rows.append(run_mlp(
        "Concat_3CNN_ResNet50+MobileNetV2+EfficientNetB0",
        train_concat,
        train_y,
        test_concat,
        test_y,
        label_count
    ))

    train_weighted = weighted_join(feature_store, USED_NETS, "train")
    test_weighted = weighted_join(feature_store, USED_NETS, "test")

    table_rows.append(run_mlp(
        "Weighted_3CNN_ResNet50+MobileNetV2+EfficientNetB0",
        train_weighted,
        train_y,
        test_weighted,
        test_y,
        label_count
    ))

    if DO_FINE_TUNING:
        for net_name in USED_NETS:
            answer = run_finetune(net_name, train_x, train_y, val_x, val_y, test_x, test_y, label_count)
            if answer is not None:
                table_rows.append(answer)

    print_results_table(table_rows)


if __name__ == "__main__":
    run_project()

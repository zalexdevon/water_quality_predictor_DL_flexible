from Mylib import myfuncs, myclasses, tf_myclasses
import os
import tensorflow as tf


def load_data(data_transformation_path, class_names_path, model_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")
    class_names = myfuncs.load_python_object(class_names_path)
    model = tf.keras.models.load_model(model_path)

    return train_ds, val_ds, class_names, model


def evaluate_model_on_train_val(
    train_ds, val_ds, class_names, model, model_evaluation_on_train_val_path
):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, train_confusion_matrix, val_confusion_matrix = (
        tf_myclasses.ClassifierEvaluator(
            model=model,
            class_names=class_names,
            train_ds=train_ds,
            val_ds=val_ds,
        ).evaluate()
    )
    final_model_results_text += model_results_text  # Thêm đoạn đánh giá vào

    # Lưu lại confusion matrix cho tập train và val
    train_confusion_matrix_path = os.path.join(
        model_evaluation_on_train_val_path, "train_confusion_matrix.png"
    )
    train_confusion_matrix.savefig(
        train_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )
    val_confusion_matrix_path = os.path.join(
        model_evaluation_on_train_val_path, "val_confusion_matrix.png"
    )
    val_confusion_matrix.savefig(
        val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(
        os.path.join(model_evaluation_on_train_val_path, "result.txt"), mode="w"
    ) as file:
        file.write(final_model_results_text)

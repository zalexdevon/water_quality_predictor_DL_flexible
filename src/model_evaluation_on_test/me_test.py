from Mylib import myfuncs, myclasses, tf_myfuncs, tf_myclasses, tf_model_evaluator
import os
import tensorflow as tf


def load_data_for_model_evaluation_on_test(
    test_data_path,
    correction_transformer_path,
    data_transformation_path,
    class_names_path,
    model_path,
):
    test_data = myfuncs.load_python_object(test_data_path)
    correction_transformer = myfuncs.load_python_object(
        os.path.join(correction_transformer_path)
    )
    feature_transformer = myfuncs.load_python_object(
        f"{data_transformation_path}/feature_transformer.pkl"
    )
    target_transformer = myfuncs.load_python_object(
        f"{data_transformation_path}/target_transformer.pkl"
    )
    batch_size = myfuncs.load_python_object(
        f"{data_transformation_path}/batch_size.pkl"
    )
    model = myfuncs.load_python_object(model_path)
    class_names = myfuncs.load_python_object(class_names_path)

    return (
        test_data,
        correction_transformer,
        feature_transformer,
        target_transformer,
        batch_size,
        model,
        class_names,
    )


def transform_test_data(
    test_data,
    correction_transformer,
    feature_transformer,
    target_transformer,
    batch_size,
):
    df_test_corrected = correction_transformer.transform(test_data)
    df_test_feature = feature_transformer.transform(df_test_corrected).astype("float32")
    df_test_target = (
        target_transformer.transform(df_test_corrected)
        .values.reshape(-1)
        .astype("int8")
    )

    test_ds = tf.data.Dataset.from_tensor_slices((df_test_feature, df_test_target))
    test_ds = test_ds.batch(batch_size)
    test_ds = tf_myfuncs.cache_prefetch_tfdataset_2(test_ds)

    return test_ds


def evaluate_model_on_test(test_ds, class_names, model, model_evaluation_on_test_path):
    # Này đánh giá tổng thể nên chắc chắn có scoring ở trong này rồi !!!!!!

    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, test_confusion_matrix = tf_model_evaluator.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_ds=test_ds,
    ).evaluate()
    final_model_results_text += model_results_text  # Thêm đoạn đánh giá vào

    # Lưu lại confusion matrix cho tập test
    test_confusion_matrix_path = os.path.join(
        model_evaluation_on_test_path, "test_confusion_matrix.png"
    )
    test_confusion_matrix.savefig(
        test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(
        os.path.join(model_evaluation_on_test_path, "result.txt"), mode="w"
    ) as file:
        file.write(final_model_results_text)

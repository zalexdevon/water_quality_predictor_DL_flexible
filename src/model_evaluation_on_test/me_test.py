from Mylib import myfuncs, myclasses, tf_myfuncs, tf_myclasses
import os
import tensorflow as tf


def load_data_for_model_evaluation_on_test(
    test_data_path,
    correction_transformer_path,
    transformation_transformer_path,
    class_names_path,
    model_path,
    batch_size_path,
):
    test_data = myfuncs.load_python_object(test_data_path)
    correction_transformer = myfuncs.load_python_object(
        os.path.join(correction_transformer_path)
    )
    transformation_transformer = myfuncs.load_python_object(
        os.path.join(transformation_transformer_path)
    )
    model = myfuncs.load_python_object(model_path)
    class_names = myfuncs.load_python_object(os.path.join(class_names_path))
    batch_size = myfuncs.load_python_object(os.path.join(batch_size_path))

    return (
        test_data,
        correction_transformer,
        transformation_transformer,
        model,
        class_names,
        batch_size,
    )


def transform_test_data(
    test_data, correction_transformer, transformation_transformer, batch_size
):
    # Transform test data
    test_data_corrected = correction_transformer.transform(test_data)
    test_data_transformed = transformation_transformer.transform(test_data_corrected)

    # Chia ra thành features và target
    target_col = myfuncs.get_target_col_from_df_26(test_data_transformed)
    test_features = test_data_transformed.drop(columns=[target_col])
    test_target = test_data_transformed[target_col]

    # Convert thành dataset
    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_target))
    test_ds = test_ds.batch(batch_size)
    test_ds = tf_myfuncs.cache_prefetch_tfdataset_2(test_ds)

    return test_ds


def evaluate_model_on_test(test_ds, class_names, model, model_evaluation_on_test_path):
    # Này đánh giá tổng thể nên chắc chắn có scoring ở trong này rồi !!!!!!

    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, test_confusion_matrix = tf_myclasses.ClassifierEvaluator(
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

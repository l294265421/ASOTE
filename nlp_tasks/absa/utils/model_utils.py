from keras.models import Model


def load_model(model_fun, model_para: tuple, pretrain_model_path: str) -> Model:
    """

    :return:
    """
    model = model_fun(*model_para)
    model.load_weights(pretrain_model_path)
    return model


if __name__ == '__main__':
    print(type(load_model))

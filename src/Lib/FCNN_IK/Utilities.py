# Typing (Support for type hints)
import typing as tp
# Sklearn (Simple and efficient tools for predictive 
# data analysis) [pip3 install scikit-learn]
import sklearn.preprocessing

def Scale_Data(range: tp.List[float], data: tp.List[float]) -> tp.Tuple[sklearn.preprocessing.MinMaxScaler,
                                                                        tp.List[float]]:
    """
    Description:
        Find the scale parameter from the dataset and transform the data using this parameter.

    Args:
        (1) range [Vector<float> 1x2]: Required range of transformed data (min, max).
        (2) data [Vector<float>]: Input data.

    Returns:
        (1) parameter 1 [sklearn.preprocessing.MinMaxScaler(object)]: Class of the min-max scaler.
        (2) parameter 2 [Vector<float>]: Output transformed (scaled) data.
    """

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(range[0], range[1]))

    return (scaler, scaler.fit_transform(data))

def Transform_Data_With_Scaler(scaler: sklearn.preprocessing.MinMaxScaler, data: tp.List[float]) -> tp.List[float]:
    """
    Description:
        Transform of data using an the scale parameter.

    Args:
        (1) scaler [sklearn.preprocessing.MinMaxScaler(object)]: Class of the min-max scaler.
        (2) data [Vector<float>]: Input data.

    Returns:
        (1) parameter [Vector<float>]: Output transformed (scaled) data.
    """

    return scaler.transform(data)

def Inverse_Data_With_Scaler(scaler: sklearn.preprocessing.MinMaxScaler, data: tp.List[float]) -> tp.List[float]:
    """
    Description:
        Inverse transformation (unscaling) of data using the scale parameter.

    Args:
        (1) scaler [sklearn.preprocessing.MinMaxScaler(object)]: Class of the min-max scaler.
        (2) data [Vector<float>]: Input data.

    Returns:
        (1) parameter [Vector<float>]: Output inversed (unscaled) data.
    """

    return scaler.inverse_transform(data)
import random
import numpy as np
import matplotlib.pyplot as plt

from src.utils.data_utils import (
    create_multi_modal_test_data,
    create_test_data
)

random.seed(45)


def test_create_multi_modal_test_data():
    """ 
    Test Create Multimodal Test Data utility for proper output.
    """
    # ensure no figures opened prior to running test
    plt.close('all')

    multimodal_data, modes, outliers, fig = create_multi_modal_test_data(
        loc_range=(0, 100),
        std_range=(1, 5),
        size_range=(500, 2000),
        n_modes=3,
        noise_std=20.0,
        noise_size=100,
        show_fig=False
        )
    
    assert len(multimodal_data) == sum(len(mode) for mode in modes) + len(outliers)
    assert len(modes) == 3
    assert len(plt.get_fignums()) == 0
    for mode in modes:
        assert (np.std(mode) >= 1) and (np.std(mode) <= 20)  # including noise std
        assert (len(mode) >= 500) and (len(mode) <= 2000 + 100 + 5)  # size range max + noise + outlier max
        assert max(mode) <= 100*1.6  # outliers sampled from range up to 60% of mode max



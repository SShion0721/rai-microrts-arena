import numpy as np

from rl_algo_impls.microrts.options import OptionAction, OptionLibrary, OptionName


def test_option_fallback_returns_independently_legal_action():
    library = OptionLibrary([2, 3])
    masks = np.array(
        [
            [False, True, False, False, True],
            [True, False, False, True, False],
        ]
    )

    action = library.to_primitive(OptionAction(OptionName.HARVEST), masks)

    np.testing.assert_array_equal(action, np.array([[1, 2], [0, 1]]))

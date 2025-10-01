import resdp
import pandas as pd
import numpy as np
def gridhead_to_dimens(val):
    return pd.DataFrame(
        val[np.newaxis, 1:4], columns = resdp.DATA_DIRECTORY['DIMENS'].specification.columns
    )



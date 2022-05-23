from collections import OrderedDict
import itertools
search_params = OrderedDict({
    "d_model": [32],
    "q": [8],
    "v": [8],
    "h": [2, 4, 8],
    "N": [2],
    "attention_size": [12],
})

def get_args(args, condi_params):
    if condi_params is None:
        yield args
    else:
        for _params in itertools.product(*condi_params.values()):
            params = {key: _params[idx]
                for idx, key in enumerate(condi_params.keys())}
            print("--------Search Params---------------", params)
            
            for idx, key in enumerate(condi_params.keys()):
                if hasattr(args, key):
                    setattr(args, key, _params[idx])
            yield args
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

def get_args(args, search_params):
    args_lst = []
    for _params in itertools.product(*search_params.values()):
        params = {key: params[idx]
            for idx, key in enumerate(search_params.keys())}
        print("--------Search Params---------------", params)
        
        for idx, key in enumerate(search_params.keys()):
            if hasattr(args, key):
                setattr(args, _params[idx])
        args_lst.append(args)
    return args_lst
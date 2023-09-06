import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings

warnings.simplefilter("ignore", UserWarning)
import pickle

from lama_cleaner.parse_args import parse_args


def entry_point():
    try:
        args = parse_args()
    except Exception:
        with open('args.pkl','rb')as f:
            args = pickle.loads(f.read())
    # To make os.environ["XDG_CACHE_HOME"] = args.model_cache_dir works for diffusers
    # https://github.com/huggingface/diffusers/blob/be99201a567c1ccd841dc16fb24e88f7f239c187/src/diffusers/utils/constants.py#L18
    from lama_cleaner.server import main
    main(args)

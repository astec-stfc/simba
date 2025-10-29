import yaml
import os
from xopt import Xopt

if __name__ == "__main__":
    config = yaml.safe_load(open('./xopt_example/example.yaml'))
    os.makedirs(config['generator']['output_path'], exist_ok=True)
    xo = Xopt(**config)
    xo.run()
    print(xo.data)
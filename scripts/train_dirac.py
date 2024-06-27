import os, sys
import pathlib
from lightning.pytorch.cli import LightningCLI
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )

from  pl_modules.ncsn_module import NCSN_Module


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.experiment_config_file", "data.init_args.experiment_config_file") 
        parser.link_arguments("model.dt", "data.init_args.dt")
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")

def cli_main():
    cli = MyLightningCLI(NCSN_Module, save_config_kwargs={"config_filename": "config.yaml", "overwrite": True}, run=True)


if __name__ == "__main__":
    cli_main()
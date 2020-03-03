import pandas as pd
from .protocols import FragList_LogP, PLLKWE

pd.options.mode.chained_assignment = None  # Help speed up code and drop weird warning pandas can give


class create_relationships:  # Class to generate the different relationship protocols

    def __init__(self, protocol, max_nodes_in_ram, *files):
        # Please note the timer is still under development and testing, time may not be accurate

        self.protocol = protocol  # Define protocols

        self.protocol_dict = {
            "FragList_LogP": FragList_LogP,
            "PLLKWE": PLLKWE
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))  # Start comparing

        self.protocol_dict[self.protocol].__compare__(max_nodes_in_ram, *files)

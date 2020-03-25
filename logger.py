import csv

class NetworkLogger():
    """
    A class to handle the recording, formating, and saving of data from
    the backprop network.
    """
    def __init__(self, filename, fieldnames, path='./'):
        """
        Initialize with a filename, the column names for the csv file,
        and a place to save it.
        """
        self.filename = filename
        self.fieldnames = fieldnames
        self.path = path
        self.data = []

    def reset(self):
        """Dump the data."""
        self.data = []

    def log(self, data):
        """
        Add the provided list of data, which will become a row in the csv file,
        to the running list of data.
        """
        self.data.append(data)

    def write_data(self):
        """
        Write the data to a csv file at the designated location.
        """
        with open(self.path + self.filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for datum in self.data:
                writer.writerow(dict([(self.fieldnames[i], datum[i]) for i in range(0, len(self.fieldnames))]))

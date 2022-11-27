from os.path import join, abspath
import xml.etree.ElementTree as ET

from winnow.storage.simple_repr_storage import SimpleReprStorage


class ReprStorage:
    """Persistent storage of various intermediate representations."""

    def __init__(self, directory, storage_factory=SimpleReprStorage):
        """Create new storage instance.

        Args:
            directory (String): Directory in which all representations will be stored.
        """
        kwargs = {}
        kwargs["save"] = save_func
        kwargs["load"] = load_func
        kwargs["repr_suffix"] = "_data.xml"
        self.directory = abspath(directory)
        self.data = storage_factory(join(self.directory, "data"), **kwargs)

    def __repr__(self):
        return f"ReprStorage('{self.directory}')"

    def close(self):
        """Release any underlying resources (close database connections, etc.)."""
        self.data.close()


def save_func(file, value):
    root = ET.Element("root")
    for key, val in value.items():
        ET.SubElement(root, key, {'data': val})
    tree = ET.ElementTree(root)
    tree.write(file)


def load_func(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ret_val = {}
    for child in root:
        ret_val[child.tag] = child.attrib['data']
    return ret_val

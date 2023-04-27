"""bdd dataset."""

import tensorflow_datasets as tfds
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features = tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape = (720, 1280, 3)),
                'label': tfds.features.Image(shape = (720, 1280, 1)),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples('bdd100k/images/100k/train', 0, 60000),
            'validate': self._generate_examples('bdd100k/images/100k/val', 0, 10000),
            'test': self._generate_examples('bdd100k/images/100k/train', 60000, 10000),
        }

    def _generate_examples(self, path, start, amount):
        chunks = path.split('/')
        labelPath = "bdd100k/labels/lane/masks/" + chunks[len(chunks) - 1]
        count = 0
        number = 0

        for f in Path(path).glob('*.jpg'):
            if number < start:
                number += 1
                continue

            if count >= amount:
                break

            count += 1

            yield f.stem, {
                'image': f,
                'label': Path(labelPath + "/" + f.stem + ".png"),
            }
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
                'image': tfds.features.Image(shape = (1280, 720, 3)),
                'label': tfds.features.Image(shape = (1280, 720, 1)),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples('bdd100k/images/train'),
            'validate': self._generate_examples('bdd100k/images/val'),
            'test': self._generate_examples('bdd100k/images/test'),
        }

    def _generate_examples(self, path):
        labelPath = ""
        chunks = path.split('/')
        used = []

        for i in range(len(chunks) - 2):
            labelPath = labelPath + chunks[i] + "/"

        for f in Path(path).glob('*.jpg'):
            if f.stem in used:
                continue
                
            used.append(f.stem)
            
            yield f.stem, {
                'image': f,
                'label': Path(labelPath + "labels/" + chunks[len(chunks) - 1] + "/" + f.stem + ".png"),
            }
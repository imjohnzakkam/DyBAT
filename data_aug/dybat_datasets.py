from torchvision.transforms import transforms
from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection

class DyBATDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_turbo_transform(crop_size):
        """Return a set of data augmentation transformations."""

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        data_transforms = transforms.Compose([transforms.Resize((crop_size, crop_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        return data_transforms

    def get_dataset(self, name, crop_size):
        valid_datasets = {'cifar10': lambda: (datasets.CIFAR10(self.root_folder, train=True, transform=self.get_turbo_transform(crop_size), download=True), 
                                              datasets.CIFAR10(self.root_folder, train=False, transform=self.get_turbo_transform(crop_size), download=True)),
                            'cifar100': lambda: (datasets.CIFAR100(self.root_folder, train=True, transform=self.get_turbo_transform(crop_size), download=True), 
                                               datasets.CIFAR100(self.root_folder, train=False, transform=self.get_turbo_transform(crop_size), download=True)),
                            'stl10': lambda: (datasets.STL10(self.root_folder, split='train', transform=self.get_turbo_transform(crop_size), download=True), 
                                              datasets.STL10(self.root_folder, split='test', transform=self.get_turbo_transform(crop_size), download=True)),
                            'stanfordcars': lambda: (datasets.StanfordCars(self.root_folder, split='train', transform=self.get_turbo_transform(crop_size), download=True),
                                                     datasets.StanfordCars(self.root_folder, split='test', transform=self.get_turbo_transform(crop_size), download=True)),
                            'oxfordiiitpet': lambda: (datasets.OxfordIIITPet(self.root_folder, split='trainval', target_types='category', transform=self.get_turbo_transform(crop_size), download=True),
                                                      datasets.OxfordIIITPet(self.root_folder, split='test', target_types='category', transform=self.get_turbo_transform(crop_size), download=True)),
                            'mnist': lambda: (datasets.MNIST(self.root_folder, train=True, transform=self.get_turbo_transform(crop_size), download=True), 
                                              datasets.MNIST(self.root_folder, train=False, transform=self.get_turbo_transform(crop_size), download=True)),
                            'flowers102': lambda: (datasets.Flowers102(self.root_folder, split='train', transform=self.get_turbo_transform(crop_size), download=True),
                                                   datasets.Flowers102(self.root_folder, split='test', transform=self.get_turbo_transform(crop_size), download=True))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

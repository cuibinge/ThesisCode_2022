class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = r'./LEVIR'
        elif data_name == 'Glacier':
            self.root_dir = './Glacier/'
        elif data_name == 'new_glacier':
            self.root_dir = './new_glacier/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

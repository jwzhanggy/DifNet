'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.setting import setting


class SettingCV(setting):
    fold = None
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result, test_acc = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return test_acc

        
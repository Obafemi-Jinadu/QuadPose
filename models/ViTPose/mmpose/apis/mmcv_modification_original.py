import mmcv
import numpy as np
import torch
import platform
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, DataContainer
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         get_dist_info,BaseRunner)
from mmcv.utils import digit_version 


from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger


from mmcv.runner import get_host_info, save_checkpoint
import time
import time
import os.path as osp

import warnings
import shutil
import time



class EpochBasedRunnerModified(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        #super().run_iter(data_batch, train_mode, **kwargs)
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            #print("question! ", dir(self.model))
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        #super().train( data_loaders, **kwargs)
        self.model.train()
        self.mode = 'train'
        self.data_loaders = data_loaders
        
        self._max_iters = self._max_epochs * (len(self.data_loaders[0])+len(self.data_loaders[1]))
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for self.data_loader in self.data_loaders:
            for i, data_batch in enumerate(self.data_loader):
                #print('data_ba ', i,data_batch)
                #self.metas_femi = data_batch['img_metas'].data
                #self.cat_ids = torch.tensor([i['category_id'] for i in self.metas_femi[0]])
                #print("data batch here: ", len(data_batch), self.cat_ids)
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter( data_batch,  train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        #super.val( data_loader, **kwargs)
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        #super().run( data_loaders, workflow, max_epochs, **kwargs)
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        #print('LENs', len(data_loaders) , len(workflow))
        #assert len(data_loaders) == len(workflow)  #important please uncomment when done!!!!!!!
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])  

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs+1):
                	
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    #print('what is IIII: ',i)
                    if mode == 'train' and len(data_loaders)>2:
                        if self.epoch <150:
                    	
                            epoch_runner([data_loaders[0],data_loaders[1]], **kwargs)
                        	
                        elif self.epoch >=150 and self.epoch <250:
                            epoch_runner([data_loaders[0],data_loaders[1], data_loaders[2]], **kwargs) 
                        else:
                        	epoch_runner([data_loaders[i],data_loaders[i+1], data_loaders[i+2], data_loaders[i+3]], **kwargs) 
                    elif mode == 'train' and len(data_loaders)==2:
                        epoch_runner([data_loaders[i],data_loaders[i+1]], **kwargs)

                    if mode == 'val':
                        
                        epoch_runner(data_loaders[i], **kwargs)
                    

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        #super().save_checkpoint(
         #               out_dir,
          #              filename_tmpl,
           #             save_optimizer,
            #            meta,
             #           create_symlink)
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)




              
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)




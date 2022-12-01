from run_manager import *
from gen_adv import *
from distillation import *
from tqdm import tqdm
import random
from utils import *

class EnsembleInOneRunManager:

    def __init__(self, path, super_net, run_config: RunConfig):
        self.warmup = False
        self.path_wb = dict()
        self.run_manager = RunManager(path, super_net, run_config, True)
        self.net = super_net

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']

    def backup_path(self):
        for _, (name, param) in enumerate(self.net.named_parameters()):
            if 'AP_path_wb' in name:
                self.path_wb[name] = param.data.clone().detach()
    
    def resume_path(self):
        for _, (name, param) in enumerate(self.net.named_parameters()):
            if 'AP_path_wb' in name:
                param.data.copy_(self.path_wb[name])

    def check_same_path(self):
        is_same = True
        assert len(self.path_wb) > 0
        for _, (name, param) in enumerate(self.net.named_parameters()):
            if name in self.path_wb:
                if False in (self.path_wb[name] == param.data):
                    is_same = False
                    break
        return is_same
        

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        #self.net.set_chosen_op_active()
        self.net.random_reset_binary_gates()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=False, return_top5=True)
        # unused modules back
        self.net.unused_modules_back()
        return valid_res

    def warm_up(self, warmup_epochs=200):
        nBatch = len(self.run_manager.run_config.train_loader)
        T_total = warmup_epochs * nBatch

        for epoch in range(0, warmup_epochs):
            data_loader = self.run_manager.run_config.train_loader
            new_lr = self.run_manager.adjust_learning_rate(epoch, nBatch=nBatch)
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')

            batch_time = AverageMeter()                                                                     
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()
                                                                                                            
            end = time.time()
            for i, data in enumerate(data_loader):
                images, labels = data
                data_time.update(time.time() - end)
                # lr
                
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                
                # select one path and compute output
                self.net.random_reset_binary_gates()
                self.net.unused_modules_off()
                output = self.run_manager.net(images)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)

                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                self.run_manager.net.zero_grad()
                loss.backward()
                self.run_manager.optimizer.step()
                self.net.unused_modules_back()
                
                # elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                            'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                            'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'.\
                            format(epoch + 1, i, nBatch - 1, batch_time = batch_time, data_time = data_time,
                                   losses = losses, top1 = top1, top5 = top5, lr = new_lr)
                    self.run_manager.write_log(batch_log, 'warmup')
            top1, top5 = self.validate()
            val_log = 'Warmup Valid [{0}/{1}]\t top-1 acc {2:.3f}\t top5 acc {3:.3f}\t'.\
                    format(epoch + 1, warmup_epochs, top1, top5)
            self.run_manager.write_log(val_log, 'valid')
            self.warmup = epoch + 1 < warmup_epochs

            state_dict = self.net.state_dict()
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')


    def train(self, opt):
        nBatch = len(self.run_manager.run_config.distill_loader) #int(50000/128) #len(data_loader.seed)
        T_total = opt.n_epochs * nBatch
        
        for epoch in range(0, opt.n_epochs):
            data_loader = self.run_manager.run_config.distill_loader
            new_lr = self.run_manager.adjust_learning_rate(epoch, nBatch=nBatch)
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, data in enumerate(data_loader):
                si, sl, ti, tl = data
                data_time.update(time.time() - end)

                si, sl, ti, tl = si.to(self.run_manager.device), sl.to(self.run_manager.device), \
                            ti.to(self.run_manager.device), tl.to(self.run_manager.device)

                if not opt.distill_fix:
                    if opt.limit_max_layer <= 0:
                        distill_layer = random.randint(1, opt.layers)
                    else:
                        distill_layer = random.randint(1, opt.limit_max_layer)
                else:
                    distill_layer = opt.distill_layer

                distill_batch_cur = min(opt.distill_batch, np.power(opt.aug_num, distill_layer))
                # sample a batch of paths from the EIO Net
                models = []
                sub_models = []
                for mi in range(distill_batch_cur):
                    self.net.random_reset_binary_gates()
                    sample = self.net.module_active_index
                    sub_sample = sample[:(distill_layer)]
                    while sub_sample in sub_models:
                        self.net.random_reset_binary_gates()
                        sample = self.net.module_active_index
                        sub_sample = sample[:(distill_layer)]
                    sub_models.append(sub_sample)
                    models.append(sample)

                distilled_data_list = []
                for m in models:
                    self.net.fill_active_index(m)
                    self.net.unused_modules_off()
                    temp = linf_distillation(self.net, si, ti, distill_layer, eps=opt.distill_eps, \
                            alpha=opt.distill_alpha, steps=opt.distill_steps, train_mode=opt.distill_train_mode)
                    distilled_data_list.append(temp)
                    self.net.unused_modules_back()
                self.run_manager.net.zero_grad()

                if opt.adv_train:
                    adv_data_list = []
                    for m in models:
                        self.net.fill_active_index(m)
                        self.net.unused_modules_off()
                        adv_data = gen_adv_iFGSM(self.net, self.run_manager.criterion, si, sl, \
                                eps=8./255., alpha=2./255., iteration=10)
                        self.net.unused_modules_back()
                        adv_data_list.append(adv_data)
                    self.run_manager.net.zero_grad()

                for mi, m in enumerate(models):
                    loss = 0
                    self.net.fill_active_index(m)
                    self.net.unused_modules_off()
                    for dj, distilled_data in enumerate(distilled_data_list):
                        distilled_data_train = distilled_data
                        if mi == dj:
                            continue
                        
                        outputs = self.net(distilled_data_train)
                        if opt.label_smoothing == 0: 
                            loss += self.run_manager.criterion(outputs, sl)
                        else:
                            loss += cross_entropy_with_label_smoothing(outputs, sl, opt.label_smoothing)
                        acc1, acc5 = accuracy(outputs, sl, topk=(1, 5))
                        top1.update(acc1[0], outputs.size(0))
                        top5.update(acc5[0], outputs.size(0))

                    if opt.adv_train:
                        adv_data = adv_data_list[mi]
                        outputs = self.net(adv_data)
                        loss += self.run_manager.criterion(outputs, sl)
                        
                    if opt.adv_train:
                        losses.update(loss.item() / (distill_batch_cur), 1)
                    else:
                        losses.update(loss.item() / (distill_batch_cur - 1), 1)
                    loss = loss / distill_batch_cur
                    loss.backward()
                    self.net.unused_modules_back()

                self.run_manager.optimizer.step()
                self.run_manager.net.zero_grad()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                            'Loss {losses.avg:.4f}\t' \
                            'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                            'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                            format(epoch + 1, i, nBatch - 1, 
                               losses=losses, top1=top1, top5=top5, lr=new_lr)
                    self.run_manager.write_log(batch_log, 'train')

            top1, top5 = self.validate()
   
            val_log = 'Train Valid [{0}/{1}]\ttop-1 acc {2:.3f}\ttop-5 acc {3:.3f}\t'.\
                  format(epoch + 1, opt.n_epochs, top1, top5) 
            self.run_manager.write_log(val_log, 'valid')
            self.train_epoch = epoch + 1 < opt.n_epochs

            state_dict = self.net.state_dict()
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'train': self.train_epoch,
            }
            if self.train_epoch:
                checkpoint['train_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='model.pth.tar')

